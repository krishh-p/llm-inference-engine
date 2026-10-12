"""
mall Qwen3 reference used to check the rust version

Tokens are rows, so x is [n, hidden]. Qwen3-1.7b uses hidden size 2048,
16 query heads, 8 KV heads, head size 128, MLP size 6144, and 28 layers.
"""

import torch
import torch.nn.functional as F


def rms_norm(x, weight, eps):
    # keep this part in float32 so bf16 rounding doesn't throw off the comparison
    xf = x.float()
    xf = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)
    return xf.to(x.dtype) * weight


def split_heads(x, heads):
    # [n, heads * 128] -> [heads, n, 128]
    n = x.shape[0]
    return x.view(n, heads, -1).transpose(0, 1)


def merge_heads(x):
    # [heads, n, 128] -> [n, heads * 128]
    heads, n, head_size = x.shape
    return x.transpose(0, 1).reshape(n, heads * head_size)


def rope(x, positions, theta):
    """
    apply rope to q or k

    x: [heads, n, 128]
    positions: [n]
    """
    half = x.shape[-1] // 2

    freq = 1.0 / theta ** (torch.arange(half, device=x.device).float() / half)
    angle = positions.float()[:, None] * freq[None, :]  # [n, 64]
    cos = angle.cos().to(x.dtype)
    sin = angle.sin().to(x.dtype)

    a = x[..., :half]
    b = x[..., half:]
    return torch.cat([a * cos - b * sin, b * cos + a * sin], dim=-1)


def attention(x, positions, cached, weights, p, cfg):
    """
    Run attention for the new tokens

    x: [n_new, 2048]
    returns: [n_new, 2048] and the updated cache
    """
    heads = cfg.num_attention_heads # 16
    kv_heads = cfg.num_key_value_heads  # 8

    q = split_heads(x @ weights[p + "q_proj.weight"].T, heads) # [16, n_new, 128]
    k = split_heads(x @ weights[p + "k_proj.weight"].T, kv_heads) # [8, n_new, 128]
    v = split_heads(x @ weights[p + "v_proj.weight"].T, kv_heads) # [8, n_new, 128]

    q = rms_norm(q, weights[p + "q_norm.weight"], cfg.rms_norm_eps)
    k = rms_norm(k, weights[p + "k_norm.weight"], cfg.rms_norm_eps)

    q = rope(q, positions, cfg.rope_theta)
    k = rope(k, positions, cfg.rope_theta)

    # add this token's k and v to the cache during decode
    if cached is not None:
        k = torch.cat([cached[0], k], dim=1)  # [8, n, 128]
        v = torch.cat([cached[1], v], dim=1)
    cached = (k, v)

    # each KV head is shared by two query heads
    group = heads // kv_heads
    k = k.repeat_interleave(group, dim=0)  # [16, n, 128]
    v = v.repeat_interleave(group, dim=0)

    # rows are new tokens, columns are the tokens they attend to
    scores = q @ k.transpose(-1, -2) / q.shape[-1] ** 0.5 # [16, n_new, n]

    # hide future tokens during prefill. decode has nothing to hide
    key_positions = torch.arange(k.shape[1], device=x.device)
    future = key_positions[None, :] > positions[:, None] # [n_new, n]
    scores = scores.masked_fill(future, float("-inf"))

    
    # hugging face does the softmax in float32 too
    attn = F.softmax(scores, dim=-1, dtype=torch.float32).to(q.dtype)
    z = attn @ v  # weighted sum of values, [16, n_new, 128]


    out = merge_heads(z) @ weights[p + "o_proj.weight"].T # [n_new, 2048]
    return out, cached


def mlp(x, weights, p):
    # MLP(x) = W_down @ (SiLU(W_gate @ x) ⊙ (W_up @ x))
    gate = F.silu(x @ weights[p + "gate_proj.weight"].T)    # [n_new, 6144]
    up = x @ weights[p + "up_proj.weight"].T                # [n_new, 6144]
    return (gate * up) @ weights[p + "down_proj.weight"].T  # [n_new, 2048]


def block(x, positions, cached, weights, i, cfg):
    # pre-norm attention, then pre-norm mlp
    p = f"model.layers.{i}."

    h = rms_norm(x, weights[p + "input_layernorm.weight"], cfg.rms_norm_eps)
    attn_out, cached = attention(h, positions, cached, weights, p + "self_attn.", cfg)
    x = x + attn_out

    h = rms_norm(x, weights[p + "post_attention_layernorm.weight"], cfg.rms_norm_eps)
    x = x + mlp(h, weights, p + "mlp.")

    return x, cached


def forward(tokens, cache, weights, cfg):
    """Run new tokens through the model and predict what comes next

    tokens: [n_new]
    logits: [vocab]
    """
    # continue positions from the end of the cache
    start = cache[0][0].shape[1] if cache else 0
    positions = torch.arange(start, start + len(tokens), device=tokens.device)

    x = weights["model.embed_tokens.weight"][tokens] # [n_new, 2048]

    new_cache = []
    for i in range(cfg.num_hidden_layers):
        cached = cache[i] if cache else None
        x, cached = block(x, positions, cached, weights, i, cfg)
        new_cache.append(cached)

    # only the last row predicts the next token
    h = rms_norm(x[-1], weights["model.norm.weight"], cfg.rms_norm_eps) # [2048]
    logits = weights["lm_head.weight"] @ h # [vocab]

    return logits, new_cache


def prefill(tokens, weights, cfg):
    # run the whole prompt and build the first cache
    return forward(tokens, [], weights, cfg)


def decode_step(token, cache, weights, cfg):
    # run one token against everything in the cache
    n = cache[0][0].shape[1]
    logits, cache = forward(token, cache, weights, cfg)

    # every layer should grow by one token
    for i, (k, v) in enumerate(cache):
        assert k.shape[1] == n + 1, f"layer {i}: k has {k.shape[1]} tokens, not {n + 1}"
        assert v.shape[1] == n + 1, f"layer {i}: v has {v.shape[1]} tokens, not {n + 1}"

    return logits, cache
