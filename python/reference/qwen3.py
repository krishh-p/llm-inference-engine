"""
Small Qwen3 reference used to check the rust version

Tokens are rows, so x is [n, hidden]. Qwen3-1.7b uses hidden size 2048,
16 query heads, 8 KV heads, head size 128, MLP size 6144, and 28 layers.
"""

import torch
import torch.nn.functional as F

MAX_LOGIT_DIFF = 1e-3


def save_trace(trace, name, tensor):
    if trace is not None:
        trace[name] = tensor.detach()


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


def rope_tables(positions, head_size, theta):
    half = head_size // 2
    freq = 1.0 / theta ** (torch.arange(half, device=positions.device).float() / half)
    angle = positions.float()[:, None] * freq[None, :]  # [n, 64]
    return angle.cos(), angle.sin()


def apply_rope(x, cos, sin):
    half = x.shape[-1] // 2

    a = x[..., :half]
    b = x[..., half:]
    return torch.cat([a * cos - b * sin, b * cos + a * sin], dim=-1)


def rope(x, positions, theta):
    """
    apply rope to q or k

    dimension i is paired with dimension i + 64
    x: [heads, n, 128]
    positions: [n]
    """
    cos, sin = rope_tables(positions, x.shape[-1], theta)
    return apply_rope(x, cos.to(x.dtype), sin.to(x.dtype))


def attention(x, positions, cached, weights, p, cfg, trace=None, trace_prefix=""):
    """
    Run attention for the new tokens

    x: [n_new, 2048]
    returns: [n_new, 2048] and the updated cache
    """
    heads = cfg.num_attention_heads  # 16
    kv_heads = cfg.num_key_value_heads  # 8

    q = split_heads(x @ weights[p + "q_proj.weight"].T, heads)  # [16, n_new, 128]
    k = split_heads(x @ weights[p + "k_proj.weight"].T, kv_heads)  # [8, n_new, 128]
    v = split_heads(x @ weights[p + "v_proj.weight"].T, kv_heads)  # [8, n_new, 128]
    save_trace(trace, trace_prefix + "q", q)
    save_trace(trace, trace_prefix + "k", k)
    save_trace(trace, trace_prefix + "v", v)

    q = rms_norm(q, weights[p + "q_norm.weight"], cfg.rms_norm_eps)
    k = rms_norm(k, weights[p + "k_norm.weight"], cfg.rms_norm_eps)
    save_trace(trace, trace_prefix + "q_norm", q)
    save_trace(trace, trace_prefix + "k_norm", k)

    q = rope(q, positions, cfg.rope_theta)
    k = rope(k, positions, cfg.rope_theta)
    save_trace(trace, trace_prefix + "q_rope", q)
    save_trace(trace, trace_prefix + "k_rope", k)

    # add this token's k and v to the cache during decode
    if cached is not None:
        save_trace(trace, trace_prefix + "cache_k_in", cached[0])
        save_trace(trace, trace_prefix + "cache_v_in", cached[1])
        k = torch.cat([cached[0], k], dim=1)  # [8, n, 128]
        v = torch.cat([cached[1], v], dim=1)
    cached = (k, v)
    save_trace(trace, trace_prefix + "cache_k_out", k)
    save_trace(trace, trace_prefix + "cache_v_out", v)

    # each KV head is shared by two query heads
    group = heads // kv_heads
    k = k.repeat_interleave(group, dim=0)  # [16, n, 128]
    v = v.repeat_interleave(group, dim=0)

    # rows are new tokens, columns are the tokens they attend to
    scores = q @ k.transpose(-1, -2) / q.shape[-1] ** 0.5  # [16, n_new, n]

    # hide future tokens during prefill. decode has nothing to hide
    key_positions = torch.arange(k.shape[1], device=x.device)
    future = key_positions[None, :] > positions[:, None]  # [n_new, n]
    scores = scores.masked_fill(future, float("-inf"))

    # hugging face does the softmax in float32 too
    attn = F.softmax(scores, dim=-1, dtype=torch.float32).to(q.dtype)
    z = attn @ v  # weighted sum of values, [16, n_new, 128]

    out = merge_heads(z) @ weights[p + "o_proj.weight"].T  # [n_new, 2048]
    save_trace(trace, trace_prefix + "attention_output", out)
    return out, cached


def mlp(x, weights, p):
    # MLP(x) = W_down @ (SiLU(W_gate @ x) ⊙ (W_up @ x))
    gate = F.silu(x @ weights[p + "gate_proj.weight"].T)  # [n_new, 6144]
    up = x @ weights[p + "up_proj.weight"].T  # [n_new, 6144]
    return (gate * up) @ weights[p + "down_proj.weight"].T  # [n_new, 2048]


def block(x, positions, cached, weights, i, cfg, trace=None):
    # pre-norm attention, then pre-norm mlp
    p = f"model.layers.{i}."
    trace_prefix = f"layer.{i}."
    save_trace(trace, trace_prefix + "input", x)

    h = rms_norm(x, weights[p + "input_layernorm.weight"], cfg.rms_norm_eps)
    save_trace(trace, trace_prefix + "input_norm", h)
    attn_out, cached = attention(
        h,
        positions,
        cached,
        weights,
        p + "self_attn.",
        cfg,
        trace,
        trace_prefix,
    )
    x = x + attn_out

    h = rms_norm(x, weights[p + "post_attention_layernorm.weight"], cfg.rms_norm_eps)
    save_trace(trace, trace_prefix + "mlp_input", h)
    mlp_out = mlp(h, weights, p + "mlp.")
    save_trace(trace, trace_prefix + "mlp_output", mlp_out)
    x = x + mlp_out
    save_trace(trace, trace_prefix + "output", x)

    return x, cached


def forward(tokens, cache, weights, cfg, trace=None, trace_layers=(0,)):
    """Run new tokens through the model and predict what comes next

    tokens: [n_new]
    logits: [vocab]
    """
    # continue positions from the end of the cache
    start = cache[0][0].shape[1] if cache else 0
    positions = torch.arange(start, start + len(tokens), device=tokens.device)
    save_trace(trace, "positions", positions)

    x = weights["model.embed_tokens.weight"][tokens]  # [n_new, 2048]
    save_trace(trace, "embedding", x)

    new_cache = []
    for i in range(cfg.num_hidden_layers):
        cached = cache[i] if cache else None
        layer_trace = trace if trace is not None and i in trace_layers else None
        x, cached = block(x, positions, cached, weights, i, cfg, layer_trace)
        new_cache.append(cached)

    # only the last row predicts the next token
    h = rms_norm(x[-1], weights["model.norm.weight"], cfg.rms_norm_eps)  # [2048]
    logits = weights["lm_head.weight"] @ h  # [vocab]
    save_trace(trace, "final_norm", h)
    save_trace(trace, "logits", logits)

    return logits, new_cache


def prefill(tokens, weights, cfg, trace=None, trace_layers=(0,)):
    # run the whole prompt and build the first cache
    return forward(tokens, [], weights, cfg, trace, trace_layers)


def decode_step(token, cache, weights, cfg, trace=None, trace_layers=(0,)):
    # run one token against everything in the cache
    n = cache[0][0].shape[1]
    logits, cache = forward(token, cache, weights, cfg, trace, trace_layers)

    # every layer should grow by one token
    for i, (k, v) in enumerate(cache):
        assert k.shape[1] == n + 1, f"layer {i}: k has {k.shape[1]} tokens, not {n + 1}"
        assert v.shape[1] == n + 1, f"layer {i}: v has {v.shape[1]} tokens, not {n + 1}"

    return logits, cache
