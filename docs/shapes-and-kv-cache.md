# Tensor shapes and KV cache memory

The Python reference leaves out the batch dimension. Tokens are rows, so the
main activation shape is `[tokens, hidden]`.

Qwen3-1.7B uses:

- hidden size: 2048
- query heads: 16
- KV heads: 8
- head size: 128
- MLP size: 6144
- layers: 28
- vocabulary size: 151936

## Prefill

For a prompt with `n` tokens:

| Tensor | Shape |
| --- | --- |
| token IDs | `[n]` |
| embeddings | `[n, 2048]` |
| Q | `[16, n, 128]` |
| K, V | `[8, n, 128]` |
| repeated K, V for attention | `[16, n, 128]` |
| attention scores | `[16, n, n]` |
| attention output | `[n, 2048]` |
| MLP gate and up | `[n, 6144]` |
| layer output | `[n, 2048]` |
| KV cache per layer | two tensors of `[8, n, 128]` |
| next-token logits | `[151936]` |

The repeated K and V tensors are used by grouped-query attention. The cache
keeps the smaller 8-head tensors.

## Decode

If the cache already contains `n` tokens, one decode step has:

| Tensor | Shape |
| --- | --- |
| new token ID | `[1]` |
| embedding | `[1, 2048]` |
| Q | `[16, 1, 128]` |
| new K, V | `[8, 1, 128]` |
| updated K, V cache | `[8, n + 1, 128]` |
| attention scores | `[16, 1, n + 1]` |
| layer output | `[1, 2048]` |
| next-token logits | `[151936]` |

Prefill and decode use the same forward path. Prefill passes many tokens with an
empty cache; decode passes one token with an existing cache.

## KV cache memory

For batch size 1:

```text
elements = 2 * layers * kv_heads * sequence_length * head_size
         = 2 * 28 * 8 * sequence_length * 128
         = 57,344 * sequence_length
```

The `2` is for K and V. Multiply by the bytes per element and by the batch size.

| Sequence length | bfloat16 / float16 | float32 |
| ---: | ---: | ---: |
| 1,024 | 112 MiB | 224 MiB |
| 4,096 | 448 MiB | 896 MiB |
| 8,192 | 896 MiB | 1.75 GiB |
| 32,768 | 3.5 GiB | 7 GiB |

This only counts the KV cache. Model weights, activations, temporary attention
tensors and allocator overhead are separate.
