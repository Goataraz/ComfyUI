# Tensor Parallelism in ComfyUI

ComfyUI supports Tensor Parallelism (TP) to split large diffusion models across multiple GPUs, enabling inference on models that exceed a single GPU's VRAM.

## How It Works

Tensor Parallelism shards linear layers across GPUs using the Megatron-LM pattern:

- **Column-parallel layers** split the output dimension across GPUs (e.g., `to_q`, `to_k`, `to_v` projections in attention blocks)
- **Row-parallel layers** split the input dimension and use an all-reduce to combine results (e.g., `to_out` projections)

Each GPU holds a shard of the model and cooperates on every forward pass via NCCL all-reduce. This is model-parallelism — distinct from ComfyUI's built-in multi-GPU work-splitting (which deep-clones full models to each GPU for batch parallelism).

## Supported Models

| Model | Class Name | TP Target Prefixes |
|-------|-----------|-------------------|
| Flux | `Flux` | `double_blocks`, `single_blocks` (packed double-stream QKV; fused `linear1` stays) |
| SD3 / MMDiT | `OpenAISignatureMMDITWrapper` | `joint_blocks` (packed `attn.qkv`; MLP shards) |
| Cosmos T2V/I2V | `GeneralDIT` | `blocks` (MLP-only; FA/CA stay replicated) |
| Cosmos Predict2 / Anima | `MiniTrainDIT` | `blocks` |
| HiDream Image | `HiDreamImageTransformer2DModel` | `double_stream_blocks`, `single_stream_blocks` |
| QwenImage | `QwenImageTransformer2DModel` | `transformer_blocks` |
| Qwen25 7B text encoder | `Llama2` | `layers` |
| WanVideo (T2V/I2V + subclasses) | `WanModel` | `blocks` |
| CausalWan | `CausalWanModel` | inherits `WanModel` (`blocks`) |
| LTXV / LTXAV | `LTXVModel` | `transformer_blocks` (MLP-only; FA/CA stay replicated) |
| HunyuanVideo (+ I2V / 1.5 / Image 2.1) | `HunyuanVideo` | `double_blocks`, `single_blocks` (packed double-stream QKV; fused `linear1` stays) |
| Chroma | `Chroma` | `double_blocks`, `single_blocks` (same packed QKV as Flux) |
| ACE-Step 1.0 | `ACEStepTransformer2DModel` | `transformer_blocks` (head-split; conv FF unreplicated) |
| ACE-Step 1.5 | `AceStepConditionGenerationModel` | `decoder.layers` (head-split + GQA; lyric encoder excluded) |
| Lumina NextDiT / Z-Image | `NextDiT` | `layers`, `noise_refiner`, `context_refiner`, `siglip_refiner` (packed GQA `attention.qkv`) |
| Ideogram 4 | `Ideogram4Transformer` | `layers` (packed `attention.qkv`; unfused SwiGLU shards) |
| MiniMax H3 | `MiniMaxH3Model` | `blocks` (packed QKV + packed SwiGLU, head-split) |
| JoyImage | `JoyImageTransformer3DModel` | `double_blocks` (packed `img_attn_qkv` / `txt_attn_qkv`) |
| Lens | `LensTransformer2DModel` | `transformer_blocks` (packed `img_qkv` / `txt_qkv`; unfused SwiGLU) |
| PixelDiT | `PixDiT_T2I` | `patch_blocks`, `pixel_blocks` (packed `qkv_x`/`qkv_y` + pixel `qkv`) |

Head-split models (heads divided by world_size): QwenImage, MiniTrainDIT, WanModel (and CausalWan via MRO), HiDreamImageTransformer2DModel, ACE-Step 1.0/1.5, MiniMax H3, Flux / HunyuanVideo / Chroma (double-stream only; `SingleStreamBlock.num_heads` stays full because fused `linear1` is unreplicated), SD3 / MMDiT, Ideogram 4, JoyImage, Lens, NextDiT (`n_local_heads` / `n_local_kv_heads`; root `n_heads` stays), PixelDiT (`MMDiTJointAttention` / `RotaryAttention`; `PiTBlock.num_heads` stays for RoPE dim).
GeneralDIT is **not** head-split — attention projections are excluded, so only MLP linears shard.
Wan / HiDream full-dim QK RMSNorms are sliced to the local shard; QwenImage / Cosmos / MiniMax per-head norms stay replicated.

CausalWan keeps `num_heads` on the outer model (KV cache) and on `WanAttentionBlock` (cross-attn). After Attention modules are head-split, leftover `heads` / `n_heads` / `num_heads` that still equal the pre-split count are synced to the local shard.

HiDream O1 is **not** supported — its integrated Llama2 LLM receives input from non-TP visual/x_embedder components and needs a full-model TP strategy that shards the vision encoder too. The `HiDreamO1Transformer` class is in `TP_UNSUPPORTED` and `get_tp_targets` returns `[]` for it.

MiniMax H3 uses packed colwise sharding: `qkv_proj` is `[Q|K|V]` and `mlp.fc1` is fused SwiGLU `[gate|up]`. Each pack is head-split independently so `.split(heads * head_dim)` and `chunk(2)` stay correct.

Model matching uses Python's MRO (Method Resolution Order) walk, so subclasses of supported models (e.g., `Anima(MiniTrainDIT)`, `CausalWanModel(WanModel)`) automatically inherit TP support.

## Launch Instructions

### Prerequisites

- 2+ NVIDIA GPUs of the same type
- NCCL-compatible interconnect (NVLink recommended, PCIe works for smaller models)
- PyTorch with CUDA and `torchrun` available

### Launch Command

```bash
torchrun --nproc_per_node=N main.py \
    --tensor-parallel \
    --port 8188 \
    [other ComfyUI flags]
```

Where `N` is the number of GPUs.

Example for 2 GPUs:

```bash
torchrun --nproc_per_node=2 main.py \
    --tensor-parallel \
    --port 8188 \
    --force-fp16 \
    --bf16-vae \
    --bf16-unet
```

### How It Works

1. `torchrun` sets `RANK`, `LOCAL_RANK`, and `WORLD_SIZE` environment variables
2. The `--tensor-parallel` flag enables TP mode (without it, `torchrun` alone won't activate TP)
3. Rank 0 runs the HTTP server and prompt queue; worker ranks (rank 1+) use `NullServer` — a no-op stand-in that safely handles all `PromptServer.instance` patterns from custom nodes
4. The prompt queue is broadcast from rank 0 to all worker ranks via NCCL
5. All ranks execute the same prompt; TP layers use all-reduce to synchronize
6. Only rank 0 saves output images and sends WebSocket notifications

### Memory Requirements

With 2 GPUs, each GPU holds approximately half of the parallelized layers plus the full non-parallelized components (embeddings, layer norms, VAE, text encoders). For example:

- **Flux-dev**: ~15 GiB per GPU (on 2x RTX 5060 Ti 16 GiB)
- Models larger than 2x your GPU VRAM will not fit; consider `--force-fp16` and quantization options

### Custom Nodes

Custom nodes load on all ranks. On worker ranks (rank 1+), server-dependent operations (WebSocket notifications, API routes, etc.) are safely handled by `NullServer` and `NullProxy` — no crashes from missing `PromptServer.instance` references.

The ComfyUI-Manager pip package works on rank 0; the legacy custom_nodes version is automatically disabled.

## Architecture

```
comfy/distributed/
  __init__.py        — Package exports
  mesh.py            — DeviceMesh: NCCL process group and rank/device mapping
  parallel_linear.py — ParallelLinear: colwise/rowwise sharded linear layer
  patcher.py         — parallelize_model(), load_tp_shards(), TP_TARGETS
  utils.py           — is_tp_active(), tp_aware_to(), get_tp_param_names()
  null_server.py     — NullServer/NullProxy/NullQueue for worker ranks
```

Key integration points in core ComfyUI:
- `main.py`: TP initialization, prompt broadcast, exception propagation
- `comfy/sd.py`: TP device override in model loading
- `comfy/model_patcher.py`: `is_tp_parallelized` guards and `tp_aware_to()`
- `comfy/model_base.py`: Skip TP params during state dict loading
- `comfy/cli_args.py`: `--tensor-parallel` flag

## Error Handling

When any rank encounters an error during prompt execution:
1. The error is caught locally
2. An `all_reduce` with `MAX` operation communicates whether any rank failed
3. All ranks proceed past the synchronization barrier — no hanging
4. Rank 0 reports the error to the queue; worker ranks skip result handling

## Limitations

- **NCCL only**: Currently requires NVIDIA GPUs with NCCL backend
- **LoRA**: DynamicVRAM attaches LowVramPatch to ParallelLinear.weight_function. Packed colwise layers (Flux/Hunyuan/Chroma `*.qkv`, MiniMax `qkv_proj`/`fc1`) slice LoRA diffs per pack — a naive row cut would mix Q/K/V. Flux+LoRA verified on unreplicated QKV; packed QKV LoRA is unit-tested. Qwen/Cosmos/Wan LoRA e2e still being expanded. DoRA/LoHa/OFT under TP are not yet fully supported.
- **Dynamic batching**: All ranks must process the same prompt; batch parallelism is not combined with TP
- **Model saving**: Only rank 0 saves output; worker ranks skip file I/O
- **HiDream O1**: Disabled — needs full-model TP including vision encoder
- **MiniMax H3**: Packed colwise. `qkv_proj` shards each of `[Q|K|V]` by heads; `fc1` shards each of `[gate|up]`; `fc2`/`out_proj` are rowwise. adaLN stays replicated.
- **Flux / HunyuanVideo / Chroma**: Packed colwise on double-stream `img_attn.qkv` / `txt_attn.qkv` (`[Q|K|V]`); `*.proj` is rowwise. Single-stream fused `linear1`/`linear2` stay replicated (QKV+MLP unequal packs). `SingleStreamBlock.num_heads` is not divided. Hunyuan `txt_in` TokenRefiner is outside TP prefixes and stays full-width.
- **SD3 / MMDiT**: Packed colwise on `attn.qkv` / `attn2.qkv` (`[Q|K|V]`, `split_qkv` layout); `*.proj` is rowwise. adaLN stays replicated. Root `num_heads` (depth metadata) is not divided.
- **LTXV / LTXAV**: MLP-only. RoPE is rebuilt from full `num_attention_heads` × `inner_dim`, so FA/CA (`to_q`/`to_k`/`to_v`/`to_out.0`) stay replicated.
- **ACE-Step 1.0**: Head-split attention (`transformer_blocks`). FF is `GLUMBConv` and stays replicated.
- **ACE-Step 1.5**: Head-split + GQA (`decoder.layers` only). Lyric/timbre encoders stay full-width. `num_kv_heads` is divided only under that prefix.
- **Lumina NextDiT / Z-Image**: Packed GQA colwise on `attention.qkv` via `pack_sizes=(n_heads*d, n_kv*d, n_kv*d)` — equal `pack_count=3` would cut the wider Q pack. `attention.out` is rowwise. Unfused SwiGLU `w1`/`w3`/`w2` shards. `adaLN_modulation` stays. Root `n_heads` is RoPE metadata and is not divided; `n_local_heads` / `n_local_kv_heads` are.
- **Ideogram 4**: Packed colwise on `attention.qkv` (`[Q|K|V]`, `view(..., 3, heads, head_dim)`); `attention.o` is rowwise. `adaln_modulation` stays replicated.
- **JoyImage**: Packed colwise on `img_attn_qkv` / `txt_attn_qkv`; `*_attn_proj` rowwise. MLP `net.0.proj` / `net.2` follow Qwen-style colwise/rowwise. `JoyImageModulate` is a Parameter table, not a Linear.
- **Lens**: Packed colwise on `img_qkv` / `txt_qkv`; `to_out.0` / `to_add_out` rowwise. Unfused SwiGLU `w1`/`w3`/`w2`. `img_mod.1` / `txt_mod.1` stay (global 6-way chunk).
- **PixelDiT**: Packed colwise on patch `attn.qkv_x` / `attn.qkv_y` and pixel `attn.qkv` (`[Q|K|V]`, `reshape(..., 3, heads, head_dim)`); `proj_*` / `attn.proj` rowwise. adaLN chunks, `compress_to_attn`, and `expand_from_attn` stay. `PiTBlock.num_heads` is RoPE metadata and is not divided. `PidNet` inherits via MRO (`lq_proj` is outside TP prefixes).
- **Cosmos GeneralDIT**: MLP-only. FA/CA Sequential projections (`attn.to_{q,k,v,out}.0`) stay replicated; `adaLN_modulation` is excluded globally.
