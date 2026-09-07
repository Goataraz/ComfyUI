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
| Flux | `Flux` | `double_blocks`, `single_blocks` |
| SD3 / MMDiT | `OpenAISignatureMMDITWrapper` | `joint_blocks` |
| Cosmos T2V/I2V | `GeneralDIT` | `blocks` (MLP-only; FA/CA stay replicated) |
| Cosmos Predict2 / Anima | `MiniTrainDIT` | `blocks` |
| HiDream Image | `HiDreamImageTransformer2DModel` | `double_stream_blocks`, `single_stream_blocks` |
| QwenImage | `QwenImageTransformer2DModel` | `transformer_blocks` |
| Qwen25 7B text encoder | `Llama2` | `layers` |
| WanVideo (T2V/I2V + subclasses) | `WanModel` | `blocks` |
| CausalWan | `CausalWanModel` | inherits `WanModel` (`blocks`) |

Head-split models (heads divided by world_size): QwenImage, MiniTrainDIT, WanModel (and CausalWan via MRO), HiDreamImageTransformer2DModel.
GeneralDIT is **not** head-split — attention projections are excluded, so only MLP linears shard.
Wan / HiDream full-dim QK RMSNorms are sliced to the local shard; QwenImage / Cosmos per-head norms stay replicated.

CausalWan keeps `num_heads` on the outer model (KV cache) and on `WanAttentionBlock` (cross-attn). After Attention modules are head-split, leftover `heads` / `n_heads` / `num_heads` that still equal the pre-split count are synced to the local shard.

HiDream O1 is **not** supported — its integrated Llama2 LLM receives input from non-TP visual/x_embedder components and needs a full-model TP strategy that shards the vision encoder too. The `HiDreamO1Transformer` class is in `TP_UNSUPPORTED` and `get_tp_targets` returns `[]` for it.

MiniMax H3 is **not** supported yet — fused `qkv_proj` (packed Q\|K\|V) and fused SwiGLU `fc1` (`ffn*2` then chunk) cannot be naively colwise-sharded.

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
- **LoRA**: DynamicVRAM now attaches LowVramPatch to ParallelLinear.weight_function. Flux+LoRA verified; Qwen/Cosmos/Wan LoRA e2e still being expanded. DoRA/LoHa/OFT under TP are not yet fully supported.
- **Dynamic batching**: All ranks must process the same prompt; batch parallelism is not combined with TP
- **Model saving**: Only rank 0 saves output; worker ranks skip file I/O
- **HiDream O1**: Disabled — needs full-model TP including vision encoder
- **MiniMax H3**: Disabled — fused QKV + fused SwiGLU need pack-aware sharding
- **Cosmos GeneralDIT**: MLP-only. FA/CA Sequential projections (`attn.to_{q,k,v,out}.0`) stay replicated; `adaLN_modulation` is excluded globally.