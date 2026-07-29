#!/bin/bash
# ComfyUI - Tensor Parallelism (2x RTX 5060 Ti) - Port 8188
# Launches ComfyUI with torchrun across both GPUs for models that need >16GB VRAM.
#
# Network/security posture (--listen, --enable-cors-header, --disable-metadata,
# --disable-auto-launch) is configured in the systemd unit file
# /etc/systemd/system/comfyui-tp.service, not here. Do not add those flags here
# or argparse will see them twice.
#
# OPTIMIZED (Jun 30 2026): Replaced --lowvram with --enable-dynamic-vram + full
# performance flag set. SageAttention 1.76x speedup, FP16/BF16 precision, FP8
# matrix multiply, async offload, RAM pressure caching.

cd /home/blackthorn/ComfyUI

# --- Use venv Python so all dependencies (git, etc.) are available ---
export VIRTUAL_ENV=/home/blackthorn/ComfyUI/venv
export PATH="/home/blackthorn/ComfyUI/venv/bin:${PATH}"

# --- cuDNN / CUDA Library Priority ---
export LD_LIBRARY_PATH="/home/blackthorn/ComfyUI/venv/lib/python3.13/site-packages/nvidia/cudnn/lib:/home/blackthorn/ComfyUI/venv/lib/python3.13/site-packages/nvidia/cublas/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

# --- PyTorch Optimizations ---
export TORCH_CUDNN_V8_AVAILABLE=1
export TORCH_CUDNN_V8_ENABLED=1

# --- CUDA Environment ---
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_MODULE_LOADING=LAZY

# --- Memory: reduce VRAM fragmentation across both GPUs ---
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# --- NCCL: single-node loopback, no InfiniBand, watchdog off ---
export NCCL_SOCKET_IFNAME=lo
export NCCL_IB_DISABLE=1
export TORCH_NCCL_ENABLE_MONITORING=0
# Allow up to 30min per collective before NCCL watchdog kills the process.
# Default 1800s — set explicitly so future PyTorch/NCCL bumps don't change it.
# Root cause of frequent restarts: rank 0 races past dist.broadcast() before
# rank 1 finishes heavy CUDA work (Anima/Flux all-reduce). The collective
# timeout then aborts both ranks. The main.py cuda.synchronize() guards are
# the primary fix; this is a safety net for genuinely stuck collectives.
export NCCL_TIMEOUT=1800
# --- GPU idle spin fix (Jul 25 2026) ---
# PyTorch 2.12+ canonical names. NCCL_ASYNC_ERROR_HANDLING=1 (set by the
# systemd unit) creates a watchdog thread that busy-polls the GPU even when
# idle, pinning GPU 1 at 100% utilization with 31W wasted power. Switching
# to TORCH_NCCL_BLOCKING_WAIT=1 makes NCCL calls block-wait (sleep) instead
# of spin-wait, and disabling async error handling removes the polling
# thread. The timeout still applies via the process group timeout.
export TORCH_NCCL_BLOCKING_WAIT=1
export NCCL_ASYNC_ERROR_HANDLING=0
export TORCH_NCCL_ASYNC_ERROR_HANDLING=0

# --- Suppress tokenizer parallelism warnings from worker ranks ---
export TOKENIZERS_PARALLELISM=false

exec torchrun --nproc_per_node=2 main.py \
    --listen 127.0.0.1 \
    --port 8188 \
    --tensor-parallel \
    --enable-dynamic-vram \
    --use-sage-attention \
    --force-fp16 \
    --bf16-vae \
    --bf16-unet \
    --fast fp16_accumulation fp8_matrix_mult cublas_ops \
    --async-offload 2 \
    --reserve-vram 1.5 \
    --enable-manager-legacy-ui \
    --database-url "sqlite:////tmp/comfyui_tp.db" \
    --front-end-version Comfy-Org/ComfyUI_frontend@latest \
    "$@"
