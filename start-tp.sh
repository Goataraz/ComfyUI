#!/bin/bash
# ComfyUI - Tensor Parallelism (2x RTX 5060 Ti) - Port 8188
# Launches ComfyUI with torchrun across both GPUs for models that need >16GB VRAM.
#
# Network/security posture (--listen, --enable-cors-header, --disable-metadata,
# --disable-auto-launch) is configured in the systemd unit file
# /etc/systemd/system/comfyui-tp.service, not here. Do not add those flags here
# or argparse will see them twice.

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

# --- Suppress tokenizer parallelism warnings from worker ranks ---
export TOKENIZERS_PARALLELISM=false

exec torchrun --nproc_per_node=2 main.py \
    --listen 127.0.0.1 \
    --port 8188 \
    --tensor-parallel \
    --lowvram \
    --enable-manager-legacy-ui \
    --database-url "sqlite:////tmp/comfyui_tp.db" \
    --front-end-version Comfy-Org/ComfyUI_frontend@latest \
    "$@"
