#!/bin/bash
# ComfyUI - Single GPU (one RTX 5060 Ti) - Port 8288
# Faithful mirror of start-tp.sh minus the tensor-parallel parts:
#   no torchrun / --nproc_per_node=2, no --tensor-parallel, no NCCL_* env.
# LoRAs apply correctly here — the TP fork on :8188 silently drops every LoRA
# (sharded ParallelLinear.weight vs full B@A reshape -> patch swallowed; see
# the comfyui-tp-drops-loras memory). This instance and comfyui-tp run in
# PARALLEL (both up all the time); switch the active one by clearing VRAM in
# the one you're leaving (POST /free with unload_models, or the Manager "Free
# VRAM" button) before loading on the other. Network flags
# (--listen 0.0.0.0 / --enable-cors-header / --disable-metadata /
# --disable-auto-launch) are appended by the comfyui-single.service unit, same
# as the TP unit does — so this script mirrors start-tp.sh's structure exactly.

cd /home/blackthorn/ComfyUI

export VIRTUAL_ENV=/home/blackthorn/ComfyUI/venv
export PATH="/home/blackthorn/ComfyUI/venv/bin:${PATH}"
export LD_LIBRARY_PATH="/home/blackthorn/ComfyUI/venv/lib/python3.13/site-packages/nvidia/cudnn/lib:/home/blackthorn/ComfyUI/venv/lib/python3.13/site-packages/nvidia/cublas/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export TORCH_CUDNN_V8_AVAILABLE=1
export TORCH_CUDNN_V8_ENABLED=1
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false

# Own db (/tmp/comfyui_single.db) — separate from TP's /tmp/comfyui_tp.db so the
# two instances don't contend on a sqlite lock while both are up in parallel.
exec python main.py \
    --listen 127.0.0.1 \
    --port 8288 \
    --lowvram \
    --enable-manager-legacy-ui \
    --database-url "sqlite:////tmp/comfyui_single.db" \
    --front-end-version Comfy-Org/ComfyUI_frontend@latest \
    "$@"