"""
    This file is part of ComfyUI.
    Copyright (C) 2024 Comfy

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""

from __future__ import annotations

import torch
import torch.nn as nn


def is_tp_active() -> bool:
    """Check whether Tensor Parallelism is currently active.

    Returns True only when torch.distributed is importable, initialized,
    and a process group exists. Safe to call from any context (single-GPU,
    multi-GPU without TP, or TP mode).
    """
    try:
        import torch.distributed as dist
        return dist.is_initialized()
    except (ImportError, RuntimeError):
        return False


def tp_runtime_info() -> dict:
    """JSON-safe TP identity for /system_stats and the e2e generation gate.

    Inactive / single-process runs report ``active=False`` and ``world_size=1``.
    """
    info = {"active": False, "world_size": 1, "rank": 0}
    if not is_tp_active():
        return info
    try:
        from comfy.distributed.mesh import get_mesh
        mesh = get_mesh()
        info["active"] = True
        info["world_size"] = int(mesh.world_size)
        info["rank"] = int(mesh.rank)
        return info
    except Exception:
        pass
    try:
        import torch.distributed as dist
        info["active"] = True
        info["world_size"] = int(dist.get_world_size())
        info["rank"] = int(dist.get_rank())
    except Exception:
        pass
    return info


def get_tp_param_names(model) -> set[str]:
    """Collect the set of fully-qualified parameter names belonging to
    TP-owned shards that must be skipped during full state-dict loads.

    Includes:
      - ParallelLinear layers (`is_tp_parallelized=True`)
      - Full-dim QK norms sliced for head-split (Wan / HiDream), marked
        with `_tp_norm_shard` during parallelize_model

    Used by model_base and model_patcher to skip TP params during
    state dict loading and device transfers.
    """
    tp_param_names = set()
    for mod_name, mod in model.named_modules():
        if getattr(mod, 'is_tp_parallelized', False):
            for pn, _ in mod.named_parameters(recurse=False):
                tp_param_names.add(f"{mod_name}.{pn}" if mod_name else pn)
        elif getattr(mod, '_tp_norm_shard', None) is not None:
            # Sliced RMSNorm / LayerNorm — checkpoint holds full-dim weight
            for pn, _ in mod.named_parameters(recurse=False):
                tp_param_names.add(f"{mod_name}.{pn}" if mod_name else pn)
    return tp_param_names


def tp_aware_to(model, device):
    """Move model parameters to device, but skip ParallelLinear shards
    that are already on their correct GPU in TP mode.

    In TP mode, ParallelLinear shards are placed on each rank's GPU by
    load_tp_shards(). We must NOT move them again -- only non-TP
    parameters (layer norms, embeddings, etc.) and buffers are moved
    to the requested device.
    """
    if not is_tp_active():
        model.to(device)
        return

    target = torch.device(device)

    tp_param_names = get_tp_param_names(model)

    # Move non-TP parameters to the requested device
    for name, param in model.named_parameters():
        if name not in tp_param_names:
            param.data = param.data.to(target)

    # Move all buffers to the requested device
    for name, buf in model.named_buffers():
        buf.data = buf.data.to(target)