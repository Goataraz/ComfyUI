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
import torch.distributed as dist
from comfy.model_patcher import ModelPatcher
from comfy.distributed.mesh import get_mesh
from comfy.distributed.parallel_linear import ParallelLinear
import logging
import comfy.utils

# Mapping of inner diffusion model class names to their TP target prefixes.
# Keys must match the __name__ of the inner diffusion model class (not the BaseModel wrapper).
# Subclasses are handled via MRO walk, so e.g. Anima(MiniTrainDIT) inherits "blocks".
TP_TARGETS = {
    "Flux": ["double_blocks", "single_blocks"],
    "OpenAISignatureMMDITWrapper": ["blocks"],       # SD3
    "GeneralDIT": ["blocks"],                          # Cosmos T2V/I2V
    "MiniTrainDIT": ["blocks"],                       # Cosmos Predict2 / Anima
    "HiDreamImageTransformer2DModel": ["double_stream_blocks", "single_stream_blocks"],
    "HiDreamO1Transformer": ["language_model.layers"],
}

# Keywords for determining TP sharding mode (rowwise = split input dim, colwise = split output dim)
# Rowwise is checked first; if a layer name matches both, rowwise takes precedence.
ROWWISE_KEYWORDS = ["to_out", "proj", "down", "w2", "to_out_t"]
COLWISE_KEYWORDS = ["to_q", "to_k", "to_v", "up", "w1", "w3", "to_q_t", "to_k_t", "to_v_t"]

# Fused layer names that CANNOT be simply sharded with colwise/rowwise.
# In Flux SingleStreamBlock, linear1 fuses QKV+MLP-in and linear2 fuses proj+MLP-out.
# These fused layers use torch.split() with full (unsharded) dimensions internally,
# so a simple colwise/rowwise split would cause dimension mismatches.
FUSED_LAYER_EXCLUSIONS = ["linear1", "linear2"]

def get_tp_targets(model):
    """Determines TP target prefixes by walking the model's MRO for an exact class name match."""
    for cls in type(model).__mro__:
        if cls.__name__ in TP_TARGETS:
            return TP_TARGETS[cls.__name__]
    return []

def parallelize_model(model):
    """
    Replaces target linear layers in the model with ParallelLinear.
    """
    targets = get_tp_targets(model)
    if not targets:
        return False

    logging.info(f"Applying Tensor Parallelism to {model.__class__.__name__} with targets: {targets}")
    mesh = get_mesh()

    count = 0
    for name, module in model.named_modules():
        if any(name.startswith(prefix) for prefix in targets):
            if isinstance(module, nn.Linear):
                # Skip fused layers that can't be simply sharded (e.g., Flux SingleStreamBlock linear1/linear2)
                if any(name.endswith(excl) for excl in FUSED_LAYER_EXCLUSIONS):
                    continue

                # Determine mode
                mode = "colwise"
                if any(k in name for k in ROWWISE_KEYWORDS):
                    mode = "rowwise"
                elif any(k in name for k in COLWISE_KEYWORDS):
                    mode = "colwise"

                new_layer = ParallelLinear(
                    in_features=module.in_features,
                    out_features=module.out_features,
                    bias=module.bias is not None,
                    mode=mode
                )

                parent_name, _, child_name = name.rpartition('.')
                parent = model
                if parent_name:
                    parent = comfy.utils.get_attr(model, parent_name)

                setattr(parent, child_name, new_layer)
                count += 1

    logging.info(f"[TP] Parallelized {count} linear layers across {len(targets)} block groups")
    return True

def load_tp_shards(model, sd, prefix=""):
    """
    Loads sharded weights from the state dict into ParallelLinear layers.

    Args:
        model: The neural network model with ParallelLinear layers
        sd: State dict containing full (un-sharded) weights
        prefix: Optional key prefix to prepend (e.g., "model.diffusion_model.")
    """
    mesh = get_mesh()
    for name, module in model.named_modules():
        if isinstance(module, ParallelLinear):
            weight_key = prefix + name + ".weight"
            if weight_key in sd:
                full_weight = sd[weight_key]
                module.load_shard(full_weight)

            if module.bias is not None:
                bias_key = prefix + name + ".bias"
                if bias_key in sd:
                    full_bias = sd[bias_key]
                    if module.mode == "colwise":
                        start = mesh.rank * module.local_out_features
                        end = start + module.local_out_features
                        module.bias.data = full_bias[start:end].to(mesh.current_device)
                    else:
                        module.bias.data = full_bias.to(mesh.current_device)
    logging.info(f"Loaded TP shards for {model.__class__.__name__}")