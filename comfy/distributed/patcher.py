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


def _is_linear_layer(module):
    """Check if a module is a linear layer that can be parallelized.

    ComfyUI uses custom Linear subclasses (disable_weight_init.Linear,
    manual_cast.Linear, mixed_precision_ops.Linear) that may not inherit
    from nn.Linear. We use duck-typing: any module with in_features and
    out_features attributes is treated as a linear layer.
    """
    return isinstance(module, nn.Linear) or (
        hasattr(module, 'in_features') and hasattr(module, 'out_features')
    )

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
# - proj/to_out/o_proj: attention output projections (reduce across shard, all-reduce)
# - mlp.2: MLP second linear (down-projection, rowwise with all-reduce)
# - down/down_proj: general down-projection patterns
# - w2: Megatron-style MLP down-projection
ROWWISE_KEYWORDS = ["to_out", "proj", "down", "w2", "mlp.2", "to_out_t", "o_proj", "down_proj"]
COLWISE_KEYWORDS = ["to_q", "to_k", "to_v", "up", "w1", "w3", "to_q_t", "to_k_t", "to_v_t", "q_proj", "k_proj", "v_proj", "gate_proj", "up_proj"]

# Layer names that must NOT be sharded. These are excluded because:
# - Modulation layers (chunk() with full dim): modulation.lin, img_mod.lin, txt_mod.lin
# - Fused QKV+MLP layers (torch.split with full dim): linear1, linear2
# - Fused QKV projections (reshape with 3*num_heads): img_attn.qkv, txt_attn.qkv
# - Coupled attention output projections (receive full-dim input from excluded qkv):
#   img_attn.proj, txt_attn.proj — must stay unsharded when qkv is excluded
EXCLUDED_LAYER_NAMES = [
    "linear1", "linear2",
    "modulation.lin", "img_mod.lin", "txt_mod.lin",
    "img_attn.qkv", "txt_attn.qkv",
    "img_attn.proj", "txt_attn.proj",
]

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
    skipped_dims = 0
    for name, module in model.named_modules():
        if any(name.startswith(prefix) for prefix in targets):
            if _is_linear_layer(module):
                # Skip layers that can't be simply sharded (fused layers, modulation layers)
                if any(name.endswith(excl) for excl in EXCLUDED_LAYER_NAMES):
                    continue

                # Determine mode
                mode = "colwise"
                if any(k in name for k in ROWWISE_KEYWORDS):
                    mode = "rowwise"
                elif any(k in name for k in COLWISE_KEYWORDS):
                    mode = "colwise"

                # Skip layers where the shard dimension isn't evenly divisible
                dim = module.out_features if mode == "colwise" else module.in_features
                if dim % mesh.world_size != 0:
                    skipped_dims += 1
                    continue

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

    logging.info(f"[TP] Parallelized {count} linear layers across {len(targets)} block groups"
                 f" ({skipped_dims} skipped for non-divisible dimensions)")
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