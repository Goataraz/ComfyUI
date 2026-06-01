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
    # HiDreamO1Transformer has an integrated Llama2 language model that cannot be
    # naively sharded — its layers receive input from non-TP visual/x_embedder
    # components. Full TP support requires sharding the vision encoder too.
    # "HiDreamO1Transformer": ["language_model.layers"],
    "QwenImageTransformer2DModel": ["transformer_blocks"],
    "Llama2": ["layers"],
}

# Keywords for determining TP sharding mode (rowwise = split input dim, colwise = split output dim)
# Rowwise is checked first; if a layer name matches both, rowwise takes precedence.
# - proj/to_out/o_proj: attention output projections (reduce across shard, all-reduce)
# - mlp.2: MLP second linear (down-projection, rowwise with all-reduce)
# - down/down_proj: general down-projection patterns
# - w2: Megatron-style MLP down-projection
ROWWISE_KEYWORDS = ["to_out", "to_add_out", "proj", "down", "w2", "mlp.2", "to_out_t", "o_proj", "down_proj"]
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
    # QwenImage uses nn.Sequential(SiLU, Linear) for modulation; the Linear is
    # at index 1 (not a ".lin" suffix). Output is 6*dim, chunked 2-way for
    # (shift, scale, gate).
    "img_mod.1", "txt_mod.1",
]

# Models where TP sharding splits HEADS (not head_dim). For these models
# each rank holds a contiguous slice of the colwise projection's output
# channels — i.e., a subset of heads with full per-head dim. The Attention
# module's `self.heads` must be divided by world_size so that the model's
# `.view(B, S, heads, dim_head)` reshape produces the correct
# (B, S, heads//ws, dim_head) layout. The per-head RMSNorm weights and
# rotary embeddings stay at full dim_head (replicated across ranks).
TP_HEAD_SPLIT_MODELS = {"QwenImageTransformer2DModel"}

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

    # For head-split models (QwenImage) the colwise shard distributes a
    # contiguous slice of heads across ranks. We need to override
    # `self.heads` on each Attention module so the model's
    # `.view(B, S, heads, dim_head)` reshape produces the right per-rank
    # layout. Detect this once up front.
    model_class_name = None
    for cls in type(model).__mro__:
        if cls.__name__ in TP_HEAD_SPLIT_MODELS:
            model_class_name = cls.__name__
            break
    head_split_active = model_class_name is not None

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
                # Check colwise first: more specific QKV/input projection
                # keywords (to_q, q_proj, etc.) take precedence over the
                # generic `proj` substring. Also detect the QwenImage
                # GELU/MLP first-Linear pattern `*.net.0.proj` (up-projection
                # with no colwise keyword) explicitly so it does not fall
                # through to the generic `proj` rowwise rule.
                mode = "colwise"
                if any(k in name for k in COLWISE_KEYWORDS):
                    mode = "colwise"
                elif ".net.0.proj" in name:
                    # GELU/MLP first-Linear in a ModuleList (QwenImage) —
                    # structurally an up-projection, colwise with gather.
                    mode = "colwise"
                elif any(k in name for k in ROWWISE_KEYWORDS) or name.endswith(".net.2"):
                    # `.net.2` covers QwenImage's MLP down-projection
                    # (the GELU+Droput+Linear ModuleList's index-2 Linear),
                    # matching Flux's `mlp.2` rowwise convention.
                    mode = "rowwise"

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

    # Head-split override: for QwenImage-style models, the colwise shard
    # distributes a contiguous slice of heads across ranks. Each rank's
    # output of to_q/to_k/to_v is `(heads/world_size) * dim_head` channels,
    # so the Attention module's `self.heads` must be divided by world_size
    # to make the model's `.view(B, S, heads, dim_head)` reshape produce
    # the right per-rank layout. Per-head norms and the rotary embedding
    # stay at full dim_head on every rank (no slicing needed).
    if head_split_active:
        head_overrides = 0
        for module in model.modules():
            # Duck-typed Attention detection: has heads, dim_head, to_q
            # already wrapped as ParallelLinear, and a per-head norm.
            to_q = getattr(module, "to_q", None)
            if not isinstance(to_q, ParallelLinear):
                continue
            heads = getattr(module, "heads", None)
            dim_head = getattr(module, "dim_head", None)
            if not isinstance(heads, int) or not isinstance(dim_head, int):
                continue
            if heads % mesh.world_size != 0:
                raise RuntimeError(
                    f"[TP] {model_class_name} attention heads={heads} not divisible by "
                    f"world_size={mesh.world_size}"
                )
            local_heads = heads // mesh.world_size
            module.heads = local_heads
            head_overrides += 1
        if head_overrides:
            logging.info(
                f"[TP] Head-split override: set heads={local_heads} (from "
                f"{heads}) on {head_overrides} Attention modules"
            )
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