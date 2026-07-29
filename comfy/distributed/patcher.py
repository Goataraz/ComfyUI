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
# Subclasses are handled via MRO walk, so e.g. Anima(MiniTrainDIT) / VaceWanModel(WanModel)
# inherit the parent entry.
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
    "WanModel": ["blocks"],                            # WanVideo T2V/I2V + subclasses
}

# Keywords for determining TP sharding mode (rowwise = split input dim, colwise = split output dim)
# Rowwise is checked first; if a layer name matches both, rowwise takes precedence.
# - proj/to_out/o_proj: attention output projections (reduce across shard, all-reduce)
# - mlp.2: MLP second linear (down-projection, rowwise with all-reduce)
# - down/down_proj: general down-projection patterns
# - w2: Megatron-style MLP down-projection
# - ffn.2 / block.layer2: Wan / Cosmos GeneralDIT MLP down-projections
ROWWISE_KEYWORDS = [
    "to_out", "to_add_out", "proj", "down", "w2", "mlp.2", "to_out_t",
    "o_proj", "down_proj", "mlp.layer2", "ffn.2",
]
COLWISE_KEYWORDS = [
    "to_q", "to_k", "to_v", "up", "w1", "w3", "to_q_t", "to_k_t", "to_v_t",
    "q_proj", "k_proj", "v_proj", "gate_proj", "up_proj",
    "mlp.layer1",
]

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
    # Cosmos Predict2 / Anima: adaln modulation Sequentials (lowercase)
    # When use_adaln_lora=True the Sequential is (SiLU, Linear_down, Linear_up)
    # — two linears at indices .1 and .2 that form a bottleneck and must stay
    # together as un-sharded layers. When use_adaln_lora=False the Sequential
    # is (SiLU, Linear) — only .1 exists. Excluding both covers both cases.
    "adaln_modulation.1", "adaln_modulation.2",
    "adaln_modulation_self_attn.1", "adaln_modulation_self_attn.2",
    "adaln_modulation_cross_attn.1", "adaln_modulation_cross_attn.2",
    "adaln_modulation_mlp.1", "adaln_modulation_mlp.2",
    # Cosmos GeneralDIT + HiDream Image use capital LN: adaLN_modulation
    # (chunked 3-way / 6-way / 12-way on full dim). Same suffix covers both.
    "adaLN_modulation.1", "adaLN_modulation.2",
]

# GeneralDIT-only exclusions. MUST NOT live in EXCLUDED_LAYER_NAMES:
# QwenImage also uses `attn.to_out.0` (ModuleList Linear), and globally
# excluding that suffix leaves Qwen QKV head-split while the out-proj stays
# full-width → matmul (…x1536 @ 3072x3072). CosmOS wraps projections in
# Sequential so names are `attn.to_{q,k,v,out}.0`; Qwen's QKV are bare
# `attn.to_q` (no `.0`) and must remain sharded.
GENERALDIT_EXCLUDED_LAYER_NAMES = (
    "attn.to_q.0", "attn.to_k.0", "attn.to_v.0", "attn.to_out.0",
)

# Models where TP sharding splits HEADS (not head_dim). For these models
# each rank holds a contiguous slice of the colwise projection's output
# channels — i.e., a subset of heads with full per-head dim. The Attention
# module's `self.heads` / `n_heads` / `num_heads` must be divided by
# world_size so that the model's `.view(B, S, heads, dim_head)` reshape
# produces the correct (B, S, heads//ws, dim_head) layout.
#
# Norm policy:
# - Per-head norms (normalized_shape == dim_head, applied AFTER rearrange):
#   leave replicated. Covered by QwenImage / Cosmos GeneralDIT.
# - Full-dim QK norms (normalized_shape == heads*dim_head, applied BEFORE
#   rearrange): must be sliced to local_heads*dim_head. Covered by Wan and
#   HiDream Image. See `_slice_full_dim_qk_norms`.
TP_HEAD_SPLIT_MODELS = {
    "QwenImageTransformer2DModel",
    "MiniTrainDIT",
    # GeneralDIT: attn excluded from TP (MLP-only); head-split not needed.
    "WanModel",
    "HiDreamImageTransformer2DModel",
}

# Attribute names used for the Q projection across architectures.
_Q_PROJ_ATTRS = ("to_q", "q_proj", "q")
# Attribute names for head count / head dim.
_HEADS_ATTRS = ("heads", "n_heads", "num_heads")
_DIM_HEAD_ATTRS = ("dim_head", "head_dim")
# Full-dim QK norms that must be sliced under head-split (Wan / HiDream).
_FULL_DIM_QK_NORM_ATTRS = (
    "norm_q", "norm_k", "norm_k_img",
    "q_rms_norm", "k_rms_norm", "q_rms_norm_t", "k_rms_norm_t",
)


def _resolve_q_proj(module):
    """Find the Q projection Linear/ParallelLinear on an Attention module.

    Handles:
      - Direct attrs: to_q / q_proj / q
      - CosmOS Sequential: to_q = Sequential(Linear, Norm) → return Linear at [0]
    """
    for attr in _Q_PROJ_ATTRS:
        q = getattr(module, attr, None)
        if q is None:
            continue
        if isinstance(q, nn.Sequential) and len(q) > 0:
            q = q[0]
        if isinstance(q, ParallelLinear) or _is_linear_layer(q):
            return q
    return None


def _get_attr_first(module, names):
    for name in names:
        val = getattr(module, name, None)
        if val is not None:
            return name, val
    return None, None


def _slice_full_dim_qk_norms(module, local_heads, dim_head, mesh):
    """Slice full-dim QK RMSNorms to match the local head shard.

    Wan and HiDream apply RMSNorm(dim) AFTER the colwise Q/K projection but
    BEFORE the heads reshape. After colwise sharding the projection output is
    `local_heads * dim_head` channels, so the norm weight and normalized_shape
    must shrink to match. Per-head norms (weight.shape == dim_head) are left alone.
    """
    local_dim = local_heads * dim_head
    full_dim = None
    # Prefer the pre-override head count stored on the module if we kept it;
    # otherwise derive from weight shape.
    sliced = 0
    for attr in _FULL_DIM_QK_NORM_ATTRS:
        norm = getattr(module, attr, None)
        if norm is None or not hasattr(norm, "weight") or norm.weight is None:
            continue
        w = norm.weight
        if w.ndim != 1:
            continue
        # Skip per-head norms (already correct size) and already-sliced norms.
        if w.shape[0] == dim_head or w.shape[0] == local_dim:
            continue
        # Only slice when weight is an exact multiple of dim_head matching
        # the original full head count (heads * dim_head).
        if w.shape[0] % dim_head != 0:
            continue
        if w.shape[0] // dim_head != local_heads * mesh.world_size:
            # Unexpected size — leave alone rather than corrupt.
            continue
        start = mesh.rank * local_dim
        end = start + local_dim
        norm.weight = nn.Parameter(
            w.data[start:end].clone(),
            requires_grad=w.requires_grad,
        )
        if hasattr(norm, "normalized_shape"):
            norm.normalized_shape = (local_dim,)
        # Mark so load_tp_shards / diagnostics can see it.
        norm._tp_norm_shard = (start, end)
        sliced += 1
    return sliced


# Models whose MRO would match a supported parent but that are NOT safe
# to shard yet (architecture-specific head bookkeeping outside Attention).
TP_UNSUPPORTED = {
    # CausalWanModel(WanModel) keeps outer/block-level num_heads for KV cache
    # allocation; head-split only updates nested Attention modules → cache
    # shape mismatch. Needs dedicated causal-Wan TP work.
    "CausalWanModel",
    # HiDreamO1: integrated Llama2 LLM + vision encoder coupling.
    "HiDreamO1Transformer",
    # GeneralDIT (Cosmos 1.0): FA/CA share the Attention class but CA K/V are
    # Linear(1024→4096) while FA is Linear(4096→4096). Even MLP-only TP still
    # crashes in cal_qkv (expected in=4096, got 1024) — root cause is NOT the
    # attention ParallelLinear path (fails before any MLP runs). Needs a
    # dedicated Cosmos TP strategy + conditioning/layout audit.
    "GeneralDIT",
}


def get_tp_targets(model):
    """Determines TP target prefixes by walking the model's MRO for an exact class name match."""
    concrete = type(model).__name__
    # Deny if the concrete class OR any MRO parent is explicitly unsupported
    # (e.g. Anima(GeneralDIT) must not inherit GeneralDIT's allowlist entry).
    for cls in type(model).__mro__:
        if cls.__name__ in TP_UNSUPPORTED:
            logging.info(f"[TP] {concrete} is explicitly unsupported via {cls.__name__} — skipping TP")
            return []
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

    # For head-split models the colwise shard distributes a contiguous slice
    # of heads across ranks. Detect once up front via MRO.
    model_class_name = None
    for cls in type(model).__mro__:
        if cls.__name__ in TP_HEAD_SPLIT_MODELS:
            model_class_name = cls.__name__
            break
    head_split_active = model_class_name is not None

    logging.info(f"Applying Tensor Parallelism to {model.__class__.__name__} with targets: {targets}")
    mesh = get_mesh()

    # GeneralDIT attention exclusions are class-scoped — see GENERALDIT_EXCLUDED_LAYER_NAMES.
    is_general_dit = any(cls.__name__ == "GeneralDIT" for cls in type(model).__mro__)
    excluded_names = EXCLUDED_LAYER_NAMES
    if is_general_dit:
        excluded_names = tuple(EXCLUDED_LAYER_NAMES) + tuple(GENERALDIT_EXCLUDED_LAYER_NAMES)

    count = 0
    skipped_dims = 0
    for name, module in model.named_modules():
        if any(name.startswith(prefix) for prefix in targets):
            if _is_linear_layer(module):
                # Skip layers that can't be simply sharded (fused layers, modulation layers)
                if any(name.endswith(excl) for excl in excluded_names):
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
                elif name.endswith((".q", ".k", ".v", ".k_img", ".v_img")):
                    # Wan uses bare `.q`/`.k`/`.v` (not to_q / q_proj).
                    mode = "colwise"
                elif name.endswith((".ffn.0", ".layer1")):
                    # Wan MLP up-projection / Cosmos GPT2FeedForward.layer1
                    mode = "colwise"
                elif (
                    any(k in name for k in ROWWISE_KEYWORDS)
                    or name.endswith(".net.2")
                    or name.endswith(".o")  # Wan attention output proj
                    or name.endswith(".layer2")  # Cosmos GPT2FeedForward.layer2
                ):
                    # `.net.2` covers QwenImage's MLP down-projection
                    # (the GELU+Dropout+Linear ModuleList's index-2 Linear),
                    # matching Flux's `mlp.2` rowwise convention.
                    # `.o` covers Wan's self_attn.o / cross_attn.o.
                    # `.layer2` covers Cosmos GeneralDIT / MiniTrainDIT MLP down.
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

    if count == 0:
        logging.warning(f"[TP] parallelize_model: 0 layers parallelized for {model.__class__.__name__}"
                        f" ({skipped_dims} skipped for non-divisible dimensions) — TP had no effect")
        return False

    logging.info(f"[TP] Parallelized {count} linear layers across {len(targets)} block groups"
                 f" ({skipped_dims} skipped for non-divisible dimensions)")

    # Head-split override: for QwenImage/Cosmos/Wan/HiDream-style models, the
    # colwise shard distributes a contiguous slice of heads across ranks.
    # Each rank's output of to_q/to_k/to_v is `(heads/world_size) * dim_head`
    # channels, so the Attention module's head count must be divided by
    # world_size. Per-head norms stay full dim_head; full-dim QK norms
    # (Wan/HiDream) are sliced via `_slice_full_dim_qk_norms`.
    if head_split_active:
        head_overrides = 0
        norms_sliced = 0
        local_heads = None
        heads = None
        for module in model.modules():
            q_proj = _resolve_q_proj(module)
            if not isinstance(q_proj, ParallelLinear):
                continue
            heads_attr, heads = _get_attr_first(module, _HEADS_ATTRS)
            dim_attr, dim_head = _get_attr_first(module, _DIM_HEAD_ATTRS)
            if not isinstance(heads, int) or not isinstance(dim_head, int):
                continue
            if heads % mesh.world_size != 0:
                raise RuntimeError(
                    f"[TP] {model_class_name} attention heads={heads} not divisible by "
                    f"world_size={mesh.world_size}"
                )
            local_heads = heads // mesh.world_size
            if heads_attr is not None:
                setattr(module, heads_attr, local_heads)
            # Also keep sibling aliases in sync (some modules expose both).
            for alt in _HEADS_ATTRS:
                if alt != heads_attr and hasattr(module, alt):
                    setattr(module, alt, local_heads)
            norms_sliced += _slice_full_dim_qk_norms(module, local_heads, dim_head, mesh)
            head_overrides += 1
        if head_overrides:
            logging.info(
                f"[TP] Head-split override ({model_class_name}): set heads={local_heads} "
                f"(from {heads}) on {head_overrides} Attention modules"
                + (f"; sliced {norms_sliced} full-dim QK norms" if norms_sliced else "")
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
    loaded = 0
    missing = 0
    for name, module in model.named_modules():
        if isinstance(module, ParallelLinear):
            weight_key = prefix + name + ".weight"
            if weight_key in sd:
                full_weight = sd[weight_key]
                module.load_shard(full_weight)
                loaded += 1
            else:
                missing += 1
                logging.warning(f"[TP] Weight not found in state dict: '{weight_key}' — layer keeps uninitialized weights")

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
                else:
                    logging.warning(f"[TP] Bias not found in state dict: '{bias_key}'")

        # Reload full-dim QK norms that were sliced during parallelize_model.
        # The checkpoint holds the full weight; we re-slice by the marker set
        # on the module. (parallelize_model already sliced from the constructed
        # module weights; this path covers load-from-sd after a fresh construct.)
        shard_range = getattr(module, "_tp_norm_shard", None)
        if shard_range is not None and hasattr(module, "weight") and module.weight is not None:
            weight_key = prefix + name + ".weight"
            if weight_key in sd:
                full = sd[weight_key]
                start, end = shard_range
                if full.ndim == 1 and end <= full.shape[0]:
                    module.weight.data = full[start:end].to(
                        device=module.weight.device, dtype=module.weight.dtype
                    )

    if missing > 0:
        logging.error(f"[TP] load_tp_shards: {loaded} weights loaded, {missing} MISSING for {model.__class__.__name__} — outputs will be incorrect")
    else:
        logging.info(f"[TP] Loaded {loaded} TP weight shards for {model.__class__.__name__}")
