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
    "OpenAISignatureMMDITWrapper": ["joint_blocks"],  # SD3 MMDiT (not "blocks")
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
    "LTXVModel": ["transformer_blocks"],               # LTXV + LTXAV (MLP-only)
    "HunyuanVideo": ["double_blocks", "single_blocks"],  # packed double-stream QKV; linear1 stays
    "Chroma": ["double_blocks", "single_blocks"],          # Flux blocks, not a Flux subclass
    "ACEStepTransformer2DModel": ["transformer_blocks"],  # ACE-Step 1.0; FF is conv
    "AceStepConditionGenerationModel": ["decoder.layers"],  # ACE-Step 1.5 GQA DiT
    "NextDiT": ["layers", "noise_refiner", "context_refiner", "siglip_refiner"],
    "Ideogram4Transformer": ["layers"],
    "MiniMaxH3Model": ["blocks"],  # packed QKV + packed SwiGLU
    "JoyImageTransformer3DModel": ["double_blocks"],  # packed img/txt_attn_qkv
    "LensTransformer2DModel": ["transformer_blocks"],  # packed img_qkv/txt_qkv
    "PixDiT_T2I": ["patch_blocks", "pixel_blocks"],  # packed qkv_x/qkv_y + pixel qkv
    "AsymmDiTJoint": ["blocks"],  # Mochi: packed qkv_x/qkv_y + packed SwiGLU w1
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
    "o_proj", "down_proj", "mlp.layer2", "ffn.2", "fc2",
]
COLWISE_KEYWORDS = [
    "to_q", "to_k", "to_v", "up", "w1", "w3", "to_q_t", "to_k_t", "to_v_t",
    "q_proj", "k_proj", "v_proj", "gate_proj", "up_proj",
    "mlp.layer1", "fc1", "qkv_proj",
]

# Layer names that must NOT be sharded. These are excluded because:
# - Modulation layers (chunk() with full dim): modulation.lin, img_mod.lin, txt_mod.lin
# - Fused QKV+MLP layers (torch.split with full dim, unequal packs): linear1, linear2
# - Fused QKV projections (reshape with 3*num_heads): img_attn.qkv, txt_attn.qkv
#   Flux / HunyuanVideo / Chroma un-exclude these at runtime and shard with
#   packed colwise (pack_count=3). SD3 attn.qkv is also packed [Q|K|V] and
#   un-excluded at runtime for OpenAISignatureMMDITWrapper / MMDiT.
# - Coupled attention output projections: img_attn.proj, txt_attn.proj — stay
#   unsharded when qkv is excluded; become rowwise when qkv is packed-colwise.
EXCLUDED_LAYER_NAMES = [
    "linear1", "linear2",
    "modulation.lin", "img_mod.lin", "txt_mod.lin",
    "img_attn.qkv", "txt_attn.qkv",
    "img_attn.proj", "txt_attn.proj",
    # SD3 MMDiT fused QKV is packed as [Q|K|V] along out_features. The
    # default is to exclude (naive colwise cuts across packs). SD3 un-excludes
    # at runtime and uses packed colwise. Leading dot is required:
    # `img_attn.qkv`.endswith(`attn.qkv`) is True.
    ".attn.qkv", ".attn.proj",
    ".attn2.qkv", ".attn2.proj",
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

# LTXV / LTXAV-only. MUST NOT live in EXCLUDED_LAYER_NAMES:
# QwenImage shards `attn.to_q` (no Sequential). LTX RoPE is built from
# full `num_attention_heads` * `inner_dim`, so FA/CA stay replicated and
# only GELU MLP (`ff.net.0.proj` / `ff.net.2`) shards.
LTXV_EXCLUDED_LAYER_NAMES = (
    "to_q", "to_k", "to_v", "to_out.0", "to_gate_logits",
)

# Lumina NextDiT / Z-Image: adaLN_modulation.0 is the Z-Image single-Linear
# Sequential (chunked). Fused GQA qkv uses pack_sizes; attention.out is rowwise.
NEXTDIT_EXCLUDED_LAYER_NAMES = (
    "adaLN_modulation.0",
)

# Ideogram 4: adaLN is a bare Linear chunked 4-way. Packed qkv/o un-exclude
# at runtime (equal-width [Q|K|V], same view as Flux/SD3).
IDEOGRAM4_EXCLUDED_LAYER_NAMES = (
    "adaln_modulation",
)

# MiniMax H3 adaLN is a chunked Linear (expand * hidden * modalities).
MINIMAX_EXCLUDED_LAYER_NAMES = (
    "adaln_proj.linear",
)

# PixelDiT: 6-way / 3-way adaLN chunks stay full-width. compress/expand couple
# P^2 pixel tokens to attn_dim and must not shard independently of qkv in_features.
PIXELDiT_EXCLUDED_LAYER_NAMES = (
    "adaLN_modulation_img.0",
    "adaLN_modulation_txt.0",
    "adaLN_modulation_msa",
    "adaLN_modulation_mlp",
    "compress_to_attn",
    "expand_from_attn",
)

# Mochi AsymmDiTJoint: 4-way (or 1-way last-block) modulation stays full-width.
MOCHI_EXCLUDED_LAYER_NAMES = (
    "mod_x",
    "mod_y",
)


# Flux-family double-stream attention: equal-width packed [Q|K|V].
# Single-stream linear1 is QKV+MLP (unequal packs) and stays excluded.
_DOUBLE_STREAM_QKV_FAMILIES = frozenset({"Flux", "HunyuanVideo", "Chroma"})
_DOUBLE_STREAM_ATTN_TP = (
    "img_attn.qkv", "txt_attn.qkv", "img_attn.proj", "txt_attn.proj",
)
# SD3 MMDiT fused QKV is the same equal-width [Q|K|V] layout (split_qkv
# reshapes to (B, S, 3, heads, head_dim)). Un-exclude at runtime.
_SD3_QKV_FAMILIES = frozenset({"OpenAISignatureMMDITWrapper", "MMDiT"})
_SD3_ATTN_TP = (
    ".attn.qkv", ".attn.proj", ".attn2.qkv", ".attn2.proj",
)
_PACKED_QKV_FAMILIES = _DOUBLE_STREAM_QKV_FAMILIES | _SD3_QKV_FAMILIES | frozenset({
    "Ideogram4Transformer",
    "JoyImageTransformer3DModel",
    "LensTransformer2DModel",
    "PixDiT_T2I",
    "AsymmDiTJoint",
})


def _packed_colwise_count(name, mro_names):
    """Equal-sized output packs that must be sharded independently.

    MiniMax Attention.qkv_proj is ``[Q|K|V]`` (3). MiniMax MLP.fc1 is fused
    SwiGLU ``[gate|up]`` (2). Flux / Hunyuan / Chroma / SD3 / Ideogram /
    JoyImage / Lens fused ``*qkv`` is ``[Q|K|V]`` (3). Other architectures
    keep pack_count=1.
    """
    if "MiniMaxH3Model" in mro_names:
        if name.endswith("qkv_proj"):
            return 3
        if name.endswith("fc1"):
            return 2
        return 1
    if mro_names & _PACKED_QKV_FAMILIES:
        if (
            name.endswith(".qkv")
            or name.endswith("_qkv")
            or name.endswith(".qkv_x")
            or name.endswith(".qkv_y")
        ):
            return 3
        if "AsymmDiTJoint" in mro_names and name.endswith(".w1"):
            return 2
        return 1
    return 1


def _gqa_pack_sizes(name, module, parent, mro_names):
    """Unequal [Q heads | K kv | V kv] packs for NextDiT fused GQA qkv.

    Equal ``pack_count=3`` is wrong here: Q is wider than K/V when
    ``n_heads != n_kv_heads``. Returns None when this is not that layer.
    """
    if "NextDiT" not in mro_names:
        return None
    if not name.endswith("attention.qkv"):
        return None
    if parent is None:
        return None
    n_q = int(getattr(parent, "n_local_heads", 0) or 0)
    n_kv = int(getattr(parent, "n_local_kv_heads", 0) or 0)
    head_dim = int(getattr(parent, "head_dim", 0) or 0)
    if n_q <= 0 or n_kv <= 0 or head_dim <= 0:
        return None
    sizes = (n_q * head_dim, n_kv * head_dim, n_kv * head_dim)
    if sum(sizes) != module.out_features:
        return None
    return sizes


# Models where TP sharding splits HEADS (not head_dim). For these models
# each rank holds a contiguous slice of the colwise projection's output
# channels — i.e., a subset of heads with full per-head dim. The Attention
# module's `self.heads` / `n_heads` / `num_heads` must be divided by
# world_size so that the model's `.view(B, S, heads, dim_head)` reshape
# produces the correct (B, S, heads//ws, dim_head) layout.
#
# Norm policy:
# - Per-head norms (normalized_shape == dim_head, applied AFTER rearrange):
#   leave replicated. Covered by QwenImage / MiniTrainDIT.
# - Full-dim QK norms (normalized_shape == heads*dim_head, applied BEFORE
#   rearrange): must be sliced to local_heads*dim_head. Covered by Wan and
#   HiDream Image. See `_slice_full_dim_qk_norms`.
TP_HEAD_SPLIT_MODELS = {
    "QwenImageTransformer2DModel",
    "MiniTrainDIT",
    # GeneralDIT: attn excluded from TP (MLP-only); head-split not needed.
    "WanModel",
    "HiDreamImageTransformer2DModel",
    "ACEStepTransformer2DModel",
    "AceStepConditionGenerationModel",
    "MiniMaxH3Model",
    # Double-stream packed QKV. SingleStreamBlock.num_heads is skipped in
    # bookkeeping because fused linear1 stays full-width.
    "Flux",
    "HunyuanVideo",
    "Chroma",
    "OpenAISignatureMMDITWrapper",
    "MMDiT",
    "Ideogram4Transformer",
    "JoyImageTransformer3DModel",
    "LensTransformer2DModel",
    # Packed GQA qkv via pack_sizes; root n_heads stays (RoPE metadata).
    "NextDiT",
    "PixDiT_T2I",
    "AsymmDiTJoint",
}

# Attribute names used for the Q projection across architectures.
_Q_PROJ_ATTRS = ("to_q", "q_proj", "q", "qkv_proj", "qkv", "img_attn_qkv", "img_qkv", "qkv_x")
# Attribute names for head count / head dim.
_HEADS_ATTRS = ("heads", "n_heads", "num_heads", "num_attention_heads", "n_local_heads")
_KV_HEADS_ATTRS = ("num_kv_heads", "n_kv_heads", "kv_heads", "n_local_kv_heads")
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


def _slice_mochi_pos_frequencies(model, local_heads, original_heads, mesh):
    """Slice Mochi per-head RoPE frequencies to the local head shard.

    ``pos_frequencies`` is ``[3, num_heads, head_dim/2]``. After packed QKV
    head-split, ``compute_mixed_rotation`` must emit ``(N, local_heads, *)``.
    """
    pf = getattr(model, "pos_frequencies", None)
    if pf is None or not hasattr(pf, "data"):
        return 0
    if pf.ndim != 3 or pf.shape[1] != original_heads:
        return 0
    if original_heads % mesh.world_size != 0:
        return 0
    start = mesh.rank * local_heads
    end = start + local_heads
    model.pos_frequencies = nn.Parameter(
        pf.data[:, start:end].clone(),
        requires_grad=pf.requires_grad,
    )
    model._tp_pos_freq_shard = (start, end)
    return 1


# Models whose MRO would match a supported parent but that are NOT safe
# to shard yet (architecture-specific coupling outside Attention).
TP_UNSUPPORTED = {
    # HiDreamO1: integrated Llama2 LLM + vision encoder coupling.
    "HiDreamO1Transformer",
    # GeneralDIT is allowlisted with MLP-only TP (attn projections stay
    # replicated via GENERALDIT_EXCLUDED_LAYER_NAMES). Full FA/CA head-split
    # needs a dedicated strategy: CA K/V are Linear(context→inner) while FA
    # is Linear(query→inner), so naive colwise+head-split is unsafe.
}


def _sync_bookkeeping_heads(model, original_heads, local_heads, targets=()):
    """Divide leftover head-count bookkeeping that is not on Attention modules.

    Head-split updates modules that own a colwise Q projection. CausalWan also
    stores ``num_heads`` on the outer model (KV cache allocation) and on
    ``WanAttentionBlock`` (cross-attn ``optimized_attention(..., heads=)``).
    Those copies would stay at the full-model head count and blow cache /
    attention shapes. Sync remaining ``heads`` / ``n_heads`` / ``num_heads``
    that still equal the pre-split value — on the root module and under TP
    target prefixes only (ACE 1.5 lyric/timbre encoders stay full-width).
    """
    if original_heads == local_heads:
        return 0
    mro_names = {cls.__name__ for cls in type(model).__mro__}
    skip_root_heads = bool(mro_names & _PACKED_QKV_FAMILIES) or "NextDiT" in mro_names
    synced = 0
    for name, module in model.named_modules():
        if name and not _name_under_targets(name, targets):
            continue
        # Attention modules with a sharded Q proj were already rewritten.
        # Do not reuse another block's original_heads (PixelDiT mixes 24 and 16).
        if isinstance(_resolve_q_proj(module), ParallelLinear):
            continue
        # Fused single-stream QKV+MLP stays replicated; PiTBlock.num_heads is
        # RoPE metadata (attn_dim // num_heads) and must stay full.
        if type(module).__name__ in ("SingleStreamBlock", "PiTBlock"):
            continue
        # Root Flux/Hunyuan/Chroma/SD3 num_heads and NextDiT.n_heads are
        # constructor / RoPE metadata — attention modules own the reshape.
        if skip_root_heads and not name:
            continue
        for attr in _HEADS_ATTRS:
            val = getattr(module, attr, None)
            if val == original_heads:
                setattr(module, attr, local_heads)
                synced += 1
    return synced


def _name_under_targets(name, targets):
    if not name:
        return False
    return any(name == t or name.startswith(t + ".") for t in targets)


def _sync_kv_heads(model, targets, world_size):
    """Divide GQA KV head counts on modules inside TP target prefixes.

    Head-split only rewrites attrs that match the Q-head count. ACE-Step 1.5
    uses ``num_kv_heads != num_heads`` (16/8). ``k_proj`` is still colwise-
    sharded, so KV bookkeeping must be divided independently — but only under
    the TP prefixes, or lyric/timbre encoders outside the DiT get corrupted.
    """
    if world_size <= 1:
        return 0
    synced = 0
    for name, module in model.named_modules():
        if not _name_under_targets(name, targets):
            continue
        for attr in _KV_HEADS_ATTRS:
            val = getattr(module, attr, None)
            if isinstance(val, int) and val >= world_size and val % world_size == 0:
                setattr(module, attr, val // world_size)
                synced += 1
    return synced


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

# Companion keys that mark a Linear as scaled/quantized FP8 (or similar).
# ParallelLinear does not carry these; sharding the weight alone drops the
# scale → broken activations (e.g. SD3.5 fp8_scaled mean ~150 → ~79).
_SCALED_COMPANION_SUFFIXES = (
    ".weight_scale",
    ".input_scale",
    ".weight_scale_inv",
    ".comfy_quant",
)


def _has_scaled_companions(sd, prefix, module_name):
    """True if state dict has FP8/quant scale (or similar) keys for this module."""
    if sd is None:
        return False
    base = f"{prefix}{module_name}"
    return any((base + suffix) in sd for suffix in _SCALED_COMPANION_SUFFIXES)


def parallelize_model(model, sd=None, prefix=""):
    """
    Replaces target linear layers in the model with ParallelLinear.

    Args:
        model: Diffusion transformer (inner model, not BaseModel wrapper).
        sd: Optional full state dict. When provided, layers with companion
            FP8/quant scale keys are left as plain Linear so scales load.
        prefix: State-dict key prefix matching ``sd`` (e.g. diffusion prefix).
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

    # Class-scoped attention exclusions (must not live in the global list —
    # suffix collisions across architectures, e.g. Qwen vs CosmOS to_out.0).
    mro_names = {cls.__name__ for cls in type(model).__mro__}
    excluded_names = tuple(EXCLUDED_LAYER_NAMES)
    if "GeneralDIT" in mro_names:
        excluded_names = excluded_names + tuple(GENERALDIT_EXCLUDED_LAYER_NAMES)
    if "LTXVModel" in mro_names:
        excluded_names = excluded_names + tuple(LTXV_EXCLUDED_LAYER_NAMES)
    if "NextDiT" in mro_names:
        excluded_names = excluded_names + tuple(NEXTDIT_EXCLUDED_LAYER_NAMES)
    if "Ideogram4Transformer" in mro_names:
        excluded_names = excluded_names + tuple(IDEOGRAM4_EXCLUDED_LAYER_NAMES)
    if "MiniMaxH3Model" in mro_names:
        excluded_names = excluded_names + tuple(MINIMAX_EXCLUDED_LAYER_NAMES)
    if "PixDiT_T2I" in mro_names:
        excluded_names = excluded_names + tuple(PIXELDiT_EXCLUDED_LAYER_NAMES)
        # RotaryAttention.qkv / .proj collide with the SD3 `.attn.qkv` suffix.
        excluded_names = tuple(e for e in excluded_names if e not in (".attn.qkv", ".attn.proj"))
    if "AsymmDiTJoint" in mro_names:
        excluded_names = excluded_names + tuple(MOCHI_EXCLUDED_LAYER_NAMES)
    if mro_names & _DOUBLE_STREAM_QKV_FAMILIES:
        excluded_names = tuple(e for e in excluded_names if e not in _DOUBLE_STREAM_ATTN_TP)
    if mro_names & _SD3_QKV_FAMILIES:
        excluded_names = tuple(e for e in excluded_names if e not in _SD3_ATTN_TP)

    count = 0
    skipped_dims = 0
    skipped_scaled = 0
    for name, module in model.named_modules():
        if any(name.startswith(t) for t in targets):
            if _is_linear_layer(module):
                # Skip layers that can't be simply sharded (fused layers, modulation layers)
                if any(name.endswith(excl) for excl in excluded_names):
                    continue

                # Skip scaled/quantized linears — ParallelLinear has no weight_scale
                if _has_scaled_companions(sd, prefix, name):
                    skipped_scaled += 1
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
                    or name.endswith(".attention.out")  # NextDiT JointAttention.out
                ):
                    # `.net.2` covers QwenImage's MLP down-projection
                    # (the GELU+Dropout+Linear ModuleList's index-2 Linear),
                    # matching Flux's `mlp.2` rowwise convention.
                    # `.o` covers Wan's self_attn.o / cross_attn.o.
                    # `.layer2` covers Cosmos GeneralDIT / MiniTrainDIT MLP down.
                    # `.attention.out` is NextDiT's attention output (not `.o`).
                    mode = "rowwise"

                parent_name, _, child_name = name.rpartition('.')
                parent = model
                if parent_name:
                    parent = comfy.utils.get_attr(model, parent_name)

                pack_sizes = _gqa_pack_sizes(name, module, parent, mro_names) if mode == "colwise" else None
                is_nextdit_qkv = "NextDiT" in mro_names and name.endswith("attention.qkv")
                pack_count = 1
                if is_nextdit_qkv:
                    # Never fall through to naive colwise — that cuts across [Q|K|V].
                    n_q = int(getattr(parent, "n_local_heads", 0) or 0)
                    n_kv = int(getattr(parent, "n_local_kv_heads", 0) or 0)
                    if (
                        pack_sizes is None
                        or n_q % mesh.world_size != 0
                        or n_kv % mesh.world_size != 0
                        or any(size % mesh.world_size != 0 for size in pack_sizes)
                    ):
                        skipped_dims += 1
                        continue
                elif mode == "colwise":
                    pack_count = _packed_colwise_count(name, mro_names)
                    if pack_count > 1:
                        if module.out_features % pack_count != 0:
                            skipped_dims += 1
                            continue
                        pack_size = module.out_features // pack_count
                        if pack_size % mesh.world_size != 0:
                            skipped_dims += 1
                            continue
                    else:
                        if module.out_features % mesh.world_size != 0:
                            skipped_dims += 1
                            continue
                else:
                    if module.in_features % mesh.world_size != 0:
                        skipped_dims += 1
                        continue

                new_layer = ParallelLinear(
                    in_features=module.in_features,
                    out_features=module.out_features,
                    bias=module.bias is not None,
                    mode=mode,
                    pack_count=pack_count,
                    pack_sizes=pack_sizes,
                )

                setattr(parent, child_name, new_layer)
                count += 1

    if count == 0:
        logging.warning(
            f"[TP] parallelize_model: 0 layers parallelized for {model.__class__.__name__}"
            f" ({skipped_dims} skipped for non-divisible dimensions"
            f", {skipped_scaled} skipped for scaled/quant companions) — TP had no effect"
        )
        return False

    logging.info(
        f"[TP] Parallelized {count} linear layers across {len(targets)} block groups"
        f" ({skipped_dims} skipped for non-divisible dimensions"
        f", {skipped_scaled} skipped for scaled/quant companions)"
    )

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
            if not isinstance(heads, int):
                continue
            if heads % mesh.world_size != 0:
                raise RuntimeError(
                    f"[TP] {model_class_name} attention heads={heads} not divisible by "
                    f"world_size={mesh.world_size}"
                )
            local_heads = heads // mesh.world_size
            if not isinstance(dim_head, int):
                # ACE-Step 1.0 Attention stores heads but not dim_head.
                # Packed QKV local_out is pack_count * local_heads * dim_head.
                local_out = getattr(q_proj, "local_out_features", None)
                pack = getattr(q_proj, "pack_count", 1) or 1
                denom = local_heads * pack
                if denom == 0 or not isinstance(local_out, int) or local_out % denom != 0:
                    continue
                dim_head = local_out // denom
            if heads_attr is not None:
                setattr(module, heads_attr, local_heads)
            # Also keep sibling aliases in sync (some modules expose both).
            for alt in _HEADS_ATTRS:
                if alt != heads_attr and hasattr(module, alt):
                    setattr(module, alt, local_heads)
            norms_sliced += _slice_full_dim_qk_norms(module, local_heads, dim_head, mesh)
            head_overrides += 1
        if head_overrides:
            original_heads = local_heads * mesh.world_size
            synced = _sync_bookkeeping_heads(model, original_heads, local_heads, targets)
            kv_synced = _sync_kv_heads(model, targets, mesh.world_size)
            pos_sliced = 0
            if "AsymmDiTJoint" in {cls.__name__ for cls in type(model).__mro__}:
                pos_sliced = _slice_mochi_pos_frequencies(model, local_heads, original_heads, mesh)
            logging.info(
                f"[TP] Head-split override ({model_class_name}): set heads={local_heads} "
                f"(from {heads}) on {head_overrides} Attention modules"
                + (f"; sliced {norms_sliced} full-dim QK norms" if norms_sliced else "")
                + (f"; synced {synced} leftover head-count attrs" if synced else "")
                + (f"; synced {kv_synced} GQA kv-head attrs" if kv_synced else "")
                + (f"; sliced pos_frequencies heads [{mesh.rank * local_heads}:{mesh.rank * local_heads + local_heads}]" if pos_sliced else "")
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
                full_weight = sd.pop(weight_key)
                module.load_shard(full_weight)
                loaded += 1
            else:
                missing += 1
                logging.warning(f"[TP] Weight not found in state dict: '{weight_key}' — layer keeps uninitialized weights")

            if module.bias is not None:
                bias_key = prefix + name + ".bias"
                if bias_key in sd:
                    full_bias = sd.pop(bias_key)
                    if module.mode == "colwise":
                        module.load_bias_shard(full_bias)
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
                full = sd.pop(weight_key)
                start, end = shard_range
                if full.ndim == 1 and end <= full.shape[0]:
                    module.weight.data = full[start:end].to(
                        device=module.weight.device, dtype=module.weight.dtype
                    )

        pos_range = getattr(module, "_tp_pos_freq_shard", None)
        pf = getattr(module, "pos_frequencies", None)
        if pos_range is not None and pf is not None:
            pf_key = prefix + ("pos_frequencies" if not name else name + ".pos_frequencies")
            if pf_key in sd:
                full = sd.pop(pf_key)
                start, end = pos_range
                if full.ndim == 3 and end <= full.shape[1]:
                    pf.data = full[:, start:end].to(device=pf.device, dtype=pf.dtype)

    if missing > 0:
        logging.error(f"[TP] load_tp_shards: {loaded} weights loaded, {missing} MISSING for {model.__class__.__name__} — outputs will be incorrect")
    else:
        logging.info(f"[TP] Loaded {loaded} TP weight shards for {model.__class__.__name__}")
