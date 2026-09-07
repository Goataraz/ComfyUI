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
import logging
from comfy.distributed.mesh import get_mesh

_TP_SHARD_META_ATTRS = (
    "_tp_pack_count",
    "_tp_rank",
    "_tp_pack_size",
    "_tp_local_pack_size",
    "_tp_mode",
)


def stamp_tp_shard_meta(tensor, module):
    """Copy ParallelLinear shard layout onto a tensor for LoRA slicing."""
    tensor._tp_pack_count = int(getattr(module, "pack_count", 1) or 1)
    tensor._tp_rank = int(getattr(module, "rank", 0) or 0)
    tensor._tp_pack_size = getattr(module, "pack_size", None)
    tensor._tp_local_pack_size = getattr(module, "local_pack_size", None)
    tensor._tp_mode = getattr(module, "mode", None)
    return tensor


def copy_tp_shard_meta(src, dst):
    """Preserve TP layout across ``.to(copy=True)`` / cast copies."""
    for attr in _TP_SHARD_META_ATTRS:
        if hasattr(src, attr):
            setattr(dst, attr, getattr(src, attr))
    return dst


def packed_colwise_row_slices(rank, pack_count, pack_size, local_pack_size):
    """Output-dim slices for this rank — one per equal-sized pack."""
    pack_count = int(pack_count) if pack_count else 1
    if pack_count <= 1:
        start = rank * local_pack_size
        return (slice(start, start + local_pack_size),)
    slices = []
    for p in range(pack_count):
        start = p * pack_size + rank * local_pack_size
        slices.append(slice(start, start + local_pack_size))
    return tuple(slices)


def shard_like_tp_weight(full, shard):
    """Slice a full-width LoRA diff to match a TP weight/bias shard.

    Packed colwise concatenates per-pack slices (same layout as
    ``ParallelLinear.load_shard``). Missing metadata falls back to a naive
    contiguous cut using mesh rank.
    """
    mode = getattr(shard, "_tp_mode", None)
    pack_count = int(getattr(shard, "_tp_pack_count", 1) or 1)
    rank = getattr(shard, "_tp_rank", None)
    pack_size = getattr(shard, "_tp_pack_size", None)
    local_pack_size = getattr(shard, "_tp_local_pack_size", None)
    if rank is None or pack_size is None or local_pack_size is None:
        from comfy.distributed.mesh import get_mesh
        mesh = get_mesh()
        rank = mesh.rank
        if full.ndim == 1:
            local_pack_size = shard.shape[0]
            pack_size = full.shape[0] if pack_count <= 1 else full.shape[0] // pack_count
        elif mode == "rowwise" or (
            full.ndim == 2 and full.shape[0] == shard.shape[0] and full.shape[1] != shard.shape[1]
        ):
            start = rank * shard.shape[1]
            return full[:, start:start + shard.shape[1]]
        else:
            local_pack_size = shard.shape[0] if pack_count <= 1 else shard.shape[0] // pack_count
            pack_size = full.shape[0] if pack_count <= 1 else full.shape[0] // pack_count
    slices = packed_colwise_row_slices(rank, pack_count, pack_size, local_pack_size)
    if full.ndim == 1:
        pieces = [full[s] for s in slices]
        return pieces[0] if len(pieces) == 1 else torch.cat(pieces, dim=0)
    if mode == "rowwise":
        start = rank * shard.shape[1]
        return full[:, start:start + shard.shape[1]]
    pieces = [full[s, :] for s in slices]
    return pieces[0] if len(pieces) == 1 else torch.cat(pieces, dim=0)


def slice_colwise_activation(tensor, module):
    """Slice a full last-dim activation to this rank's packed colwise shard."""
    pieces = [tensor[..., s] for s in module._packed_colwise_slices()]
    return pieces[0] if len(pieces) == 1 else torch.cat(pieces, dim=-1)


class ParallelLinear(nn.Module):
    """A linear layer that supports Tensor Parallelism.
    Can be configured as Colwise or Rowwise.

    ``pack_count`` > 1 (colwise only) shards *inside* each equal-sized pack
    along the output dim, then concatenates. That is the correct layout for
    fused QKV (pack=3, ``[Q|K|V]``) and fused SwiGLU (pack=2, ``[gate|up]``).
    Naive colwise would cut across packs.
    """
    def __init__(self, in_features, out_features, bias=True, mode="colwise", pack_count=1):
        super().__init__()
        self.mesh = get_mesh()
        self.rank = self.mesh.rank
        self.world_size = self.mesh.world_size
        self.mode = mode
        self.pack_count = int(pack_count) if pack_count else 1
        if self.pack_count < 1:
            raise ValueError(f"pack_count must be >= 1, got {pack_count}")
        if self.pack_count > 1 and mode != "colwise":
            raise ValueError("pack_count > 1 is only valid for colwise layers")

        # Determine local dimensions based on TP mode
        if mode == "colwise":
            if self.pack_count > 1:
                if out_features % self.pack_count != 0:
                    raise ValueError(
                        f"[TP] packed colwise: out_features={out_features} not divisible by "
                        f"pack_count={self.pack_count}"
                    )
                pack_size = out_features // self.pack_count
                if pack_size % self.world_size != 0:
                    raise ValueError(
                        f"[TP] packed colwise: pack_size={pack_size} not divisible by "
                        f"world_size={self.world_size}"
                    )
                self.pack_size = pack_size
                self.local_pack_size = pack_size // self.world_size
                self.local_out_features = self.pack_count * self.local_pack_size
                self.local_in_features = in_features
            else:
                assert out_features % self.world_size == 0, (
                    f"[TP] colwise: out_features={out_features} not divisible by world_size={self.world_size}"
                )
                self.pack_size = out_features
                self.local_pack_size = out_features // self.world_size
                self.local_out_features = out_features // self.world_size
                self.local_in_features = in_features
        elif mode == "rowwise":
            assert in_features % self.world_size == 0, (
                f"[TP] rowwise: in_features={in_features} not divisible by world_size={self.world_size}"
            )
            self.local_in_features = in_features // self.world_size
            self.local_out_features = out_features
            self.pack_size = out_features
            self.local_pack_size = out_features
        else:
            raise ValueError(f"Invalid TP mode: {mode}. Must be 'colwise' or 'rowwise'.")

        # Allocate on CPU to avoid OOM — shards are moved to GPU in load_shard()
        self.weight = nn.Parameter(torch.empty(
            (self.local_out_features, self.local_in_features),
            dtype=torch.float32,
            device="cpu"
        ))
        stamp_tp_shard_meta(self.weight, self)

        self.is_tp_parallelized = True

        # LoRA / weight-patch hooks (populated by the model patcher, e.g.
        # LowVramPatch under DynamicVRAM). Empty by default.
        self.weight_function = []
        self.bias_function = []

        if bias:
            self.bias = nn.Parameter(torch.empty(
                self.local_out_features,
                dtype=torch.float32,
                device="cpu"
            ))
            stamp_tp_shard_meta(self.bias, self)
        else:
            self.register_parameter('bias', None)

    _forward_debug = False  # Set to True to debug shape mismatches

    def forward(self, x):
        # Cast weight to match input dtype for manual-cast models (e.g., fp8 weights with fp16 compute)
        # This handles the case where ComfyUI's manual casting system sets the compute dtype
        # but TP weights retain their storage dtype (fp8_e4m3fn, etc.)
        # Apply weight_function (LoRA patches via LowVramPatch) if present.
        # The standard comfy ops check len(weight_function) > 0 and redirect
        # to forward_comfy_cast_weights — ParallelLinear must do the same
        # or LoRA patches registered as LowVramPatch would be silently ignored.
        weight_function = getattr(self, 'weight_function', [])
        bias_function = getattr(self, 'bias_function', [])

        # When weight/bias functions are present, force a fresh copy on cast so
        # a same-dtype .to() returns the real Parameter storage and the patch
        # mutates a throwaway tensor rather than accumulating into it every forward.
        w = self.weight.to(dtype=x.dtype, copy=True) if len(weight_function) > 0 else self.weight.to(x.dtype)
        if len(weight_function) > 0:
            stamp_tp_shard_meta(w, self)
            copy_tp_shard_meta(self.weight, w)
        if self.bias is not None:
            b = self.bias.to(dtype=x.dtype, copy=True) if len(bias_function) > 0 else self.bias.to(x.dtype)
            if len(bias_function) > 0:
                stamp_tp_shard_meta(b, self)
                copy_tp_shard_meta(self.bias, b)
        else:
            b = None

        if len(weight_function) > 0:
            for f in weight_function:
                w = f(w)
        if b is not None and len(bias_function) > 0:
            for f in bias_function:
                b = f(b)

        if ParallelLinear._forward_debug:
            logging.debug(f"[TP] {self.mode} forward: x.shape={x.shape}, w.shape={w.shape}, "
                          f"local_in={self.local_in_features}, local_out={self.local_out_features}")

        if self.mode == "colwise":
            # Column Parallelism: Each GPU computes a shard of the output
            res = torch.matmul(x, w.t())
            if b is not None:
                res += b
            return res

        elif self.mode == "rowwise":
            # Row Parallelism: Each GPU computes a partial sum, then all-reduce.
            # If the input has more features than this rank's shard (e.g.,
            # context from a non-TP text encoder in cross-attention), slice
            # it to this rank's input shard before the matmul.
            if x.shape[-1] > self.local_in_features:
                start = self.rank * self.local_in_features
                end = start + self.local_in_features
                x = x[..., start:end]
            res = torch.matmul(x, w.t())
            dist.all_reduce(res, op=dist.ReduceOp.SUM)
            if b is not None:
                res += b
            return res

    def _packed_colwise_slices(self):
        """Slices along the output dim for this rank, one per pack."""
        return packed_colwise_row_slices(
            self.rank, self.pack_count, self.pack_size, self.local_pack_size,
        )

    def load_shard(self, full_weight_tensor):
        """Slices the full weight tensor and loads the shard for this rank."""
        if self.mode == "colwise":
            pieces = [full_weight_tensor[s, :] for s in self._packed_colwise_slices()]
            shard = pieces[0] if len(pieces) == 1 else torch.cat(pieces, dim=0)
        elif self.mode == "rowwise":
            start = self.rank * self.local_in_features
            end = start + self.local_in_features
            shard = full_weight_tensor[:, start:end]
        else:
            raise RuntimeError(f"[TP] load_shard: unexpected mode '{self.mode}'")

        self.weight.data = shard.to(device=self.mesh.current_device)

    def load_bias_shard(self, full_bias_tensor):
        """Slice a full bias the same way as the colwise weight output dim."""
        if self.mode != "colwise":
            self.bias.data = full_bias_tensor.to(device=self.mesh.current_device)
            return
        pieces = [full_bias_tensor[s] for s in self._packed_colwise_slices()]
        shard = pieces[0] if len(pieces) == 1 else torch.cat(pieces, dim=0)
        self.bias.data = shard.to(device=self.mesh.current_device)