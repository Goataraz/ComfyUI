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
from comfy.distributed.mesh import get_mesh


class ParallelLinear(nn.Module):
    """A linear layer that supports Tensor Parallelism.
    Can be configured as Colwise or Rowwise."""
    def __init__(self, in_features, out_features, bias=True, mode="colwise"):
        super().__init__()
        self.mesh = get_mesh()
        self.rank = self.mesh.rank
        self.world_size = self.mesh.world_size
        self.mode = mode

        # Determine local dimensions based on TP mode
        if mode == "colwise":
            self.local_out_features = out_features // self.world_size
            self.local_in_features = in_features
        elif mode == "rowwise":
            self.local_in_features = in_features // self.world_size
            self.local_out_features = out_features
        else:
            raise ValueError(f"Invalid TP mode: {mode}. Must be 'colwise' or 'rowwise'.")

        # Allocate on CPU to avoid OOM — shards are moved to GPU in load_shard()
        self.weight = nn.Parameter(torch.empty(
            (self.local_out_features, self.local_in_features),
            dtype=torch.float32,
            device="cpu"
        ))

        self.is_tp_parallelized = True

        if bias:
            self.bias = nn.Parameter(torch.empty(
                self.local_out_features,
                dtype=torch.float32,
                device="cpu"
            ))
        else:
            self.register_parameter('bias', None)

    def forward(self, x):
        # Cast weight to match input dtype for manual-cast models (e.g., fp8 weights with fp16 compute)
        # This handles the case where ComfyUI's manual casting system sets the compute dtype
        # but TP weights retain their storage dtype (fp8_e4m3fn, etc.)
        w = self.weight.to(x.dtype)

        if self.mode == "colwise":
            # Column Parallelism: Each GPU computes a shard of the output
            res = torch.matmul(x, w.t())
            if self.bias is not None:
                res += self.bias.to(x.dtype)
            return res

        elif self.mode == "rowwise":
            # Row Parallelism: Each GPU computes a partial sum, then all-reduce
            res = torch.matmul(x, w.t())
            dist.all_reduce(res, op=dist.ReduceOp.SUM)
            if self.bias is not None:
                res += self.bias.to(x.dtype)
            return res

    def load_shard(self, full_weight_tensor):
        """Slices the full weight tensor and loads the shard for this rank."""
        out_f, in_f = full_weight_tensor.shape

        if self.mode == "colwise":
            start = self.rank * self.local_out_features
            end = start + self.local_out_features
            shard = full_weight_tensor[start:end, :]
        elif self.mode == "rowwise":
            start = self.rank * self.local_in_features
            end = start + self.local_in_features
            shard = full_weight_tensor[:, start:end]

        self.weight.data = shard.to(device=self.mesh.current_device)