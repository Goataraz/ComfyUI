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
import torch.distributed as dist
import logging


class DeviceMesh:
    """Manages the mapping of model shards to GPUs in a distributed environment."""
    def __init__(self, devices: list[int] | None = None):
        if not dist.is_initialized():
            raise RuntimeError("Distributed process group must be initialized before creating DeviceMesh.")

        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()

        if devices is None:
            self.devices = list(range(self.world_size))
        else:
            self.devices = devices

        self.current_device = self.devices[self.rank]
        torch.cuda.set_device(self.current_device)
        logging.info(f"DeviceMesh initialized: Rank {self.rank}/{self.world_size} on GPU {self.current_device}")

    def get_device_for_rank(self, rank: int) -> int:
        return self.devices[rank]

    def get_shards_count(self) -> int:
        return self.world_size


# Global mesh instance
_mesh: DeviceMesh | None = None


def get_mesh() -> DeviceMesh:
    global _mesh
    if _mesh is None:
        _mesh = DeviceMesh()
    return _mesh


def init_mesh(devices: list[int] | None = None) -> DeviceMesh:
    global _mesh
    _mesh = DeviceMesh(devices)
    return _mesh