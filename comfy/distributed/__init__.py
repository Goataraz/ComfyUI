"""ComfyUI Tensor Parallelism — native model weight distribution across GPUs."""

from comfy.distributed.utils import is_tp_active, tp_aware_to, get_tp_param_names
from comfy.distributed.patcher import parallelize_model, load_tp_shards, get_tp_targets
from comfy.distributed.mesh import DeviceMesh, get_mesh
from comfy.distributed.parallel_linear import ParallelLinear
from comfy.distributed.null_server import NullServer, NullQueue