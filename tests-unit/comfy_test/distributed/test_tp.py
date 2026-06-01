"""Unit tests for ComfyUI Tensor Parallelism utilities.

These tests run without GPUs or torch.distributed — they verify
the pure-Python logic for model matching, NullProxy, NullServer,
and the is_tp_active guard.
"""

import pytest
from unittest.mock import patch


# ---------------------------------------------------------------------------
# NullProxy / NullServer tests
# ---------------------------------------------------------------------------

class TestNullProxy:
    """Verify NullProxy safely handles chained attribute access and calls."""

    def test_chained_attr_access(self):
        from comfy.distributed.null_server import NullProxy
        proxy = NullProxy()
        # Deeply chained access should never crash
        result = proxy.a.b.c.d
        assert isinstance(result, NullProxy)

    def test_call_returns_proxy(self):
        from comfy.distributed.null_server import NullProxy
        proxy = NullProxy()
        result = proxy.some.method("arg", keyword="value")
        # Calling a NullProxy returns another NullProxy (chainable, awaitable)
        assert isinstance(result, NullProxy)

    def test_awaitable(self):
        import asyncio
        from comfy.distributed.null_server import NullProxy

        async def run():
            proxy = NullProxy()
            result = await proxy.async_method("arg")
            # Awaited NullProxy returns None
            assert result is None

        asyncio.run(run())

    def test_call_returns_null_proxy(self):
        from comfy.distributed.null_server import NullProxy
        proxy = NullProxy()
        result = proxy.some_method("arg")
        # Calling a NullProxy returns another NullProxy (which is awaitable)
        assert isinstance(result, NullProxy)

    def test_bool_is_false(self):
        from comfy.distributed.null_server import NullProxy
        proxy = NullProxy()
        assert not proxy
        assert bool(proxy) is False

    def test_int_is_zero(self):
        from comfy.distributed.null_server import NullProxy
        proxy = NullProxy()
        assert int(proxy) == 0

    def test_iadd_is_noop(self):
        from comfy.distributed.null_server import NullProxy
        proxy = NullProxy()
        proxy += 42  # Should not raise
        assert isinstance(proxy, NullProxy)

    def test_setattr_is_noop(self):
        from comfy.distributed.null_server import NullProxy
        proxy = NullProxy()
        proxy.some_attr = "value"  # Should not raise
        # The attribute should NOT be stored (NullProxy discards setattr)
        assert isinstance(proxy, NullProxy)


class TestNullServer:
    """Verify NullServer mirrors the PromptServer interface."""

    def test_explicit_methods_exist(self):
        from comfy.distributed.null_server import NullServer
        server = NullServer()
        # These should all work without crashing
        server.send_sync("event", {})
        server.queue_updated()
        server.send_progress_text("text", "node1")
        server.add_on_prompt_handler(lambda x: x)
        result = server.get_queue_info()
        assert result == {}
        result = server.trigger_on_prompt({"key": "value"})
        assert result == {"key": "value"}

    def test_async_send(self):
        import asyncio
        from comfy.distributed.null_server import NullServer

        async def run():
            server = NullServer()
            result = await server.send("event", {"data": 1})
            assert result is None

        asyncio.run(run())

    def test_unknown_attr_returns_null_proxy(self):
        from comfy.distributed.null_server import NullServer, NullProxy
        server = NullServer()
        result = server.nonexistent_method("arg")
        # Unknown methods return NullProxy (chainable, awaitable)
        assert isinstance(result, NullProxy)

    def test_instance_refers_to_self(self):
        from comfy.distributed.null_server import NullServer
        server = NullServer()
        assert server.instance is server

    def test_routes_decorator_passthrough(self):
        from comfy.distributed.null_server import NullServer
        server = NullServer()
        # Route decorators should pass the function through
        @server.routes.get("/test")
        async def handler(request):
            return "ok"
        assert handler is not None

    def test_app_is_null_proxy(self):
        from comfy.distributed.null_server import NullServer, NullProxy
        server = NullServer()
        assert isinstance(server.app, NullProxy)


class TestNullQueue:
    """Verify NullQueue behaves as a no-op queue."""

    def test_get_returns_none(self):
        from comfy.distributed.null_server import NullQueue
        q = NullQueue()
        assert q.get() is None

    def test_put_is_noop(self):
        from comfy.distributed.null_server import NullQueue
        q = NullQueue()
        q.put("item")  # Should not raise

    def test_currently_running(self):
        from comfy.distributed.null_server import NullQueue
        q = NullQueue()
        assert q.currently_running is False

    def test_get_flags_returns_empty(self):
        from comfy.distributed.null_server import NullQueue
        q = NullQueue()
        assert q.get_flags() == {}


# ---------------------------------------------------------------------------
# Model matching tests (no GPU required)
# ---------------------------------------------------------------------------

class TestGetTPTargets:
    """Verify MRO-based model matching for Tensor Parallelism."""

    def _make_model_class(self, name, bases=()):
        """Create a synthetic model class with a given name."""
        return type(name, bases, {})

    def test_flux_matches(self):
        from comfy.distributed.patcher import get_tp_targets
        Flux = self._make_model_class("Flux")
        assert get_tp_targets(Flux()) == ["double_blocks", "single_blocks"]

    def test_sd3_matches_openai_wrapper(self):
        from comfy.distributed.patcher import get_tp_targets
        SD3 = self._make_model_class("OpenAISignatureMMDITWrapper")
        assert get_tp_targets(SD3()) == ["blocks"]

    def test_cosmos_general_dit(self):
        from comfy.distributed.patcher import get_tp_targets
        CosmosT2V = self._make_model_class("GeneralDIT")
        assert get_tp_targets(CosmosT2V()) == ["blocks"]

    def test_cosmos_mini_train_dit(self):
        from comfy.distributed.patcher import get_tp_targets
        CosmosP2 = self._make_model_class("MiniTrainDIT")
        assert get_tp_targets(CosmosP2()) == ["blocks"]

    def test_hidream_image_matches(self):
        from comfy.distributed.patcher import get_tp_targets
        HiDream = self._make_model_class("HiDreamImageTransformer2DModel")
        assert get_tp_targets(HiDream()) == ["double_stream_blocks", "single_stream_blocks"]

    def test_hidream_o1_matches(self):
        from comfy.distributed.patcher import get_tp_targets
        HiDreamO1 = self._make_model_class("HiDreamO1Transformer")
        # HiDreamO1 is intentionally disabled in TP_TARGETS because its integrated
        # Llama2 LLM can't be naively sharded (needs full-model TP strategy)
        assert get_tp_targets(HiDreamO1()) == []

    def test_mro_subclass_inherits(self):
        from comfy.distributed.patcher import get_tp_targets
        """Subclass of a matched class should inherit the TP targets."""
        GeneralDIT = self._make_model_class("GeneralDIT")
        Anima = self._make_model_class("Anima", (GeneralDIT,))
        assert get_tp_targets(Anima()) == ["blocks"]

    def test_unknown_model_returns_empty(self):
        from comfy.distributed.patcher import get_tp_targets
        Unknown = self._make_model_class("SomeRandomModel")
        assert get_tp_targets(Unknown()) == []

    def test_mro_priority_first_match(self):
        from comfy.distributed.patcher import get_tp_targets
        """When a class inherits from multiple TP targets, the first MRO match wins."""
        GeneralDIT = self._make_model_class("GeneralDIT")
        HiDreamO1 = self._make_model_class("HiDreamO1Transformer")
        # Create a class that inherits from both — first in MRO wins
        Hybrid = self._make_model_class("HybridModel", (GeneralDIT, HiDreamO1))
        result = get_tp_targets(Hybrid())
        # GeneralDIT appears first in MRO after HybridModel and object
        assert result == ["blocks"]


# ---------------------------------------------------------------------------
# is_tp_active tests
# ---------------------------------------------------------------------------

class TestIsTPActive:
    """Verify is_tp_active correctly detects TP state."""

    def test_returns_false_when_dist_not_initialized(self):
        from comfy.distributed.utils import is_tp_active
        # Without torch.distributed initialized, should return False
        assert is_tp_active() is False

    def test_returns_false_when_import_fails(self):
        from comfy.distributed.utils import is_tp_active
        # Should not crash even if torch.distributed is unavailable
        with patch.dict("sys.modules", {"torch.distributed": None}):
            # This would raise ImportError on import, which is caught
            assert is_tp_active() is False


# ---------------------------------------------------------------------------
# QwenImage + Llama2 TP allowlist tests
# ---------------------------------------------------------------------------

class TestQwenImageTP:
    """Verify QwenImageTransformer2DModel and Llama2 are matched by the TP allowlist."""

    def test_qwen_image_transformer_targets_identified(self):
        from comfy.distributed.patcher import get_tp_targets
        Qwen = type("QwenImageTransformer2DModel", (), {})
        assert get_tp_targets(Qwen()) == ["transformer_blocks"]

    def test_qwen_image_modulation_layers_excluded(self):
        """img_mod.1 / txt_mod.1 are inside nn.Sequential(SiLU, Linear) and must not
        be replaced — the outer Sequential in QwenImageTransformerBlock feeds
        6*dim through chunk(2, dim=-1) for (shift, scale, gate)."""
        from comfy.distributed.patcher import EXCLUDED_LAYER_NAMES
        assert "img_mod.1" in EXCLUDED_LAYER_NAMES
        assert "txt_mod.1" in EXCLUDED_LAYER_NAMES


class TestLlamaTP:
    """Verify the Qwen25 7B text encoder (Llama2) is matched by the TP allowlist."""

    def test_llama_layers_identified(self):
        from comfy.distributed.patcher import get_tp_targets
        Llama2 = type("Llama2", (), {})
        assert get_tp_targets(Llama2()) == ["layers"]


class TestShardingMode:
    """Verify the colwise/rowwise classification for QwenImage and Llama2 layer
    names. Builds a synthetic model whose layer names mirror the real
    architectures, then runs parallelize_model and inspects the
    ParallelLinear.mode that was assigned to each replacement."""

    def _collect_modes(self, model):
        from comfy.distributed.parallel_linear import ParallelLinear
        return {name: module.mode for name, module in model.named_modules()
                if isinstance(module, ParallelLinear)}

    def test_qwen_image_sharding_modes(self, monkeypatch):
        """QwenImageTransformerBlock has the following Linear layers under
        `transformer_blocks.*`:

          attn.to_q / attn.to_k / attn.to_v       → colwise
          attn.add_q_proj / attn.add_k_proj / attn.add_v_proj → colwise
          attn.to_out.0                            → rowwise
          attn.to_add_out                          → rowwise
          img_mlp.net.0.proj (GELU inner Linear)   → colwise (up-projection)
          img_mlp.net.2 (MLP down-projection)     → rowwise

        Modulation layers (img_mod.1, txt_mod.1) must be excluded.
        """
        import torch.nn as nn
        from comfy.distributed import patcher
        from comfy.distributed import parallel_linear as pl_module

        def L(in_f, out_f):
            return nn.Linear(in_f, out_f, bias=True)

        class FakeMesh:
            world_size = 2
            rank = 0
            current_device = "cpu"
        # Patch both modules — patcher and ParallelLinear both call get_mesh()
        monkeypatch.setattr(patcher, "get_mesh", lambda: FakeMesh())
        monkeypatch.setattr(pl_module, "get_mesh", lambda: FakeMesh())

        class QwenBlock(nn.Module):
            def __init__(self):
                super().__init__()
                # Modulation: nn.Sequential(SiLU, Linear) — the Linear is at .1
                self.img_mod = nn.Sequential(nn.SiLU(), L(64, 384))
                self.txt_mod = nn.Sequential(nn.SiLU(), L(64, 384))
                # Attention projections
                self.attn = nn.Module()
                self.attn.to_q = L(64, 64)
                self.attn.to_k = L(64, 64)
                self.attn.to_v = L(64, 64)
                self.attn.add_q_proj = L(64, 64)
                self.attn.add_k_proj = L(64, 64)
                self.attn.add_v_proj = L(64, 64)
                self.attn.to_out = nn.ModuleList([L(64, 64), nn.Identity()])
                self.attn.to_add_out = L(64, 64)
                # MLP: ModuleList of [GELU-with-proj, Dropout, Linear]
                self.img_mlp = nn.Module()
                self.img_mlp.net = nn.ModuleList()
                gelu = nn.Sequential()
                gelu.proj = L(64, 256)  # up-projection
                self.img_mlp.net.append(gelu)
                self.img_mlp.net.append(nn.Dropout(0.0))
                self.img_mlp.net.append(L(256, 64))  # down-projection

        class QwenModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.transformer_blocks = nn.ModuleList([QwenBlock()])

        # Tag the model class so the MRO match works
        class QwenImageTransformer2DModel(QwenModel):
            pass

        model = QwenImageTransformer2DModel()
        patched = patcher.parallelize_model(model)
        assert patched is True

        modes = self._collect_modes(model)
        expected = {
            "transformer_blocks.0.attn.to_q": "colwise",
            "transformer_blocks.0.attn.to_k": "colwise",
            "transformer_blocks.0.attn.to_v": "colwise",
            "transformer_blocks.0.attn.add_q_proj": "colwise",
            "transformer_blocks.0.attn.add_k_proj": "colwise",
            "transformer_blocks.0.attn.add_v_proj": "colwise",
            "transformer_blocks.0.attn.to_out.0": "rowwise",
            "transformer_blocks.0.attn.to_add_out": "rowwise",
            "transformer_blocks.0.img_mlp.net.0.proj": "colwise",
            "transformer_blocks.0.img_mlp.net.2": "rowwise",
        }
        for layer_name, expected_mode in expected.items():
            assert layer_name in modes, f"Layer {layer_name} was not sharded"
            assert modes[layer_name] == expected_mode, (
                f"{layer_name}: expected {expected_mode}, got {modes[layer_name]}"
            )

        # Modulation layers must be excluded
        for excluded in ("transformer_blocks.0.img_mod.1",
                         "transformer_blocks.0.txt_mod.1"):
            assert excluded not in modes, f"{excluded} should be excluded"

    def test_llama2_sharding_modes(self, monkeypatch):
        """Llama2 (Qwen25 7B text encoder) layer names under `layers.*`:

          self_attn.q_proj / k_proj / v_proj  → colwise
          self_attn.o_proj                     → rowwise
          mlp.gate_proj / mlp.up_proj          → colwise
          mlp.down_proj                        → rowwise
        """
        import torch.nn as nn
        from comfy.distributed import patcher
        from comfy.distributed import parallel_linear as pl_module

        def L(in_f, out_f):
            return nn.Linear(in_f, out_f, bias=True)

        class FakeMesh:
            world_size = 2
            rank = 0
            current_device = "cpu"
        monkeypatch.setattr(patcher, "get_mesh", lambda: FakeMesh())
        monkeypatch.setattr(pl_module, "get_mesh", lambda: FakeMesh())

        class LlamaBlock(nn.Module):
            def __init__(self):
                super().__init__()
                self.self_attn = nn.Module()
                self.self_attn.q_proj = L(64, 64)
                self.self_attn.k_proj = L(64, 64)
                self.self_attn.v_proj = L(64, 64)
                self.self_attn.o_proj = L(64, 64)
                self.mlp = nn.Module()
                self.mlp.gate_proj = L(64, 256)
                self.mlp.up_proj = L(64, 256)
                self.mlp.down_proj = L(256, 64)

        class Llama2Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.layers = nn.ModuleList([LlamaBlock()])

        class Llama2(Llama2Model):
            pass

        model = Llama2()
        patcher.parallelize_model(model)
        modes = self._collect_modes(model)
        expected = {
            "layers.0.self_attn.q_proj": "colwise",
            "layers.0.self_attn.k_proj": "colwise",
            "layers.0.self_attn.v_proj": "colwise",
            "layers.0.self_attn.o_proj": "rowwise",
            "layers.0.mlp.gate_proj": "colwise",
            "layers.0.mlp.up_proj": "colwise",
            "layers.0.mlp.down_proj": "rowwise",
        }
        for layer_name, expected_mode in expected.items():
            assert layer_name in modes, f"Layer {layer_name} was not sharded"
            assert modes[layer_name] == expected_mode, (
                f"{layer_name}: expected {expected_mode}, got {modes[layer_name]}"
            )


class TestQwenImageHeadSplit:
    """For QwenImage (and similar head-split models), the colwise shard of
    to_q/to_k/to_v distributes a contiguous slice of heads across ranks,
    so each rank holds `(heads/world_size) * dim_head` channels with full
    per-head dim_head. The Attention module's `self.heads` is divided by
    world_size; the per-head norm weights and rotary embeddings are NOT
    sharded (they replicate on every rank).
    """

    def _build_synthetic_model(self, head_dim, n_heads, parent_class_name):
        import torch.nn as nn

        class _Attn(nn.Module):
            def __init__(self, dim):
                super().__init__()
                self.heads = n_heads
                self.dim_head = head_dim
                self.to_q = nn.Linear(dim, dim, bias=False)
                self.norm_q = nn.RMSNorm(head_dim, eps=1e-6)
                self.norm_k = nn.RMSNorm(head_dim, eps=1e-6)
                self.norm_added_q = nn.RMSNorm(head_dim, eps=1e-6)
                self.norm_added_k = nn.RMSNorm(head_dim, eps=1e-6)

        class _Block(nn.Module):
            def __init__(self):
                super().__init__()
                dim = n_heads * head_dim
                self.attn = _Attn(dim)

        class _TransformerBlocks(nn.Module):
            def __init__(self):
                super().__init__()
                self.transformer_blocks = nn.ModuleList([_Block()])

        if parent_class_name == "QwenImageTransformer2DModel":
            class Model(_TransformerBlocks):
                pass
            Model.__name__ = "QwenImageTransformer2DModel"
            return Model()
        raise ValueError(f"Unknown parent class name: {parent_class_name}")

    def test_heads_overridden_to_world_size_divisor(self, monkeypatch):
        from comfy.distributed import patcher
        from comfy.distributed import parallel_linear as pl_module

        class FakeMesh:
            world_size = 2
            rank = 0
            current_device = "cpu"
        monkeypatch.setattr(patcher, "get_mesh", lambda: FakeMesh())
        monkeypatch.setattr(pl_module, "get_mesh", lambda: FakeMesh())

        model = self._build_synthetic_model(
            head_dim=128, n_heads=24, parent_class_name="QwenImageTransformer2DModel"
        )
        attn_before = model.transformer_blocks[0].attn
        assert attn_before.heads == 24
        assert attn_before.dim_head == 128

        patcher.parallelize_model(model)

        attn_after = model.transformer_blocks[0].attn
        assert attn_after.heads == 12, (
            f"expected heads=12 after TP (world_size=2), got {attn_after.heads}"
        )
        assert attn_after.dim_head == 128, (
            f"expected dim_head=128 (unchanged), got {attn_after.dim_head}"
        )

    def test_per_head_norms_not_flagged_or_sliced(self, monkeypatch):
        from comfy.distributed import patcher
        from comfy.distributed import parallel_linear as pl_module

        class FakeMesh:
            world_size = 2
            rank = 0
            current_device = "cpu"
        monkeypatch.setattr(patcher, "get_mesh", lambda: FakeMesh())
        monkeypatch.setattr(pl_module, "get_mesh", lambda: FakeMesh())

        model = self._build_synthetic_model(
            head_dim=128, n_heads=24, parent_class_name="QwenImageTransformer2DModel"
        )
        patcher.parallelize_model(model)

        flagged = {n for n, m in model.named_modules() if getattr(m, "_tp_norm_shard", None) is not None}
        assert flagged == set(), (
            f"Expected no _tp_norm_shard flags under head-split, got: {flagged}"
        )

        attn = model.transformer_blocks[0].attn
        for n in ("norm_q", "norm_k", "norm_added_q", "norm_added_k"):
            norm = getattr(attn, n)
            assert tuple(norm.weight.shape) == (128,), (
                f"{n}: weight shape should be (128,), got {tuple(norm.weight.shape)}"
            )
            assert tuple(norm.normalized_shape) == (128,), (
                f"{n}: normalized_shape should stay (128,), got {tuple(norm.normalized_shape)}"
            )

    def test_non_qwenimage_attn_not_heads_overridden(self, monkeypatch):
        from comfy.distributed import patcher
        from comfy.distributed import parallel_linear as pl_module
        import torch.nn as nn

        class FakeMesh:
            world_size = 2
            rank = 0
            current_device = "cpu"
        monkeypatch.setattr(patcher, "get_mesh", lambda: FakeMesh())
        monkeypatch.setattr(pl_module, "get_mesh", lambda: FakeMesh())

        class _Attn(nn.Module):
            def __init__(self):
                super().__init__()
                self.heads = 20
                self.dim_head = 64
                self.to_q = nn.Linear(1280, 1280, bias=False)

        class _Block(nn.Module):
            def __init__(self):
                super().__init__()
                self.attn = _Attn()

        class _Outer(nn.Module):
            def __init__(self):
                super().__init__()
                self.double_stream_blocks = nn.ModuleList([_Block()])

        class Model(_Outer):
            pass
        Model.__name__ = "HiDreamImageTransformer2DModel"
        model = Model()

        patcher.parallelize_model(model)

        attn = model.double_stream_blocks[0].attn
        assert attn.heads == 20, (
            f"HiDream heads should NOT be overridden, got {attn.heads}"
        )
