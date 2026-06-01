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
