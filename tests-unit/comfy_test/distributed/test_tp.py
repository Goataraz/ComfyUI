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
        assert get_tp_targets(SD3()) == ["joint_blocks"]

    def test_cosmos_general_dit(self):
        from comfy.distributed.patcher import get_tp_targets, TP_UNSUPPORTED
        CosmosT2V = self._make_model_class("GeneralDIT")
        assert "GeneralDIT" not in TP_UNSUPPORTED
        assert get_tp_targets(CosmosT2V()) == ["blocks"]

    def test_cosmos_mini_train_dit(self):
        from comfy.distributed.patcher import get_tp_targets
        CosmosP2 = self._make_model_class("MiniTrainDIT")
        assert get_tp_targets(CosmosP2()) == ["blocks"]

    def test_anima_inherits_minitraindit(self):
        from comfy.distributed.patcher import get_tp_targets
        Mini = self._make_model_class("MiniTrainDIT")
        Anima = self._make_model_class("Anima", (Mini,))
        assert get_tp_targets(Anima()) == ["blocks"]

    def test_flux2_inherits_flux(self):
        from comfy.distributed.patcher import get_tp_targets
        Flux = self._make_model_class("Flux")
        Flux2 = self._make_model_class("Flux2", (Flux,))
        assert get_tp_targets(Flux2()) == ["double_blocks", "single_blocks"]

    def test_hidream_image_matches(self):
        from comfy.distributed.patcher import get_tp_targets
        HiDream = self._make_model_class("HiDreamImageTransformer2DModel")
        assert get_tp_targets(HiDream()) == ["double_stream_blocks", "single_stream_blocks"]

    def test_hidream_o1_matches(self):
        from comfy.distributed.patcher import get_tp_targets, TP_UNSUPPORTED
        HiDreamO1 = self._make_model_class("HiDreamO1Transformer")
        # Vision + LLM both divide on 2-GPU (16 heads / 32/8 GQA). Embeddings stay.
        assert "HiDreamO1Transformer" not in TP_UNSUPPORTED
        assert get_tp_targets(HiDreamO1()) == [
            "language_model.layers",
            "visual.blocks",
        ]

    def test_ltxv_matches(self):
        from comfy.distributed.patcher import get_tp_targets
        LTXV = self._make_model_class("LTXVModel")
        assert get_tp_targets(LTXV()) == ["transformer_blocks"]

    def test_ltxav_inherits_ltxv(self):
        from comfy.distributed.patcher import get_tp_targets
        LTXV = self._make_model_class("LTXVModel")
        LTXAV = self._make_model_class("LTXAVModel", (LTXV,))
        assert get_tp_targets(LTXAV()) == ["transformer_blocks"]

    def test_hunyuan_video_matches(self):
        from comfy.distributed.patcher import get_tp_targets
        HY = self._make_model_class("HunyuanVideo")
        HY15 = self._make_model_class("HunyuanVideo15", (HY,))
        assert get_tp_targets(HY()) == ["double_blocks", "single_blocks"]
        assert get_tp_targets(HY15()) == ["double_blocks", "single_blocks"]

    def test_chroma_matches(self):
        from comfy.distributed.patcher import get_tp_targets
        Chroma = self._make_model_class("Chroma")
        assert get_tp_targets(Chroma()) == ["double_blocks", "single_blocks"]

    def test_chroma_radiance_inherits(self):
        from comfy.distributed.patcher import get_tp_targets
        Chroma = self._make_model_class("Chroma")
        Radiance = self._make_model_class("ChromaRadiance", (Chroma,))
        assert get_tp_targets(Radiance()) == ["double_blocks", "single_blocks"]

    def test_hunyuan3dv2_matches(self):
        from comfy.distributed.patcher import get_tp_targets
        HY3D = self._make_model_class("Hunyuan3Dv2")
        assert get_tp_targets(HY3D()) == ["double_blocks", "single_blocks"]

    def test_kandinsky5_matches(self):
        from comfy.distributed.patcher import get_tp_targets
        K5 = self._make_model_class("Kandinsky5")
        assert get_tp_targets(K5()) == [
            "text_transformer_blocks",
            "visual_transformer_blocks",
        ]

    def test_ernie_image_matches(self):
        from comfy.distributed.patcher import get_tp_targets
        Ernie = self._make_model_class("ErnieImageModel")
        assert get_tp_targets(Ernie()) == ["layers"]

    def test_triposplat_matches(self):
        from comfy.distributed.patcher import get_tp_targets
        Tripo = self._make_model_class("LatentSeqMMFlowModel")
        assert get_tp_targets(Tripo()) == [
            "noise_refiner", "context_refiner", "blocks",
            "noise_repo_layers", "context_repo_layers", "repo_layers",
        ]

    def test_ace_step_matches(self):
        from comfy.distributed.patcher import get_tp_targets
        ACE = self._make_model_class("ACEStepTransformer2DModel")
        ACE15 = self._make_model_class("AceStepConditionGenerationModel")
        assert get_tp_targets(ACE()) == ["transformer_blocks"]
        assert get_tp_targets(ACE15()) == ["decoder.layers"]

    def test_nextdit_matches(self):
        from comfy.distributed.patcher import get_tp_targets
        Next = self._make_model_class("NextDiT")
        assert get_tp_targets(Next()) == ["layers", "noise_refiner", "context_refiner", "siglip_refiner"]

    def test_nextdit_pixel_space_inherits(self):
        from comfy.distributed.patcher import get_tp_targets
        Next = self._make_model_class("NextDiT")
        Pixel = self._make_model_class("NextDiTPixelSpace", (Next,))
        assert get_tp_targets(Pixel()) == [
            "layers", "noise_refiner", "context_refiner", "siglip_refiner",
        ]

    def test_ideogram4_matches(self):
        from comfy.distributed.patcher import get_tp_targets
        I4 = self._make_model_class("Ideogram4Transformer")
        I42D = self._make_model_class("Ideogram4Transformer2DModel", (I4,))
        assert get_tp_targets(I4()) == ["layers"]
        assert get_tp_targets(I42D()) == ["layers"]

    def test_joyimage_matches(self):
        from comfy.distributed.patcher import get_tp_targets
        Joy = self._make_model_class("JoyImageTransformer3DModel")
        assert get_tp_targets(Joy()) == ["double_blocks"]

    def test_lens_matches(self):
        from comfy.distributed.patcher import get_tp_targets
        Lens = self._make_model_class("LensTransformer2DModel")
        assert get_tp_targets(Lens()) == ["transformer_blocks"]

    def test_pixeldit_matches(self):
        from comfy.distributed.patcher import get_tp_targets
        Pix = self._make_model_class("PixDiT_T2I")
        Pid = self._make_model_class("PidNet", (Pix,))
        assert get_tp_targets(Pix()) == ["patch_blocks", "pixel_blocks"]
        assert get_tp_targets(Pid()) == ["patch_blocks", "pixel_blocks"]

    def test_mochi_matches(self):
        from comfy.distributed.patcher import get_tp_targets
        Mochi = self._make_model_class("AsymmDiTJoint")
        assert get_tp_targets(Mochi()) == ["blocks"]

    def test_hunyuandit_matches(self):
        from comfy.distributed.patcher import get_tp_targets
        HY = self._make_model_class("HunYuanDiT")
        Plain = self._make_model_class("HunYuanDiTPlain")
        assert get_tp_targets(HY()) == ["blocks"]
        assert get_tp_targets(Plain()) == ["blocks"]

    def test_cogvideox_matches(self):
        from comfy.distributed.patcher import get_tp_targets
        Cog = self._make_model_class("CogVideoXTransformer3DModel")
        assert get_tp_targets(Cog()) == ["blocks"]

    def test_nadit_matches(self):
        from comfy.distributed.patcher import get_tp_targets
        Na = self._make_model_class("NaDiT")
        assert get_tp_targets(Na()) == ["blocks"]

    def test_auraflow_mmdit_does_not_steal_sd3_wrapper(self):
        from comfy.distributed.patcher import get_tp_targets
        Aura = self._make_model_class("MMDiT")
        SD3 = self._make_model_class("OpenAISignatureMMDITWrapper")
        assert get_tp_targets(Aura()) == ["double_layers", "single_layers"]
        # SD3 inner model is the wrapper; MRO hits that key first.
        assert get_tp_targets(SD3()) == ["joint_blocks"]

    def test_pixart_matches(self):
        from comfy.distributed.patcher import get_tp_targets
        PixArt = self._make_model_class("PixArtMS")
        Sigma = self._make_model_class("PixArtMSSigma", (PixArt,))
        assert get_tp_targets(PixArt()) == ["blocks"]
        assert get_tp_targets(Sigma()) == ["blocks"]

    def test_audio_dit_matches(self):
        from comfy.distributed.patcher import get_tp_targets
        Audio = self._make_model_class("AudioDiffusionTransformer")
        assert get_tp_targets(Audio()) == ["transformer.layers"]

    def test_boogu_matches_omnigen2_denied(self):
        from comfy.distributed.patcher import get_tp_targets
        Boogu = self._make_model_class("BooguTransformer2DModel")
        Omni = self._make_model_class("OmniGen2Transformer2DModel")
        assert get_tp_targets(Boogu()) == [
            "noise_refiner", "ref_image_refiner", "context_refiner",
            "double_stream_layers", "single_stream_layers",
        ]
        # Production OmniGen2 is 21/7 — neither divides by 2-GPU world_size.
        assert get_tp_targets(Omni()) == []

    def test_krea2_matches(self):
        from comfy.distributed.patcher import get_tp_targets
        Krea = self._make_model_class("SingleStreamDiT")
        assert get_tp_targets(Krea()) == ["blocks"]

    def test_mageflow_matches(self):
        from comfy.distributed.patcher import get_tp_targets
        Mage = self._make_model_class("MageFlowTransformer2DModel")
        assert get_tp_targets(Mage()) == ["transformer_blocks"]

    def test_minimax_h3_matches(self):
        from comfy.distributed.patcher import get_tp_targets, TP_UNSUPPORTED
        MiniMax = self._make_model_class("MiniMaxH3Model")
        assert "MiniMaxH3Model" not in TP_UNSUPPORTED
        assert get_tp_targets(MiniMax()) == ["blocks"]

    def test_mro_subclass_inherits(self):
        from comfy.distributed.patcher import get_tp_targets
        """Subclass of a matched class should inherit the TP targets."""
        # Use WanModel — MRO inheritance also covered by VaceWanModel / Wan tests.
        Wan = self._make_model_class("WanModel")
        Vace = self._make_model_class("VaceWanModel", (Wan,))
        assert get_tp_targets(Vace()) == ["blocks"]

    def test_unknown_model_returns_empty(self):
        from comfy.distributed.patcher import get_tp_targets
        Unknown = self._make_model_class("SomeRandomModel")
        assert get_tp_targets(Unknown()) == []

    def test_mro_priority_first_match(self):
        from comfy.distributed.patcher import get_tp_targets
        """When a class inherits from multiple TP targets, the first MRO match wins."""
        Wan = self._make_model_class("WanModel")
        Flux = self._make_model_class("Flux")
        # Create a class that inherits from both — first in MRO wins
        Hybrid = self._make_model_class("HybridModel", (Wan, Flux))
        result = get_tp_targets(Hybrid())
        # WanModel appears first in MRO after HybridModel
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


class TestTPRuntimeInfo:
    """JSON payload for /system_stats so the e2e gate can refuse non-TP servers."""

    def test_inactive_defaults(self):
        from comfy.distributed.utils import tp_runtime_info
        assert tp_runtime_info() == {"active": False, "world_size": 1, "rank": 0}

    def test_active_reads_mesh(self, monkeypatch):
        from comfy.distributed import utils as tp_utils

        class FakeMesh:
            world_size = 2
            rank = 1

        monkeypatch.setattr(tp_utils, "is_tp_active", lambda: True)
        monkeypatch.setattr("comfy.distributed.mesh.get_mesh", lambda: FakeMesh())
        assert tp_utils.tp_runtime_info() == {"active": True, "world_size": 2, "rank": 1}


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
        """QwenImageTransformerBlock under `transformer_blocks.*`:

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
        monkeypatch.setattr(patcher, "get_mesh", lambda: FakeMesh())
        monkeypatch.setattr(pl_module, "get_mesh", lambda: FakeMesh())

        class QwenBlock(nn.Module):
            def __init__(self):
                super().__init__()
                self.img_mod = nn.Sequential(nn.SiLU(), L(64, 384))
                self.txt_mod = nn.Sequential(nn.SiLU(), L(64, 384))
                self.attn = nn.Module()
                self.attn.to_q = L(64, 64)
                self.attn.to_k = L(64, 64)
                self.attn.to_v = L(64, 64)
                self.attn.add_q_proj = L(64, 64)
                self.attn.add_k_proj = L(64, 64)
                self.attn.add_v_proj = L(64, 64)
                self.attn.to_out = nn.ModuleList([L(64, 64), nn.Identity()])
                self.attn.to_add_out = L(64, 64)
                self.img_mlp = nn.Module()
                self.img_mlp.net = nn.ModuleList()
                gelu = nn.Sequential()
                gelu.proj = L(64, 256)
                self.img_mlp.net.append(gelu)
                self.img_mlp.net.append(nn.Dropout(0.0))
                self.img_mlp.net.append(L(256, 64))

        class QwenModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.transformer_blocks = nn.ModuleList([QwenBlock()])

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
    """QwenImage head-split: colwise QKV distributes heads; self.heads /= world_size.
    SageAttention is disabled under TP in qwen_image/model.py (black-image fix).
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

    def test_hidream_heads_overridden_and_full_dim_norms_sliced(self, monkeypatch):
        """HiDream Image is head-split; full-dim QK RMSNorms must shrink to the
        local shard (unlike QwenImage's per-head norms which stay replicated)."""
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
                inner = 20 * 64
                self.to_q = nn.Linear(inner, inner, bias=False)
                self.q_rms_norm = nn.RMSNorm(inner, eps=1e-6)
                self.k_rms_norm = nn.RMSNorm(inner, eps=1e-6)

        class _Block(nn.Module):
            def __init__(self):
                super().__init__()
                self.attn1 = _Attn()
                # Modulation must stay un-sharded (chunk on full dim)
                self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(1280, 6 * 1280))

        class _Outer(nn.Module):
            def __init__(self):
                super().__init__()
                self.double_stream_blocks = nn.ModuleList([_Block()])

        class Model(_Outer):
            pass
        Model.__name__ = "HiDreamImageTransformer2DModel"
        model = Model()

        patcher.parallelize_model(model)

        attn = model.double_stream_blocks[0].attn1
        assert attn.heads == 10, f"HiDream heads should be 10 after TP, got {attn.heads}"
        assert tuple(attn.q_rms_norm.weight.shape) == (640,), (
            f"q_rms_norm should be sliced to 640, got {tuple(attn.q_rms_norm.weight.shape)}"
        )
        assert tuple(attn.k_rms_norm.normalized_shape) == (640,)
        # Modulation Linear must remain nn.Linear (not ParallelLinear)
        from comfy.distributed.parallel_linear import ParallelLinear
        mod_linear = model.double_stream_blocks[0].adaLN_modulation[1]
        assert not isinstance(mod_linear, ParallelLinear), "adaLN_modulation.1 must be excluded"


# ---------------------------------------------------------------------------
# ParallelLinear weight_function (LoRA) application tests
# ---------------------------------------------------------------------------

class TestParallelLinearWeightFunction:
    """Verify that weight_function patches (e.g. LowVramPatch/LoRA) are applied
    on forward WITHOUT mutating the underlying weight Parameter. A same-dtype
    ``.to()`` returns the real storage, so patches must operate on a fresh copy
    or LoRA would accumulate into the weight on every forward pass."""

    def _make_layer(self, monkeypatch):
        import torch
        from comfy.distributed import parallel_linear as pl_module

        class FakeMesh:
            world_size = 2
            rank = 0
            current_device = "cpu"
        monkeypatch.setattr(pl_module, "get_mesh", lambda: FakeMesh())

        # colwise: out_features must be divisible by world_size (2)
        layer = pl_module.ParallelLinear(64, 64, bias=True, mode="colwise")
        # weight/bias are allocated empty on CPU — give them deterministic data
        layer.weight.data = torch.ones((layer.local_out_features, layer.local_in_features),
                                       dtype=torch.float32)
        layer.bias.data = torch.zeros(layer.local_out_features, dtype=torch.float32)
        return layer

    def test_default_lists_present(self, monkeypatch):
        layer = self._make_layer(monkeypatch)
        assert layer.weight_function == []
        assert layer.bias_function == []

    def test_weight_not_mutated_across_forwards(self, monkeypatch):
        import torch
        layer = self._make_layer(monkeypatch)

        # A dummy patch that adds 1.0 to the weight it receives (mimics a LoRA
        # applied via LowVramPatch — same-dtype cast returns the live storage).
        layer.weight_function = [lambda w: w + 1.0]

        original = layer.weight.data.clone()
        x = torch.randn(3, 64, dtype=torch.float32)

        out1 = layer(x)
        assert torch.equal(layer.weight.data, original), (
            "weight Parameter mutated after first forward — LoRA leaked into storage"
        )

        out2 = layer(x)
        assert torch.equal(layer.weight.data, original), (
            "weight Parameter mutated after second forward — LoRA accumulated into storage"
        )

        # The patch effect must still be reflected in the output, and be stable
        # across forwards (no accumulation).
        assert torch.allclose(out1, out2), "forward output drifted across passes"

        # Sanity: with weight of ones + patch (+1) => effective weight of 2s.
        expected = torch.matmul(x, (layer.weight.data + 1.0).t()) + layer.bias.data
        assert torch.allclose(out1, expected), "weight_function was not applied on forward"


# ---------------------------------------------------------------------------
class TestSD3JointBlocksTP:
    """SD3 MMDiT: packed attn.qkv [Q|K|V]; MLP fc1/fc2; adaLN stays."""

    def test_targets(self):
        from comfy.distributed.patcher import get_tp_targets, EXCLUDED_LAYER_NAMES, TP_HEAD_SPLIT_MODELS
        SD3 = type("OpenAISignatureMMDITWrapper", (), {})
        assert get_tp_targets(SD3()) == ["joint_blocks"]
        # Global list still names the suffixes; SD3 un-excludes them at runtime.
        assert ".attn.qkv" in EXCLUDED_LAYER_NAMES
        assert ".attn.proj" in EXCLUDED_LAYER_NAMES
        assert "OpenAISignatureMMDITWrapper" in TP_HEAD_SPLIT_MODELS

    def test_packed_qkv_and_mlp_sharding(self, monkeypatch):
        import torch.nn as nn
        from comfy.distributed import patcher
        from comfy.distributed import parallel_linear as pl_module
        from comfy.distributed.parallel_linear import ParallelLinear

        class FakeMesh:
            world_size = 2
            rank = 0
            current_device = "cpu"
        monkeypatch.setattr(patcher, "get_mesh", lambda: FakeMesh())
        monkeypatch.setattr(pl_module, "get_mesh", lambda: FakeMesh())

        def L(i, o):
            return nn.Linear(i, o, bias=True)

        class Attn(nn.Module):
            def __init__(self):
                super().__init__()
                self.num_heads = 24
                self.head_dim = 64
                dim = 24 * 64
                self.qkv = L(dim, dim * 3)
                self.proj = L(dim, dim)

        class Block(nn.Module):
            def __init__(self):
                super().__init__()
                self.attn = Attn()
                self.mlp = nn.Module()
                self.mlp.fc1 = L(1536, 6144)
                self.mlp.fc2 = L(6144, 1536)
                self.adaLN_modulation = nn.Sequential(nn.SiLU(), L(1536, 6 * 1536))

        class JointBlock(nn.Module):
            def __init__(self):
                super().__init__()
                self.context_block = Block()
                self.x_block = Block()

        class OpenAISignatureMMDITWrapper(nn.Module):
            def __init__(self):
                super().__init__()
                self.num_heads = 24
                self.joint_blocks = nn.ModuleList([JointBlock()])

        model = OpenAISignatureMMDITWrapper()
        assert patcher.parallelize_model(model) is True
        qkv = model.joint_blocks[0].x_block.attn.qkv
        assert isinstance(qkv, ParallelLinear)
        assert qkv.mode == "colwise"
        assert qkv.pack_count == 3
        assert qkv.local_out_features == 3 * (1536 // 2)
        proj = model.joint_blocks[0].x_block.attn.proj
        assert isinstance(proj, ParallelLinear)
        assert proj.mode == "rowwise"
        modes = {n: m.mode for n, m in model.named_modules() if isinstance(m, ParallelLinear)}
        assert modes["joint_blocks.0.x_block.mlp.fc1"] == "colwise"
        assert modes["joint_blocks.0.x_block.mlp.fc2"] == "rowwise"
        assert modes["joint_blocks.0.context_block.mlp.fc1"] == "colwise"
        assert "joint_blocks.0.x_block.adaLN_modulation.1" not in modes
        assert model.joint_blocks[0].x_block.attn.num_heads == 12
        assert model.num_heads == 24

    def test_skips_scaled_fp8_companions(self, monkeypatch):
        """FP8-scaled checkpoints attach weight_scale; ParallelLinear must
        not replace those Linears or scales are dropped and quality tanks."""
        import torch
        import torch.nn as nn
        from comfy.distributed import patcher
        from comfy.distributed import parallel_linear as pl_module
        from comfy.distributed.parallel_linear import ParallelLinear

        class FakeMesh:
            world_size = 2
            rank = 0
            current_device = "cpu"
        monkeypatch.setattr(patcher, "get_mesh", lambda: FakeMesh())
        monkeypatch.setattr(pl_module, "get_mesh", lambda: FakeMesh())

        class Block(nn.Module):
            def __init__(self):
                super().__init__()
                self.mlp = nn.Module()
                self.mlp.fc1 = nn.Linear(1536, 6144)
                self.mlp.fc2 = nn.Linear(6144, 1536)

        class JointBlock(nn.Module):
            def __init__(self):
                super().__init__()
                self.x_block = Block()

        class OpenAISignatureMMDITWrapper(nn.Module):
            def __init__(self):
                super().__init__()
                self.joint_blocks = nn.ModuleList([JointBlock()])

        model = OpenAISignatureMMDITWrapper()
        # Only fc1 has a scale companion → must stay Linear; fc2 can shard.
        sd = {
            "joint_blocks.0.x_block.mlp.fc1.weight": torch.empty(6144, 1536),
            "joint_blocks.0.x_block.mlp.fc1.weight_scale": torch.tensor(1.0),
            "joint_blocks.0.x_block.mlp.fc2.weight": torch.empty(1536, 6144),
        }
        assert patcher.parallelize_model(model, sd=sd, prefix="") is True
        assert not isinstance(model.joint_blocks[0].x_block.mlp.fc1, ParallelLinear)
        assert isinstance(model.joint_blocks[0].x_block.mlp.fc1, nn.Linear)
        assert isinstance(model.joint_blocks[0].x_block.mlp.fc2, ParallelLinear)
        assert model.joint_blocks[0].x_block.mlp.fc2.mode == "rowwise"

    def test_all_scaled_returns_false(self, monkeypatch):
        """When every candidate linear is scaled, TP must no-op (fallback load)."""
        import torch
        import torch.nn as nn
        from comfy.distributed import patcher
        from comfy.distributed import parallel_linear as pl_module

        class FakeMesh:
            world_size = 2
            rank = 0
            current_device = "cpu"
        monkeypatch.setattr(patcher, "get_mesh", lambda: FakeMesh())
        monkeypatch.setattr(pl_module, "get_mesh", lambda: FakeMesh())

        class Block(nn.Module):
            def __init__(self):
                super().__init__()
                self.mlp = nn.Module()
                self.mlp.fc1 = nn.Linear(1536, 6144)
                self.mlp.fc2 = nn.Linear(6144, 1536)

        class JointBlock(nn.Module):
            def __init__(self):
                super().__init__()
                self.x_block = Block()

        class OpenAISignatureMMDITWrapper(nn.Module):
            def __init__(self):
                super().__init__()
                self.joint_blocks = nn.ModuleList([JointBlock()])

        model = OpenAISignatureMMDITWrapper()
        sd = {
            "joint_blocks.0.x_block.mlp.fc1.weight_scale": torch.tensor(1.0),
            "joint_blocks.0.x_block.mlp.fc2.weight_scale": torch.tensor(1.0),
        }
        assert patcher.parallelize_model(model, sd=sd) is False


# Cosmos GeneralDIT / Wan / expanded head-split coverage
# ---------------------------------------------------------------------------

class TestCosmosGeneralDITTP:
    """GeneralDIT MLP-only TP: FA/CA projections stay replicated."""

    def test_targets_and_exclusions(self):
        from comfy.distributed.patcher import (
            get_tp_targets, EXCLUDED_LAYER_NAMES, GENERALDIT_EXCLUDED_LAYER_NAMES,
            TP_HEAD_SPLIT_MODELS, TP_UNSUPPORTED, TP_TARGETS,
        )
        GeneralDIT = type("GeneralDIT", (), {})
        assert "GeneralDIT" not in TP_UNSUPPORTED
        assert "GeneralDIT" in TP_TARGETS
        assert get_tp_targets(GeneralDIT()) == ["blocks"]
        assert "adaLN_modulation.1" in EXCLUDED_LAYER_NAMES
        assert "adaLN_modulation.2" in EXCLUDED_LAYER_NAMES
        # Must be GeneralDIT-scoped — QwenImage also uses attn.to_out.0
        assert "attn.to_out.0" not in EXCLUDED_LAYER_NAMES
        assert "attn.to_out.0" in GENERALDIT_EXCLUDED_LAYER_NAMES
        assert "attn.to_q.0" in GENERALDIT_EXCLUDED_LAYER_NAMES
        assert "GeneralDIT" not in TP_HEAD_SPLIT_MODELS

    def test_mlp_only_sharding(self, monkeypatch):
        """MLP layer1/layer2 shard; FA/CA Sequential projections stay Linear."""
        import torch.nn as nn
        from comfy.distributed import patcher
        from comfy.distributed import parallel_linear as pl_module
        from comfy.distributed.parallel_linear import ParallelLinear

        class FakeMesh:
            world_size = 2
            rank = 0
            current_device = "cpu"
        monkeypatch.setattr(patcher, "get_mesh", lambda: FakeMesh())
        monkeypatch.setattr(pl_module, "get_mesh", lambda: FakeMesh())

        def L(i, o):
            return nn.Linear(i, o, bias=True)

        class Attn(nn.Module):
            def __init__(self, context_dim):
                super().__init__()
                self.heads = 16
                self.dim_head = 64
                inner = 16 * 64
                self.to_q = nn.Sequential(L(inner, inner), nn.RMSNorm(64))
                self.to_k = nn.Sequential(L(context_dim, inner), nn.RMSNorm(64))
                self.to_v = nn.Sequential(L(context_dim, inner), nn.Identity())
                self.to_out = nn.Sequential(L(inner, inner), nn.Dropout(0.0))

        class BuildingBlock(nn.Module):
            def __init__(self, kind):
                super().__init__()
                self.adaLN_modulation = nn.Sequential(nn.SiLU(), L(1024, 3 * 1024))
                if kind == "mlp":
                    self.block = nn.Module()
                    self.block.layer1 = L(1024, 4096)
                    self.block.layer2 = L(4096, 1024)
                else:
                    self.block = nn.Module()
                    self.block.attn = Attn(1024 if kind == "ca" else 1024)

        class TransformerBlock(nn.Module):
            def __init__(self):
                super().__init__()
                self.blocks = nn.ModuleList([
                    BuildingBlock("fa"), BuildingBlock("ca"), BuildingBlock("mlp"),
                ])

        class GeneralDIT(nn.Module):
            def __init__(self):
                super().__init__()
                self.blocks = nn.ModuleDict({"block0": TransformerBlock()})

        model = GeneralDIT()
        assert patcher.parallelize_model(model) is True

        modes = {n: m.mode for n, m in model.named_modules() if isinstance(m, ParallelLinear)}
        assert "blocks.block0.blocks.0.block.attn.to_q.0" not in modes
        assert "blocks.block0.blocks.0.block.attn.to_out.0" not in modes
        assert "blocks.block0.blocks.1.block.attn.to_k.0" not in modes
        assert modes["blocks.block0.blocks.2.block.layer1"] == "colwise"
        assert modes["blocks.block0.blocks.2.block.layer2"] == "rowwise"
        assert "blocks.block0.blocks.2.adaLN_modulation.1" not in modes
        assert not isinstance(model.blocks["block0"].blocks[0].block.attn.to_q[0], ParallelLinear)
        assert model.blocks["block0"].blocks[1].block.attn.to_k[0].in_features == 1024


class TestWanModelTP:
    """WanModel: bare .q/.k/.v/.o naming, ffn.0/ffn.2, full-dim QK norm slice."""

    def test_targets(self):
        from comfy.distributed.patcher import get_tp_targets, TP_HEAD_SPLIT_MODELS
        Wan = type("WanModel", (), {})
        Vace = type("VaceWanModel", (Wan,), {})
        assert get_tp_targets(Wan()) == ["blocks"]
        assert get_tp_targets(Vace()) == ["blocks"]  # MRO inherit
        assert "WanModel" in TP_HEAD_SPLIT_MODELS

    def test_sharding_modes_head_split_and_norm_slice(self, monkeypatch):
        import torch.nn as nn
        from comfy.distributed import patcher
        from comfy.distributed import parallel_linear as pl_module
        from comfy.distributed.parallel_linear import ParallelLinear

        class FakeMesh:
            world_size = 2
            rank = 1  # non-zero rank to verify slice offset
            current_device = "cpu"
        monkeypatch.setattr(patcher, "get_mesh", lambda: FakeMesh())
        monkeypatch.setattr(pl_module, "get_mesh", lambda: FakeMesh())

        dim = 768  # 12 heads * 64
        n_heads = 12
        head_dim = 64

        class SelfAttn(nn.Module):
            def __init__(self):
                super().__init__()
                self.num_heads = n_heads
                self.head_dim = head_dim
                self.q = nn.Linear(dim, dim, bias=False)
                self.k = nn.Linear(dim, dim, bias=False)
                self.v = nn.Linear(dim, dim, bias=False)
                self.o = nn.Linear(dim, dim, bias=False)
                self.norm_q = nn.RMSNorm(dim)
                self.norm_k = nn.RMSNorm(dim)

        class Block(nn.Module):
            def __init__(self):
                super().__init__()
                self.self_attn = SelfAttn()
                self.ffn = nn.ModuleList([
                    nn.Linear(dim, dim * 4, bias=True),  # .0 up
                    nn.GELU(),
                    nn.Linear(dim * 4, dim, bias=True),  # .2 down
                ])

        class WanModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.blocks = nn.ModuleList([Block()])

        model = WanModel()
        # Stash original norm weight to verify rank-1 slice
        orig_norm = model.blocks[0].self_attn.norm_q.weight.data.clone()

        assert patcher.parallelize_model(model) is True

        modes = {n: m.mode for n, m in model.named_modules() if isinstance(m, ParallelLinear)}
        expected = {
            "blocks.0.self_attn.q": "colwise",
            "blocks.0.self_attn.k": "colwise",
            "blocks.0.self_attn.v": "colwise",
            "blocks.0.self_attn.o": "rowwise",
            "blocks.0.ffn.0": "colwise",
            "blocks.0.ffn.2": "rowwise",
        }
        for name, mode in expected.items():
            assert name in modes, f"{name} not sharded"
            assert modes[name] == mode, f"{name}: expected {mode}, got {modes[name]}"

        attn = model.blocks[0].self_attn
        assert attn.num_heads == 6
        local_dim = 6 * 64
        assert tuple(attn.norm_q.weight.shape) == (local_dim,)
        assert tuple(attn.norm_q.normalized_shape) == (local_dim,)
        # Rank 1 should hold the second half of the original norm weight
        import torch
        assert torch.allclose(attn.norm_q.weight.data, orig_norm[local_dim:])


class TestTPParamNamesAndDeny:
    def test_sliced_norms_included_in_tp_param_names(self, monkeypatch):
        import torch.nn as nn
        from comfy.distributed import patcher, utils as tp_utils
        from comfy.distributed import parallel_linear as pl_module

        class FakeMesh:
            world_size = 2
            rank = 0
            current_device = "cpu"
        monkeypatch.setattr(patcher, "get_mesh", lambda: FakeMesh())
        monkeypatch.setattr(pl_module, "get_mesh", lambda: FakeMesh())

        class SelfAttn(nn.Module):
            def __init__(self):
                super().__init__()
                self.num_heads = 8
                self.head_dim = 16
                dim = 128
                self.q = nn.Linear(dim, dim, bias=False)
                self.norm_q = nn.RMSNorm(dim)

        class WanModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.blocks = nn.ModuleList([nn.Module()])
                self.blocks[0].self_attn = SelfAttn()

        model = WanModel()
        patcher.parallelize_model(model)
        names = tp_utils.get_tp_param_names(model)
        assert "blocks.0.self_attn.q.weight" in names
        assert "blocks.0.self_attn.norm_q.weight" in names, (
            "sliced QK norms must be excluded from full state_dict loads"
        )

    def test_causal_wan_inherits_wan_targets(self):
        from comfy.distributed.patcher import get_tp_targets, TP_UNSUPPORTED
        Wan = type("WanModel", (), {})
        Causal = type("CausalWanModel", (Wan,), {})
        assert "CausalWanModel" not in TP_UNSUPPORTED
        assert get_tp_targets(Wan()) == ["blocks"]
        assert get_tp_targets(Causal()) == ["blocks"]


class TestCausalWanHeadBookkeeping:
    """CausalWan KV cache uses outer/block num_heads. Head-split must sync them."""

    def test_outer_and_block_heads_follow_attention_shard(self, monkeypatch):
        import torch.nn as nn
        from comfy.distributed import patcher
        from comfy.distributed import parallel_linear as pl_module
        from comfy.distributed.parallel_linear import ParallelLinear

        class FakeMesh:
            world_size = 2
            rank = 0
            current_device = "cpu"
        monkeypatch.setattr(patcher, "get_mesh", lambda: FakeMesh())
        monkeypatch.setattr(pl_module, "get_mesh", lambda: FakeMesh())

        dim = 128
        n_heads = 8
        head_dim = 16

        class CausalWanSelfAttention(nn.Module):
            def __init__(self):
                super().__init__()
                self.num_heads = n_heads
                self.head_dim = head_dim
                self.q = nn.Linear(dim, dim, bias=False)
                self.k = nn.Linear(dim, dim, bias=False)
                self.v = nn.Linear(dim, dim, bias=False)
                self.o = nn.Linear(dim, dim, bias=False)
                self.norm_q = nn.RMSNorm(dim)
                self.norm_k = nn.RMSNorm(dim)

        class CausalWanAttentionBlock(nn.Module):
            def __init__(self):
                super().__init__()
                self.num_heads = n_heads
                self.self_attn = CausalWanSelfAttention()

        class WanModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.num_heads = n_heads
                self.head_dim = head_dim
                self.blocks = nn.ModuleList([CausalWanAttentionBlock()])

        class CausalWanModel(WanModel):
            pass

        model = CausalWanModel()
        assert patcher.parallelize_model(model) is True
        assert isinstance(model.blocks[0].self_attn.q, ParallelLinear)
        local = n_heads // 2
        assert model.blocks[0].self_attn.num_heads == local
        assert model.blocks[0].num_heads == local, (
            "block-level num_heads is used for cross-attn cache + optimized_attention"
        )
        assert model.num_heads == local, (
            "CausalWanModel.init_kv_caches allocates [B, S, num_heads, head_dim]"
        )


class TestLTXVMLPOnly:
    """LTXV MLP-only TP: RoPE + full-dim QK norms stay coupled to full heads."""

    def test_targets_and_exclusions(self):
        from comfy.distributed.patcher import (
            get_tp_targets, TP_HEAD_SPLIT_MODELS, TP_TARGETS, LTXV_EXCLUDED_LAYER_NAMES,
        )
        LTXV = type("LTXVModel", (), {})
        assert "LTXVModel" in TP_TARGETS
        assert get_tp_targets(LTXV()) == ["transformer_blocks"]
        assert "LTXVModel" not in TP_HEAD_SPLIT_MODELS
        assert "to_q" in LTXV_EXCLUDED_LAYER_NAMES
        assert "to_out.0" in LTXV_EXCLUDED_LAYER_NAMES
        # Must stay LTX-scoped — QwenImage shards attn.to_q.
        from comfy.distributed.patcher import EXCLUDED_LAYER_NAMES
        assert "to_q" not in EXCLUDED_LAYER_NAMES

    def test_mlp_only_sharding(self, monkeypatch):
        import torch.nn as nn
        from comfy.distributed import patcher
        from comfy.distributed import parallel_linear as pl_module
        from comfy.distributed.parallel_linear import ParallelLinear

        class FakeMesh:
            world_size = 2
            rank = 0
            current_device = "cpu"
        monkeypatch.setattr(patcher, "get_mesh", lambda: FakeMesh())
        monkeypatch.setattr(pl_module, "get_mesh", lambda: FakeMesh())

        dim, heads, head_dim = 128, 8, 16
        inner_ff = dim * 4

        class CrossAttention(nn.Module):
            def __init__(self):
                super().__init__()
                self.heads = heads
                self.dim_head = head_dim
                self.to_q = nn.Linear(dim, dim, bias=True)
                self.to_k = nn.Linear(dim, dim, bias=True)
                self.to_v = nn.Linear(dim, dim, bias=True)
                self.to_out = nn.Sequential(nn.Linear(dim, dim), nn.Dropout(0.0))
                self.q_norm = nn.RMSNorm(dim)
                self.k_norm = nn.RMSNorm(dim)

        class GELUApprox(nn.Module):
            def __init__(self):
                super().__init__()
                self.proj = nn.Linear(dim, inner_ff)

        class FeedForward(nn.Module):
            def __init__(self):
                super().__init__()
                self.net = nn.Sequential(GELUApprox(), nn.Dropout(0.0), nn.Linear(inner_ff, dim))

        class BasicTransformerBlock(nn.Module):
            def __init__(self):
                super().__init__()
                self.attn1 = CrossAttention()
                self.attn2 = CrossAttention()
                self.ff = FeedForward()

        class LTXVModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.num_attention_heads = heads
                self.inner_dim = dim
                self.transformer_blocks = nn.ModuleList([BasicTransformerBlock()])

        model = LTXVModel()
        assert patcher.parallelize_model(model) is True
        assert isinstance(model.transformer_blocks[0].attn1.to_q, nn.Linear)
        assert isinstance(model.transformer_blocks[0].attn1.to_out[0], nn.Linear)
        assert isinstance(model.transformer_blocks[0].ff.net[0].proj, ParallelLinear)
        assert model.transformer_blocks[0].ff.net[0].proj.mode == "colwise"
        assert isinstance(model.transformer_blocks[0].ff.net[2], ParallelLinear)
        assert model.transformer_blocks[0].ff.net[2].mode == "rowwise"
        assert model.transformer_blocks[0].attn1.heads == heads
        assert model.num_attention_heads == heads
        assert tuple(model.transformer_blocks[0].attn1.q_norm.weight.shape) == (dim,)

    def test_ltxav_audio_ff_shards(self, monkeypatch):
        """LTXAVModel inherits LTXV exclusions; audio_attn stays, audio_ff shards."""
        import torch.nn as nn
        from comfy.distributed import patcher
        from comfy.distributed import parallel_linear as pl_module
        from comfy.distributed.parallel_linear import ParallelLinear

        class FakeMesh:
            world_size = 2
            rank = 0
            current_device = "cpu"
        monkeypatch.setattr(patcher, "get_mesh", lambda: FakeMesh())
        monkeypatch.setattr(pl_module, "get_mesh", lambda: FakeMesh())

        dim = 128

        class CrossAttention(nn.Module):
            def __init__(self):
                super().__init__()
                self.heads = 8
                self.to_q = nn.Linear(dim, dim)
                self.to_k = nn.Linear(dim, dim)
                self.to_v = nn.Linear(dim, dim)
                self.to_out = nn.Sequential(nn.Linear(dim, dim), nn.Dropout(0.0))

        class GELUApprox(nn.Module):
            def __init__(self):
                super().__init__()
                self.proj = nn.Linear(dim, dim * 4)

        class FeedForward(nn.Module):
            def __init__(self):
                super().__init__()
                self.net = nn.Sequential(GELUApprox(), nn.Dropout(0.0), nn.Linear(dim * 4, dim))

        class BasicAVTransformerBlock(nn.Module):
            def __init__(self):
                super().__init__()
                self.attn1 = CrossAttention()
                self.audio_attn1 = CrossAttention()
                self.audio_to_video_attn = CrossAttention()
                self.ff = FeedForward()
                self.audio_ff = FeedForward()

        class LTXVModel(nn.Module):
            pass

        class LTXAVModel(LTXVModel):
            def __init__(self):
                super().__init__()
                self.transformer_blocks = nn.ModuleList([BasicAVTransformerBlock()])

        model = LTXAVModel()
        assert patcher.parallelize_model(model) is True
        assert isinstance(model.transformer_blocks[0].audio_attn1.to_q, nn.Linear)
        assert isinstance(model.transformer_blocks[0].audio_to_video_attn.to_q, nn.Linear)
        assert isinstance(model.transformer_blocks[0].audio_ff.net[0].proj, ParallelLinear)
        assert model.transformer_blocks[0].audio_ff.net[0].proj.mode == "colwise"


class TestFluxDoubleStreamTP:
    """Flux: packed double-stream [Q|K|V]; fused single-stream linear1 stays excluded."""

    def test_packed_qkv_and_head_split(self, monkeypatch):
        import torch.nn as nn
        from comfy.distributed import patcher
        from comfy.distributed import parallel_linear as pl_module
        from comfy.distributed.parallel_linear import ParallelLinear
        from comfy.distributed.patcher import TP_HEAD_SPLIT_MODELS

        class FakeMesh:
            world_size = 2
            rank = 0
            current_device = "cpu"
        monkeypatch.setattr(patcher, "get_mesh", lambda: FakeMesh())
        monkeypatch.setattr(pl_module, "get_mesh", lambda: FakeMesh())

        dim = 128
        assert "Flux" in TP_HEAD_SPLIT_MODELS
        assert "HunyuanVideo" in TP_HEAD_SPLIT_MODELS
        assert "Chroma" in TP_HEAD_SPLIT_MODELS

        class SelfAttention(nn.Module):
            def __init__(self):
                super().__init__()
                self.num_heads = 8
                self.qkv = nn.Linear(dim, dim * 3, bias=False)
                self.proj = nn.Linear(dim, dim)

        class DoubleStreamBlock(nn.Module):
            def __init__(self):
                super().__init__()
                self.num_heads = 8
                self.img_attn = SelfAttention()
                self.txt_attn = SelfAttention()
                self.img_mlp = nn.Sequential(nn.Linear(dim, dim * 4), nn.GELU(), nn.Linear(dim * 4, dim))
                self.txt_mlp = nn.Sequential(nn.Linear(dim, dim * 4), nn.GELU(), nn.Linear(dim * 4, dim))

        class SingleStreamBlock(nn.Module):
            def __init__(self):
                super().__init__()
                self.num_heads = 8
                self.linear1 = nn.Linear(dim, dim * 3 + dim * 4)
                self.linear2 = nn.Linear(dim + dim * 4, dim)

        class Flux(nn.Module):
            def __init__(self):
                super().__init__()
                self.num_heads = 8
                self.double_blocks = nn.ModuleList([DoubleStreamBlock()])
                self.single_blocks = nn.ModuleList([SingleStreamBlock()])

        model = Flux()
        assert patcher.parallelize_model(model) is True
        qkv = model.double_blocks[0].img_attn.qkv
        assert isinstance(qkv, ParallelLinear)
        assert qkv.mode == "colwise"
        assert qkv.pack_count == 3
        assert qkv.local_out_features == 3 * (dim // 2)
        assert isinstance(model.double_blocks[0].img_attn.proj, ParallelLinear)
        assert model.double_blocks[0].img_attn.proj.mode == "rowwise"
        assert isinstance(model.single_blocks[0].linear1, nn.Linear)
        assert model.double_blocks[0].img_attn.num_heads == 4
        assert model.double_blocks[0].num_heads == 4
        assert model.single_blocks[0].num_heads == 8
        assert model.num_heads == 8


class TestHunyuanVideoTP:
    """HunyuanVideo: packed double-stream QKV; single-stream linear1 stays; MLP shards."""

    def test_double_stream_packed_qkv_single_stream_stays(self, monkeypatch):
        import torch.nn as nn
        from comfy.distributed import patcher
        from comfy.distributed import parallel_linear as pl_module
        from comfy.distributed.parallel_linear import ParallelLinear

        class FakeMesh:
            world_size = 2
            rank = 0
            current_device = "cpu"
        monkeypatch.setattr(patcher, "get_mesh", lambda: FakeMesh())
        monkeypatch.setattr(pl_module, "get_mesh", lambda: FakeMesh())

        dim = 128

        class SelfAttention(nn.Module):
            def __init__(self):
                super().__init__()
                self.num_heads = 8
                self.qkv = nn.Linear(dim, dim * 3, bias=False)
                self.proj = nn.Linear(dim, dim)

        class DoubleStreamBlock(nn.Module):
            def __init__(self):
                super().__init__()
                self.num_heads = 8
                self.img_attn = SelfAttention()
                self.txt_attn = SelfAttention()
                self.img_mlp = nn.Sequential(nn.Linear(dim, dim * 4), nn.GELU(), nn.Linear(dim * 4, dim))
                self.txt_mlp = nn.Sequential(nn.Linear(dim, dim * 4), nn.GELU(), nn.Linear(dim * 4, dim))
                self.img_mod = nn.Module()
                self.img_mod.lin = nn.Linear(dim, 6 * dim)

        class SingleStreamBlock(nn.Module):
            def __init__(self):
                super().__init__()
                self.num_heads = 8
                self.linear1 = nn.Linear(dim, dim * 3 + dim * 4)
                self.linear2 = nn.Linear(dim + dim * 4, dim)

        class HunyuanVideo(nn.Module):
            def __init__(self):
                super().__init__()
                self.num_heads = 8
                self.double_blocks = nn.ModuleList([DoubleStreamBlock()])
                self.single_blocks = nn.ModuleList([SingleStreamBlock()])

        model = HunyuanVideo()
        assert patcher.parallelize_model(model) is True
        img_attn = model.double_blocks[0].img_attn
        assert isinstance(img_attn.qkv, ParallelLinear)
        assert img_attn.qkv.mode == "colwise"
        assert img_attn.qkv.pack_count == 3
        assert img_attn.qkv.local_out_features == 3 * (dim // 2)
        assert isinstance(img_attn.proj, ParallelLinear)
        assert img_attn.proj.mode == "rowwise"
        txt_attn = model.double_blocks[0].txt_attn
        assert isinstance(txt_attn.qkv, ParallelLinear)
        assert txt_attn.qkv.pack_count == 3
        assert isinstance(model.single_blocks[0].linear1, nn.Linear)
        assert isinstance(model.single_blocks[0].linear2, nn.Linear)
        assert isinstance(model.double_blocks[0].img_mlp[0], ParallelLinear)
        assert model.double_blocks[0].img_mlp[0].mode == "colwise"
        assert isinstance(model.double_blocks[0].img_mlp[2], ParallelLinear)
        assert model.double_blocks[0].img_mlp[2].mode == "rowwise"
        assert isinstance(model.double_blocks[0].img_mod.lin, nn.Linear)
        assert img_attn.num_heads == 4
        assert model.double_blocks[0].num_heads == 4
        assert model.single_blocks[0].num_heads == 8
        assert model.num_heads == 8


class TestChromaTP:
    def test_chroma_packed_qkv_like_flux(self, monkeypatch):
        import torch.nn as nn
        from comfy.distributed import patcher
        from comfy.distributed import parallel_linear as pl_module
        from comfy.distributed.parallel_linear import ParallelLinear

        class FakeMesh:
            world_size = 2
            rank = 0
            current_device = "cpu"
        monkeypatch.setattr(patcher, "get_mesh", lambda: FakeMesh())
        monkeypatch.setattr(pl_module, "get_mesh", lambda: FakeMesh())

        dim = 128

        class SelfAttention(nn.Module):
            def __init__(self):
                super().__init__()
                self.num_heads = 8
                self.qkv = nn.Linear(dim, dim * 3, bias=False)
                self.proj = nn.Linear(dim, dim)

        class DoubleStreamBlock(nn.Module):
            def __init__(self):
                super().__init__()
                self.num_heads = 8
                self.img_attn = SelfAttention()
                self.img_mlp = nn.Sequential(nn.Linear(dim, dim * 4), nn.GELU(), nn.Linear(dim * 4, dim))

        class SingleStreamBlock(nn.Module):
            def __init__(self):
                super().__init__()
                self.num_heads = 8
                self.linear1 = nn.Linear(dim, dim * 3 + dim * 4)
                self.linear2 = nn.Linear(dim + dim * 4, dim)

        class Chroma(nn.Module):
            def __init__(self):
                super().__init__()
                self.num_heads = 8
                self.double_blocks = nn.ModuleList([DoubleStreamBlock()])
                self.single_blocks = nn.ModuleList([SingleStreamBlock()])

        model = Chroma()
        assert patcher.parallelize_model(model) is True
        qkv = model.double_blocks[0].img_attn.qkv
        assert isinstance(qkv, ParallelLinear)
        assert qkv.mode == "colwise"
        assert qkv.pack_count == 3
        assert isinstance(model.double_blocks[0].img_attn.proj, ParallelLinear)
        assert model.double_blocks[0].img_attn.proj.mode == "rowwise"
        assert isinstance(model.double_blocks[0].img_mlp[0], ParallelLinear)
        assert model.double_blocks[0].img_mlp[0].mode == "colwise"
        assert isinstance(model.single_blocks[0].linear1, nn.Linear)
        assert model.double_blocks[0].num_heads == 4
        assert model.single_blocks[0].num_heads == 8
        assert model.num_heads == 8


class TestACEStepTP:
    """ACE-Step 1.0: head-split attention. ACE 1.5: GQA + unfused SwiGLU under decoder.layers."""

    def test_ace10_head_split(self, monkeypatch):
        import torch.nn as nn
        from comfy.distributed import patcher
        from comfy.distributed import parallel_linear as pl_module
        from comfy.distributed.parallel_linear import ParallelLinear

        class FakeMesh:
            world_size = 2
            rank = 0
            current_device = "cpu"
        monkeypatch.setattr(patcher, "get_mesh", lambda: FakeMesh())
        monkeypatch.setattr(pl_module, "get_mesh", lambda: FakeMesh())

        dim, heads, head_dim = 128, 8, 16

        class Attention(nn.Module):
            def __init__(self):
                super().__init__()
                self.heads = heads
                self.to_q = nn.Linear(dim, dim)
                self.to_k = nn.Linear(dim, dim)
                self.to_v = nn.Linear(dim, dim)
                self.to_out = nn.ModuleList([nn.Linear(dim, dim), nn.Dropout(0.0)])

        class LinearTransformerBlock(nn.Module):
            def __init__(self):
                super().__init__()
                self.attn = Attention()

        class ACEStepTransformer2DModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.num_attention_heads = heads
                self.transformer_blocks = nn.ModuleList([LinearTransformerBlock()])

        model = ACEStepTransformer2DModel()
        assert patcher.parallelize_model(model) is True
        assert isinstance(model.transformer_blocks[0].attn.to_q, ParallelLinear)
        assert model.transformer_blocks[0].attn.to_q.mode == "colwise"
        assert isinstance(model.transformer_blocks[0].attn.to_out[0], ParallelLinear)
        assert model.transformer_blocks[0].attn.to_out[0].mode == "rowwise"
        assert model.transformer_blocks[0].attn.heads == heads // 2

    def test_ace15_gqa_and_mlp_under_decoder(self, monkeypatch):
        import torch.nn as nn
        from comfy.distributed import patcher
        from comfy.distributed import parallel_linear as pl_module
        from comfy.distributed.parallel_linear import ParallelLinear

        class FakeMesh:
            world_size = 2
            rank = 0
            current_device = "cpu"
        monkeypatch.setattr(patcher, "get_mesh", lambda: FakeMesh())
        monkeypatch.setattr(pl_module, "get_mesh", lambda: FakeMesh())

        hidden, heads, kv_heads, head_dim = 128, 16, 8, 8
        q_dim = heads * head_dim
        kv_dim = kv_heads * head_dim
        ffn = 256

        class AceStepAttention(nn.Module):
            def __init__(self):
                super().__init__()
                self.num_heads = heads
                self.num_kv_heads = kv_heads
                self.head_dim = head_dim
                self.q_proj = nn.Linear(hidden, q_dim, bias=False)
                self.k_proj = nn.Linear(hidden, kv_dim, bias=False)
                self.v_proj = nn.Linear(hidden, kv_dim, bias=False)
                self.o_proj = nn.Linear(q_dim, hidden, bias=False)
                self.q_norm = nn.RMSNorm(head_dim)
                self.k_norm = nn.RMSNorm(head_dim)

        class MLP(nn.Module):
            def __init__(self):
                super().__init__()
                self.gate_proj = nn.Linear(hidden, ffn, bias=False)
                self.up_proj = nn.Linear(hidden, ffn, bias=False)
                self.down_proj = nn.Linear(ffn, hidden, bias=False)

        class AceStepDiTLayer(nn.Module):
            def __init__(self):
                super().__init__()
                self.self_attn = AceStepAttention()
                self.mlp = MLP()

        class LyricEncoder(nn.Module):
            def __init__(self):
                super().__init__()
                self.num_heads = heads
                self.num_kv_heads = kv_heads
                self.q_proj = nn.Linear(hidden, q_dim, bias=False)

        class AceStepConditionGenerationModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.decoder = nn.Module()
                self.decoder.layers = nn.ModuleList([AceStepDiTLayer()])
                self.lyric_encoder = LyricEncoder()

        model = AceStepConditionGenerationModel()
        assert patcher.parallelize_model(model) is True
        attn = model.decoder.layers[0].self_attn
        assert isinstance(attn.q_proj, ParallelLinear)
        assert attn.q_proj.mode == "colwise"
        assert isinstance(attn.k_proj, ParallelLinear)
        assert isinstance(attn.o_proj, ParallelLinear)
        assert attn.o_proj.mode == "rowwise"
        assert attn.num_heads == heads // 2
        assert attn.num_kv_heads == kv_heads // 2, "GQA kv heads must follow the k_proj shard"
        mlp = model.decoder.layers[0].mlp
        assert isinstance(mlp.gate_proj, ParallelLinear) and mlp.gate_proj.mode == "colwise"
        assert isinstance(mlp.down_proj, ParallelLinear) and mlp.down_proj.mode == "rowwise"
        # Lyric encoder is outside decoder.layers — must stay full-width.
        assert isinstance(model.lyric_encoder.q_proj, nn.Linear)
        assert model.lyric_encoder.num_heads == heads
        assert model.lyric_encoder.num_kv_heads == kv_heads


class TestNextDiTGQATP:
    """Lumina NextDiT / Z-Image: packed GQA qkv + unfused SwiGLU; adaLN stays."""

    def test_gqa_packed_qkv_and_ffn(self, monkeypatch):
        import torch.nn as nn
        from comfy.distributed import patcher
        from comfy.distributed import parallel_linear as pl_module
        from comfy.distributed.parallel_linear import ParallelLinear
        from comfy.distributed.patcher import TP_HEAD_SPLIT_MODELS

        class FakeMesh:
            world_size = 2
            rank = 0
            current_device = "cpu"
        monkeypatch.setattr(patcher, "get_mesh", lambda: FakeMesh())
        monkeypatch.setattr(pl_module, "get_mesh", lambda: FakeMesh())

        dim, heads, kv_heads, head_dim = 128, 8, 4, 16
        qkv_out = (heads + kv_heads + kv_heads) * head_dim
        assert "NextDiT" in TP_HEAD_SPLIT_MODELS

        class JointAttention(nn.Module):
            def __init__(self):
                super().__init__()
                self.n_local_heads = heads
                self.n_local_kv_heads = kv_heads
                self.n_kv_heads = kv_heads
                self.head_dim = head_dim
                self.qkv = nn.Linear(dim, qkv_out, bias=False)
                self.out = nn.Linear(heads * head_dim, dim, bias=False)

        class FeedForward(nn.Module):
            def __init__(self):
                super().__init__()
                self.w1 = nn.Linear(dim, dim * 2, bias=False)
                self.w3 = nn.Linear(dim, dim * 2, bias=False)
                self.w2 = nn.Linear(dim * 2, dim, bias=False)

        class JointTransformerBlock(nn.Module):
            def __init__(self):
                super().__init__()
                self.attention = JointAttention()
                self.feed_forward = FeedForward()
                self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(128, 4 * dim))

        class NextDiT(nn.Module):
            def __init__(self):
                super().__init__()
                self.n_heads = heads
                self.layers = nn.ModuleList([JointTransformerBlock()])
                self.noise_refiner = nn.ModuleList([JointTransformerBlock()])

        model = NextDiT()
        assert patcher.parallelize_model(model) is True
        qkv = model.layers[0].attention.qkv
        assert isinstance(qkv, ParallelLinear)
        assert qkv.mode == "colwise"
        assert qkv.pack_sizes == (heads * head_dim, kv_heads * head_dim, kv_heads * head_dim)
        assert qkv.local_out_features == qkv_out // 2
        assert isinstance(model.layers[0].attention.out, ParallelLinear)
        assert model.layers[0].attention.out.mode == "rowwise"
        assert isinstance(model.layers[0].feed_forward.w1, ParallelLinear)
        assert model.layers[0].feed_forward.w1.mode == "colwise"
        assert isinstance(model.layers[0].feed_forward.w2, ParallelLinear)
        assert model.layers[0].feed_forward.w2.mode == "rowwise"
        assert isinstance(model.layers[0].adaLN_modulation[1], nn.Linear)
        assert model.layers[0].attention.n_local_heads == heads // 2
        assert model.layers[0].attention.n_local_kv_heads == kv_heads // 2
        assert model.n_heads == heads
        assert isinstance(model.noise_refiner[0].attention.qkv, ParallelLinear)


class TestIdeogram4TP:
    def test_packed_qkv_and_ffn(self, monkeypatch):
        import torch.nn as nn
        from comfy.distributed import patcher
        from comfy.distributed import parallel_linear as pl_module
        from comfy.distributed.parallel_linear import ParallelLinear
        from comfy.distributed.patcher import TP_HEAD_SPLIT_MODELS

        class FakeMesh:
            world_size = 2
            rank = 0
            current_device = "cpu"
        monkeypatch.setattr(patcher, "get_mesh", lambda: FakeMesh())
        monkeypatch.setattr(pl_module, "get_mesh", lambda: FakeMesh())

        dim = 128
        assert "Ideogram4Transformer" in TP_HEAD_SPLIT_MODELS

        class Ideogram4Attention(nn.Module):
            def __init__(self):
                super().__init__()
                self.num_heads = 8
                self.head_dim = 16
                self.qkv = nn.Linear(dim, dim * 3, bias=False)
                self.o = nn.Linear(dim, dim, bias=False)

        class FeedForward(nn.Module):
            def __init__(self):
                super().__init__()
                self.w1 = nn.Linear(dim, dim * 2, bias=False)
                self.w3 = nn.Linear(dim, dim * 2, bias=False)
                self.w2 = nn.Linear(dim * 2, dim, bias=False)

        class Ideogram4TransformerBlock(nn.Module):
            def __init__(self):
                super().__init__()
                self.attention = Ideogram4Attention()
                self.feed_forward = FeedForward()
                self.adaln_modulation = nn.Linear(64, 4 * dim)

        class Ideogram4Transformer(nn.Module):
            def __init__(self):
                super().__init__()
                self.head_dim = 16
                self.layers = nn.ModuleList([Ideogram4TransformerBlock()])

        model = Ideogram4Transformer()
        assert patcher.parallelize_model(model) is True
        qkv = model.layers[0].attention.qkv
        assert isinstance(qkv, ParallelLinear)
        assert qkv.mode == "colwise"
        assert qkv.pack_count == 3
        assert qkv.local_out_features == 3 * (dim // 2)
        assert isinstance(model.layers[0].attention.o, ParallelLinear)
        assert model.layers[0].attention.o.mode == "rowwise"
        assert isinstance(model.layers[0].adaln_modulation, nn.Linear)
        assert isinstance(model.layers[0].feed_forward.w1, ParallelLinear)
        assert model.layers[0].feed_forward.w2.mode == "rowwise"
        assert model.layers[0].attention.num_heads == 4
        assert model.head_dim == 16


class TestJoyImageTP:
    def test_packed_double_stream_qkv(self, monkeypatch):
        import torch.nn as nn
        from comfy.distributed import patcher
        from comfy.distributed import parallel_linear as pl_module
        from comfy.distributed.parallel_linear import ParallelLinear

        class FakeMesh:
            world_size = 2
            rank = 0
            current_device = "cpu"
        monkeypatch.setattr(patcher, "get_mesh", lambda: FakeMesh())
        monkeypatch.setattr(pl_module, "get_mesh", lambda: FakeMesh())

        dim = 128

        class JoyImageAttention(nn.Module):
            def __init__(self):
                super().__init__()
                self.num_attention_heads = 8
                self.img_attn_qkv = nn.Linear(dim, dim * 3, bias=True)
                self.img_attn_proj = nn.Linear(dim, dim, bias=True)
                self.txt_attn_qkv = nn.Linear(dim, dim * 3, bias=True)
                self.txt_attn_proj = nn.Linear(dim, dim, bias=True)

        class JoyImageFeedForward(nn.Module):
            def __init__(self):
                super().__init__()
                self.net = nn.ModuleList([
                    nn.Sequential(nn.Linear(dim, dim * 4)),
                    nn.Identity(),
                    nn.Linear(dim * 4, dim),
                ])
                self.net[0].proj = self.net[0][0]

        class JoyImageTransformerBlock(nn.Module):
            def __init__(self):
                super().__init__()
                self.attn = JoyImageAttention()
                self.img_mlp = JoyImageFeedForward()
                self.txt_mlp = JoyImageFeedForward()

        class JoyImageTransformer3DModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.double_blocks = nn.ModuleList([JoyImageTransformerBlock()])

        model = JoyImageTransformer3DModel()
        assert patcher.parallelize_model(model) is True
        qkv = model.double_blocks[0].attn.img_attn_qkv
        assert isinstance(qkv, ParallelLinear)
        assert qkv.mode == "colwise"
        assert qkv.pack_count == 3
        assert isinstance(model.double_blocks[0].attn.txt_attn_qkv, ParallelLinear)
        assert model.double_blocks[0].attn.txt_attn_qkv.pack_count == 3
        assert model.double_blocks[0].attn.img_attn_proj.mode == "rowwise"
        assert model.double_blocks[0].attn.txt_attn_proj.mode == "rowwise"
        assert isinstance(model.double_blocks[0].img_mlp.net[2], ParallelLinear)
        assert model.double_blocks[0].img_mlp.net[2].mode == "rowwise"
        assert model.double_blocks[0].attn.num_attention_heads == 4


class TestLensTP:
    def test_packed_joint_qkv_and_swiglu(self, monkeypatch):
        import torch.nn as nn
        from comfy.distributed import patcher
        from comfy.distributed import parallel_linear as pl_module
        from comfy.distributed.parallel_linear import ParallelLinear

        class FakeMesh:
            world_size = 2
            rank = 0
            current_device = "cpu"
        monkeypatch.setattr(patcher, "get_mesh", lambda: FakeMesh())
        monkeypatch.setattr(pl_module, "get_mesh", lambda: FakeMesh())

        dim = 128

        class LensJointAttention(nn.Module):
            def __init__(self):
                super().__init__()
                self.heads = 8
                self.dim_head = 16
                self.img_qkv = nn.Linear(dim, 3 * dim, bias=True)
                self.txt_qkv = nn.Linear(dim, 3 * dim, bias=True)
                self.to_out = nn.ModuleList([nn.Linear(dim, dim), nn.Identity()])
                self.to_add_out = nn.Linear(dim, dim)

        class GateMLP(nn.Module):
            def __init__(self):
                super().__init__()
                self.w1 = nn.Linear(dim, dim * 2, bias=False)
                self.w3 = nn.Linear(dim, dim * 2, bias=False)
                self.w2 = nn.Linear(dim * 2, dim, bias=False)

        class LensTransformerBlock(nn.Module):
            def __init__(self):
                super().__init__()
                self.attn = LensJointAttention()
                self.img_mod = nn.Sequential(nn.SiLU(), nn.Linear(dim, 6 * dim))
                self.img_mlp = GateMLP()
                self.txt_mod = nn.Sequential(nn.SiLU(), nn.Linear(dim, 6 * dim))
                self.txt_mlp = GateMLP()

        class LensTransformer2DModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.transformer_blocks = nn.ModuleList([LensTransformerBlock()])

        model = LensTransformer2DModel()
        assert patcher.parallelize_model(model) is True
        qkv = model.transformer_blocks[0].attn.img_qkv
        assert isinstance(qkv, ParallelLinear)
        assert qkv.pack_count == 3
        assert isinstance(model.transformer_blocks[0].attn.txt_qkv, ParallelLinear)
        assert model.transformer_blocks[0].attn.to_out[0].mode == "rowwise"
        assert model.transformer_blocks[0].attn.to_add_out.mode == "rowwise"
        assert isinstance(model.transformer_blocks[0].img_mod[1], nn.Linear)
        assert model.transformer_blocks[0].img_mlp.w1.mode == "colwise"
        assert model.transformer_blocks[0].img_mlp.w2.mode == "rowwise"
        assert model.transformer_blocks[0].attn.heads == 4


class TestPixelDiTTP:
    """PixelDiT: packed qkv_x/qkv_y + pixel RotaryAttention.qkv; adaLN/compress stay."""

    def test_packed_joint_and_pixel_qkv(self, monkeypatch):
        import torch.nn as nn
        from comfy.distributed import patcher
        from comfy.distributed import parallel_linear as pl_module
        from comfy.distributed.parallel_linear import ParallelLinear
        from comfy.distributed.patcher import TP_HEAD_SPLIT_MODELS

        class FakeMesh:
            world_size = 2
            rank = 0
            current_device = "cpu"
        monkeypatch.setattr(patcher, "get_mesh", lambda: FakeMesh())
        monkeypatch.setattr(pl_module, "get_mesh", lambda: FakeMesh())

        dim, pixel_dim, attn_dim = 128, 16, 64
        patch_heads, pixel_heads = 8, 4
        assert "PixDiT_T2I" in TP_HEAD_SPLIT_MODELS

        class MMDiTJointAttention(nn.Module):
            def __init__(self):
                super().__init__()
                self.num_heads = patch_heads
                self.head_dim = dim // patch_heads
                self.qkv_x = nn.Linear(dim, dim * 3, bias=False)
                self.qkv_y = nn.Linear(dim, dim * 3, bias=False)
                self.proj_x = nn.Linear(dim, dim)
                self.proj_y = nn.Linear(dim, dim)

        class GateMLP(nn.Module):
            def __init__(self):
                super().__init__()
                self.w1 = nn.Linear(dim, dim * 2, bias=False)
                self.w3 = nn.Linear(dim, dim * 2, bias=False)
                self.w2 = nn.Linear(dim * 2, dim, bias=False)

        class MMDiTBlockT2I(nn.Module):
            def __init__(self):
                super().__init__()
                self.attn = MMDiTJointAttention()
                self.mlp_x = GateMLP()
                self.mlp_y = GateMLP()
                self.adaLN_modulation_img = nn.Sequential(nn.Linear(dim, 6 * dim))
                self.adaLN_modulation_txt = nn.Sequential(nn.Linear(dim, 6 * dim))

        class RotaryAttention(nn.Module):
            def __init__(self):
                super().__init__()
                self.num_heads = pixel_heads
                self.head_dim = attn_dim // pixel_heads
                self.qkv = nn.Linear(attn_dim, attn_dim * 3, bias=False)
                self.proj = nn.Linear(attn_dim, attn_dim)

        class PiTBlock(nn.Module):
            def __init__(self):
                super().__init__()
                self.num_heads = pixel_heads
                self.attn_dim = attn_dim
                self.compress_to_attn = nn.Linear(256 * pixel_dim, attn_dim)
                self.expand_from_attn = nn.Linear(attn_dim, 256 * pixel_dim)
                self.attn = RotaryAttention()
                self.adaLN_modulation_msa = nn.Linear(dim, 3 * pixel_dim * 256)
                self.adaLN_modulation_mlp = nn.Linear(dim, 3 * pixel_dim * 256)
                self.mlp = nn.Module()
                self.mlp.fc1 = nn.Linear(pixel_dim, pixel_dim * 4)
                self.mlp.fc2 = nn.Linear(pixel_dim * 4, pixel_dim)

        class PixDiT_T2I(nn.Module):
            def __init__(self):
                super().__init__()
                self.num_groups = patch_heads
                self.pixel_num_groups = pixel_heads
                self.patch_blocks = nn.ModuleList([MMDiTBlockT2I()])
                self.pixel_blocks = nn.ModuleList([PiTBlock()])

        model = PixDiT_T2I()
        assert patcher.parallelize_model(model) is True
        qkv_x = model.patch_blocks[0].attn.qkv_x
        assert isinstance(qkv_x, ParallelLinear)
        assert qkv_x.mode == "colwise"
        assert qkv_x.pack_count == 3
        assert isinstance(model.patch_blocks[0].attn.qkv_y, ParallelLinear)
        assert model.patch_blocks[0].attn.qkv_y.pack_count == 3
        assert model.patch_blocks[0].attn.proj_x.mode == "rowwise"
        assert model.patch_blocks[0].attn.proj_y.mode == "rowwise"
        assert isinstance(model.patch_blocks[0].adaLN_modulation_img[0], nn.Linear)
        assert isinstance(model.patch_blocks[0].mlp_x.w1, ParallelLinear)
        assert model.patch_blocks[0].mlp_x.w2.mode == "rowwise"
        assert model.patch_blocks[0].attn.num_heads == patch_heads // 2
        pixel_qkv = model.pixel_blocks[0].attn.qkv
        assert isinstance(pixel_qkv, ParallelLinear)
        assert pixel_qkv.pack_count == 3
        assert model.pixel_blocks[0].attn.proj.mode == "rowwise"
        assert isinstance(model.pixel_blocks[0].compress_to_attn, nn.Linear)
        assert isinstance(model.pixel_blocks[0].expand_from_attn, nn.Linear)
        assert isinstance(model.pixel_blocks[0].adaLN_modulation_msa, nn.Linear)
        assert isinstance(model.pixel_blocks[0].mlp.fc1, ParallelLinear)
        assert model.pixel_blocks[0].mlp.fc1.mode == "colwise"
        assert model.pixel_blocks[0].mlp.fc2.mode == "rowwise"
        assert model.pixel_blocks[0].attn.num_heads == pixel_heads // 2
        assert model.pixel_blocks[0].num_heads == pixel_heads
        assert model.num_groups == patch_heads


class TestMochiTP:
    """Mochi AsymmDiTJoint: packed qkv_x/qkv_y + packed SwiGLU w1; per-head RoPE sliced."""

    def test_packed_qkv_swiglu_and_pos_frequencies(self, monkeypatch):
        import torch
        import torch.nn as nn
        from comfy.distributed import patcher
        from comfy.distributed import parallel_linear as pl_module
        from comfy.distributed.parallel_linear import ParallelLinear
        from comfy.distributed.patcher import TP_HEAD_SPLIT_MODELS

        class FakeMesh:
            world_size = 2
            rank = 0
            current_device = "cpu"
        monkeypatch.setattr(patcher, "get_mesh", lambda: FakeMesh())
        monkeypatch.setattr(pl_module, "get_mesh", lambda: FakeMesh())

        dim, heads, head_dim = 128, 8, 16
        assert "AsymmDiTJoint" in TP_HEAD_SPLIT_MODELS

        class AsymmetricAttention(nn.Module):
            def __init__(self):
                super().__init__()
                self.num_heads = heads
                self.head_dim = head_dim
                self.qkv_x = nn.Linear(dim, dim * 3, bias=True)
                self.qkv_y = nn.Linear(dim, dim * 3, bias=True)
                self.proj_x = nn.Linear(dim, dim)
                self.proj_y = nn.Linear(dim, dim)

        class FeedForward(nn.Module):
            def __init__(self):
                super().__init__()
                hidden = 64
                self.w1 = nn.Linear(dim, 2 * hidden, bias=False)
                self.w2 = nn.Linear(hidden, dim, bias=False)

        class AsymmetricJointBlock(nn.Module):
            def __init__(self):
                super().__init__()
                self.mod_x = nn.Linear(dim, 4 * dim)
                self.mod_y = nn.Linear(dim, 4 * dim)
                self.attn = AsymmetricAttention()
                self.mlp_x = FeedForward()
                self.mlp_y = FeedForward()

        class AsymmDiTJoint(nn.Module):
            def __init__(self):
                super().__init__()
                self.num_heads = heads
                self.head_dim = head_dim
                self.pos_frequencies = nn.Parameter(torch.arange(3 * heads * (head_dim // 2), dtype=torch.float32).reshape(3, heads, head_dim // 2))
                self.blocks = nn.ModuleList([AsymmetricJointBlock()])

        model = AsymmDiTJoint()
        full_pf = model.pos_frequencies.data.clone()
        assert patcher.parallelize_model(model) is True
        qkv = model.blocks[0].attn.qkv_x
        assert isinstance(qkv, ParallelLinear)
        assert qkv.pack_count == 3
        assert isinstance(model.blocks[0].attn.qkv_y, ParallelLinear)
        assert model.blocks[0].attn.proj_x.mode == "rowwise"
        assert isinstance(model.blocks[0].mod_x, nn.Linear)
        assert isinstance(model.blocks[0].mlp_x.w1, ParallelLinear)
        assert model.blocks[0].mlp_x.w1.pack_count == 2
        assert model.blocks[0].mlp_x.w2.mode == "rowwise"
        assert model.blocks[0].attn.num_heads == heads // 2
        assert model.num_heads == heads
        assert model.pos_frequencies.shape == (3, heads // 2, head_dim // 2)
        torch.testing.assert_close(model.pos_frequencies.data, full_pf[:, : heads // 2])


class TestHunYuanDiTTP:
    """HunyuanDiT: packed Wqkv + packed kv_proj; modulation and skip stay."""

    def test_packed_self_attn_and_cross_kv(self, monkeypatch):
        import torch.nn as nn
        from comfy.distributed import patcher
        from comfy.distributed import parallel_linear as pl_module
        from comfy.distributed.parallel_linear import ParallelLinear
        from comfy.distributed.patcher import TP_HEAD_SPLIT_MODELS

        class FakeMesh:
            world_size = 2
            rank = 0
            current_device = "cpu"
        monkeypatch.setattr(patcher, "get_mesh", lambda: FakeMesh())
        monkeypatch.setattr(pl_module, "get_mesh", lambda: FakeMesh())

        dim, ctx, heads = 128, 64, 8
        assert "HunYuanDiT" in TP_HEAD_SPLIT_MODELS

        class Attention(nn.Module):
            def __init__(self):
                super().__init__()
                self.num_heads = heads
                self.head_dim = dim // heads
                self.Wqkv = nn.Linear(dim, dim * 3, bias=True)
                self.out_proj = nn.Linear(dim, dim)

        class CrossAttention(nn.Module):
            def __init__(self):
                super().__init__()
                self.num_heads = heads
                self.head_dim = dim // heads
                self.q_proj = nn.Linear(dim, dim, bias=True)
                self.kv_proj = nn.Linear(ctx, 2 * dim, bias=True)
                self.out_proj = nn.Linear(dim, dim)

        class HunYuanDiTBlock(nn.Module):
            def __init__(self, skip=False):
                super().__init__()
                self.attn1 = Attention()
                self.attn2 = CrossAttention()
                self.mlp = nn.Module()
                self.mlp.fc1 = nn.Linear(dim, dim * 4)
                self.mlp.fc2 = nn.Linear(dim * 4, dim)
                self.default_modulation = nn.Sequential(nn.SiLU(), nn.Linear(dim, dim))
                self.skip_linear = nn.Linear(2 * dim, dim) if skip else None

        class HunYuanDiT(nn.Module):
            def __init__(self):
                super().__init__()
                self.num_heads = heads
                self.blocks = nn.ModuleList([HunYuanDiTBlock(skip=False), HunYuanDiTBlock(skip=True)])

        model = HunYuanDiT()
        assert patcher.parallelize_model(model) is True
        wqkv = model.blocks[0].attn1.Wqkv
        assert isinstance(wqkv, ParallelLinear)
        assert wqkv.pack_count == 3
        assert model.blocks[0].attn1.out_proj.mode == "rowwise"
        kv = model.blocks[0].attn2.kv_proj
        assert isinstance(kv, ParallelLinear)
        assert kv.pack_count == 2
        assert isinstance(model.blocks[0].attn2.q_proj, ParallelLinear)
        assert model.blocks[0].attn2.q_proj.mode == "colwise"
        assert model.blocks[0].attn2.out_proj.mode == "rowwise"
        assert isinstance(model.blocks[0].default_modulation[1], nn.Linear)
        assert isinstance(model.blocks[1].skip_linear, nn.Linear)
        assert model.blocks[0].attn1.num_heads == heads // 2
        assert model.blocks[0].attn2.num_heads == heads // 2
        assert model.num_heads == heads
        assert isinstance(model.blocks[0].mlp.fc1, ParallelLinear)
        assert model.blocks[0].mlp.fc2.mode == "rowwise"


class TestPixArtTP:
    """PixArtMS: packed attn.qkv + packed kv_linear; KV-compress / QK-norm stay."""

    def test_packed_qkv_and_keep_replicated_blocks(self, monkeypatch):
        import torch.nn as nn
        from comfy.distributed import patcher
        from comfy.distributed import parallel_linear as pl_module
        from comfy.distributed.parallel_linear import ParallelLinear
        from comfy.distributed.patcher import TP_HEAD_SPLIT_MODELS

        class FakeMesh:
            world_size = 2
            rank = 0
            current_device = "cpu"
        monkeypatch.setattr(patcher, "get_mesh", lambda: FakeMesh())
        monkeypatch.setattr(pl_module, "get_mesh", lambda: FakeMesh())

        dim, heads, head_dim = 64, 8, 8
        assert "PixArtMS" in TP_HEAD_SPLIT_MODELS

        class AttentionKVCompress(nn.Module):
            def __init__(self, sr_ratio=1, qk_norm=False):
                super().__init__()
                self.num_heads = heads
                self.head_dim = head_dim
                self.sr_ratio = sr_ratio
                self.qkv = nn.Linear(dim, dim * 3, bias=True)
                self.proj = nn.Linear(dim, dim)
                self.q_norm = nn.LayerNorm(dim) if qk_norm else nn.Identity()

        class MultiHeadCrossAttention(nn.Module):
            def __init__(self):
                super().__init__()
                self.num_heads = heads
                self.head_dim = head_dim
                self.q_linear = nn.Linear(dim, dim)
                self.kv_linear = nn.Linear(dim, dim * 2)
                self.proj = nn.Linear(dim, dim)

        class PixArtMSBlock(nn.Module):
            def __init__(self, sr_ratio=1, qk_norm=False):
                super().__init__()
                self.attn = AttentionKVCompress(sr_ratio=sr_ratio, qk_norm=qk_norm)
                self.cross_attn = MultiHeadCrossAttention()
                self.mlp = nn.Module()
                self.mlp.fc1 = nn.Linear(dim, dim * 4)
                self.mlp.fc2 = nn.Linear(dim * 4, dim)

        class PixArtMS(nn.Module):
            def __init__(self):
                super().__init__()
                self.num_heads = heads
                self.t_block = nn.Sequential(nn.SiLU(), nn.Linear(dim, 6 * dim))
                self.blocks = nn.ModuleList([
                    PixArtMSBlock(),
                    PixArtMSBlock(sr_ratio=2),
                    PixArtMSBlock(qk_norm=True),
                ])

        model = PixArtMS()
        assert patcher.parallelize_model(model) is True

        packed = model.blocks[0]
        qkv = packed.attn.qkv
        assert isinstance(qkv, ParallelLinear)
        assert qkv.pack_count == 3
        assert packed.attn.proj.mode == "rowwise"
        kv = packed.cross_attn.kv_linear
        assert isinstance(kv, ParallelLinear)
        assert kv.pack_count == 2
        assert isinstance(packed.cross_attn.q_linear, ParallelLinear)
        assert packed.cross_attn.q_linear.mode == "colwise"
        assert packed.cross_attn.proj.mode == "rowwise"
        assert isinstance(packed.mlp.fc1, ParallelLinear)
        assert packed.mlp.fc2.mode == "rowwise"
        assert packed.attn.num_heads == heads // 2
        assert packed.cross_attn.num_heads == heads // 2

        compressed = model.blocks[1]
        assert isinstance(compressed.attn.qkv, nn.Linear)
        assert not isinstance(compressed.attn.qkv, ParallelLinear)
        assert isinstance(compressed.attn.proj, nn.Linear)
        assert not isinstance(compressed.attn.proj, ParallelLinear)
        assert compressed.attn.num_heads == heads
        assert isinstance(compressed.cross_attn.kv_linear, ParallelLinear)
        assert compressed.cross_attn.num_heads == heads // 2

        qknorm = model.blocks[2]
        assert isinstance(qknorm.attn.qkv, nn.Linear)
        assert not isinstance(qknorm.attn.qkv, ParallelLinear)
        assert qknorm.attn.num_heads == heads

        assert isinstance(model.t_block[1], nn.Linear)
        assert not isinstance(model.t_block[1], ParallelLinear)
        assert model.num_heads == heads

    def test_self_attn_reshape_uses_qkv_width_not_residual(self, monkeypatch):
        import torch
        import torch.nn as nn
        import comfy.ldm.pixart.blocks as pixart_blocks
        from comfy.ldm.pixart.blocks import AttentionKVCompress

        def fake_attn(q, k, v, heads, mask=None, skip_reshape=False):
            # Packed local inner is dim/2. skip_reshape inputs are (B, H, N, D).
            B = q.shape[0]
            if skip_reshape:
                heads_dim, seq, head_dim = q.shape[1], q.shape[2], q.shape[3]
                return torch.zeros(B, seq, heads_dim * head_dim)
            return torch.zeros_like(q)

        monkeypatch.setattr(pixart_blocks, "optimized_attention", fake_attn)

        dim, heads = 32, 4
        attn = AttentionKVCompress(
            dim=dim, num_heads=heads, sr_ratio=1, qk_norm=False, operations=nn,
        )
        # Simulate packed colwise (ws=2): local QKV is 3 * (dim/2), local heads.
        attn.qkv = nn.Linear(dim, dim * 3 // 2, bias=True)
        attn.num_heads = heads // 2
        attn.proj = nn.Linear(dim // 2, dim, bias=True)
        x = torch.randn(2, 16, dim)
        out = attn(x, HW=(4, 4))
        assert out.shape == (2, 16, dim)

    def test_cross_attn_reshape_uses_local_inner(self, monkeypatch):
        import torch
        import torch.nn as nn
        import comfy.ldm.pixart.blocks as pixart_blocks
        from comfy.ldm.pixart.blocks import MultiHeadCrossAttention

        def fake_attn(q, k, v, heads, mask=None, skip_reshape=False):
            return torch.zeros_like(q)

        monkeypatch.setattr(pixart_blocks, "optimized_attention", fake_attn)

        dim, heads = 32, 4
        ca = MultiHeadCrossAttention(d_model=dim, num_heads=heads, operations=nn)
        ca.q_linear = nn.Linear(dim, dim // 2)
        ca.kv_linear = nn.Linear(dim, dim)
        ca.num_heads = heads // 2
        ca.head_dim = dim // heads
        ca.proj = nn.Linear(dim // 2, dim)
        x = torch.randn(2, 16, dim)
        cond = torch.randn(2, 8, dim)
        out = ca(x, cond)
        assert out.shape == (2, 16, dim)


class TestAudioDitTP:
    """Stable Audio DiT: packed to_qkv / to_kv / GLU; conformer and adaLN stay."""

    def test_packed_self_cross_glu_and_exclusions(self, monkeypatch):
        import torch.nn as nn
        from comfy.distributed import patcher
        from comfy.distributed import parallel_linear as pl_module
        from comfy.distributed.parallel_linear import ParallelLinear
        from comfy.distributed.patcher import TP_HEAD_SPLIT_MODELS

        class FakeMesh:
            world_size = 2
            rank = 0
            current_device = "cpu"
        monkeypatch.setattr(patcher, "get_mesh", lambda: FakeMesh())
        monkeypatch.setattr(pl_module, "get_mesh", lambda: FakeMesh())

        dim, heads, head_dim, inner = 64, 8, 8, 256
        assert "AudioDiffusionTransformer" in TP_HEAD_SPLIT_MODELS

        class GLU(nn.Module):
            def __init__(self):
                super().__init__()
                self.proj = nn.Linear(dim, inner * 2)

        class FeedForward(nn.Module):
            def __init__(self):
                super().__init__()
                self.ff = nn.Sequential(GLU(), nn.Identity(), nn.Linear(inner, dim), nn.Identity())

        class SelfAttention(nn.Module):
            def __init__(self, differential=False):
                super().__init__()
                self.num_heads = heads
                self.kv_heads = heads
                self.dim_heads = head_dim
                pack = 5 if differential else 3
                self.to_qkv = nn.Linear(dim, dim * pack, bias=False)
                self.to_out = nn.Linear(dim, dim, bias=False)

        class CrossAttention(nn.Module):
            def __init__(self):
                super().__init__()
                self.num_heads = heads
                self.kv_heads = heads
                self.dim_heads = head_dim
                self.to_q = nn.Linear(dim, dim, bias=False)
                self.to_kv = nn.Linear(dim, dim * 2, bias=False)
                self.to_out = nn.Linear(dim, dim, bias=False)

        class ConformerModule(nn.Module):
            def __init__(self):
                super().__init__()
                self.glu = GLU()

        class TransformerBlock(nn.Module):
            def __init__(self, differential=False, conformer=False):
                super().__init__()
                self.self_attn = SelfAttention(differential=differential)
                self.cross_attn = CrossAttention()
                self.ff = FeedForward()
                self.to_scale_shift_gate = nn.Sequential(nn.SiLU(), nn.Linear(dim, dim * 6, bias=False))
                self.conformer = ConformerModule() if conformer else None

        class ContinuousTransformer(nn.Module):
            def __init__(self):
                super().__init__()
                self.project_in = nn.Linear(32, dim, bias=False)
                self.project_out = nn.Linear(dim, 32, bias=False)
                self.layers = nn.ModuleList([
                    TransformerBlock(),
                    TransformerBlock(differential=True, conformer=True),
                ])

        class AudioDiffusionTransformer(nn.Module):
            def __init__(self):
                super().__init__()
                self.transformer = ContinuousTransformer()

        model = AudioDiffusionTransformer()
        assert patcher.parallelize_model(model) is True

        packed = model.transformer.layers[0]
        qkv = packed.self_attn.to_qkv
        assert isinstance(qkv, ParallelLinear)
        assert qkv.pack_count == 3
        assert packed.self_attn.to_out.mode == "rowwise"
        assert packed.self_attn.num_heads == heads // 2
        assert packed.self_attn.kv_heads == heads // 2
        kv = packed.cross_attn.to_kv
        assert isinstance(kv, ParallelLinear)
        assert kv.pack_count == 2
        assert isinstance(packed.cross_attn.to_q, ParallelLinear)
        assert packed.cross_attn.to_q.mode == "colwise"
        assert packed.cross_attn.to_out.mode == "rowwise"
        glu = packed.ff.ff[0].proj
        assert isinstance(glu, ParallelLinear)
        assert glu.pack_count == 2
        assert glu.mode == "colwise"
        assert packed.ff.ff[2].mode == "rowwise"
        assert isinstance(packed.to_scale_shift_gate[1], nn.Linear)
        assert not isinstance(packed.to_scale_shift_gate[1], ParallelLinear)

        diff_block = model.transformer.layers[1]
        assert isinstance(diff_block.self_attn.to_qkv, ParallelLinear)
        assert diff_block.self_attn.to_qkv.pack_count == 5
        assert isinstance(diff_block.conformer.glu.proj, nn.Linear)
        assert not isinstance(diff_block.conformer.glu.proj, ParallelLinear)
        assert isinstance(model.transformer.project_in, nn.Linear)
        assert not isinstance(model.transformer.project_in, ParallelLinear)
        assert isinstance(model.transformer.project_out, nn.Linear)


class TestBooguTP:
    """Boogu 28/7 GQA: shard Q heads, keep K/V replicated when kv_heads % ws != 0."""

    def test_q_split_kv_replicated(self, monkeypatch):
        import torch.nn as nn
        from comfy.distributed import patcher
        from comfy.distributed import parallel_linear as pl_module
        from comfy.distributed.parallel_linear import ParallelLinear
        from comfy.distributed.patcher import TP_HEAD_SPLIT_MODELS

        class FakeMesh:
            world_size = 2
            rank = 0
            current_device = "cpu"
        monkeypatch.setattr(patcher, "get_mesh", lambda: FakeMesh())
        monkeypatch.setattr(pl_module, "get_mesh", lambda: FakeMesh())

        dim, heads, kv_heads, head_dim = 64, 8, 1, 8
        inner = 128
        assert "BooguTransformer2DModel" in TP_HEAD_SPLIT_MODELS

        class Attention(nn.Module):
            def __init__(self):
                super().__init__()
                self.heads = heads
                self.kv_heads = kv_heads
                self.dim_head = head_dim
                self.to_q = nn.Linear(dim, heads * head_dim, bias=False)
                self.to_k = nn.Linear(dim, kv_heads * head_dim, bias=False)
                self.to_v = nn.Linear(dim, kv_heads * head_dim, bias=False)
                self.to_out = nn.Sequential(nn.Linear(heads * head_dim, dim, bias=False), nn.Dropout(0.0))
                self.norm_q = nn.RMSNorm(head_dim)
                self.norm_k = nn.RMSNorm(head_dim)

        class Processor(nn.Module):
            def __init__(self):
                super().__init__()
                self.heads = heads
                self.kv_heads = kv_heads
                self.dim_head = head_dim
                self.img_to_q = nn.Linear(dim, heads * head_dim, bias=False)
                self.img_to_k = nn.Linear(dim, kv_heads * head_dim, bias=False)
                self.img_to_v = nn.Linear(dim, kv_heads * head_dim, bias=False)
                self.instruct_to_q = nn.Linear(dim, heads * head_dim, bias=False)
                self.instruct_to_k = nn.Linear(dim, kv_heads * head_dim, bias=False)
                self.instruct_to_v = nn.Linear(dim, kv_heads * head_dim, bias=False)
                self.instruct_out = nn.Linear(dim, dim, bias=False)
                self.img_out = nn.Linear(dim, dim, bias=False)

        class BooguJointAttention(nn.Module):
            def __init__(self):
                super().__init__()
                self.heads = heads
                self.kv_heads = kv_heads
                self.dim_head = head_dim
                self.to_out = nn.Sequential(nn.Linear(heads * head_dim, dim, bias=False), nn.Dropout(0.0))
                self.processor = Processor()

        class FeedForward(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear_1 = nn.Linear(dim, inner, bias=False)
                self.linear_2 = nn.Linear(inner, dim, bias=False)
                self.linear_3 = nn.Linear(dim, inner, bias=False)

        class OmniGen2TransformerBlock(nn.Module):
            def __init__(self):
                super().__init__()
                self.attn = Attention()
                self.feed_forward = FeedForward()
                self.norm1 = nn.Module()
                self.norm1.linear = nn.Linear(dim, 4 * dim)

        class BooguDoubleStreamBlock(nn.Module):
            def __init__(self):
                super().__init__()
                self.img_instruct_attn = BooguJointAttention()
                self.img_self_attn = Attention()
                self.img_feed_forward = FeedForward()
                self.instruct_feed_forward = FeedForward()
                self.img_norm1 = nn.Module()
                self.img_norm1.linear = nn.Linear(dim, 4 * dim)

        class BooguTransformer2DModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.noise_refiner = nn.ModuleList([OmniGen2TransformerBlock()])
                self.ref_image_refiner = nn.ModuleList([OmniGen2TransformerBlock()])
                self.context_refiner = nn.ModuleList([OmniGen2TransformerBlock()])
                self.double_stream_layers = nn.ModuleList([BooguDoubleStreamBlock()])
                self.single_stream_layers = nn.ModuleList([OmniGen2TransformerBlock()])
                self.x_embedder = nn.Linear(16, dim)

        model = BooguTransformer2DModel()
        assert patcher.parallelize_model(model) is True

        single = model.single_stream_layers[0]
        assert isinstance(single.attn.to_q, ParallelLinear)
        assert single.attn.to_q.mode == "colwise"
        assert isinstance(single.attn.to_k, nn.Linear)
        assert not isinstance(single.attn.to_k, ParallelLinear)
        assert isinstance(single.attn.to_v, nn.Linear)
        assert not isinstance(single.attn.to_v, ParallelLinear)
        assert single.attn.to_out[0].mode == "rowwise"
        assert single.attn.heads == heads // 2
        assert single.attn.kv_heads == kv_heads
        assert isinstance(single.feed_forward.linear_1, ParallelLinear)
        assert single.feed_forward.linear_1.mode == "colwise"
        assert isinstance(single.feed_forward.linear_3, ParallelLinear)
        assert single.feed_forward.linear_2.mode == "rowwise"
        assert isinstance(single.norm1.linear, nn.Linear)
        assert not isinstance(single.norm1.linear, ParallelLinear)

        joint = model.double_stream_layers[0].img_instruct_attn
        proc = joint.processor
        assert isinstance(proc.img_to_q, ParallelLinear)
        assert isinstance(proc.instruct_to_q, ParallelLinear)
        assert isinstance(proc.img_to_k, nn.Linear)
        assert not isinstance(proc.img_to_k, ParallelLinear)
        assert isinstance(proc.instruct_to_v, nn.Linear)
        assert not isinstance(proc.instruct_to_v, ParallelLinear)
        assert proc.img_out.mode == "rowwise"
        assert proc.instruct_out.mode == "rowwise"
        assert joint.to_out[0].mode == "rowwise"
        assert joint.heads == heads // 2
        assert joint.kv_heads == kv_heads
        assert model.double_stream_layers[0].img_self_attn.heads == heads // 2
        assert isinstance(model.x_embedder, nn.Linear)
        assert not isinstance(model.x_embedder, ParallelLinear)


class TestKrea2TP:
    """Krea2 SingleStreamDiT: unfused GQA 48/12 — both divide by 2. Shard Q+KV, keep txtfusion."""

    def test_gqa_wq_wk_and_gate(self, monkeypatch):
        import torch.nn as nn
        from comfy.distributed import patcher
        from comfy.distributed import parallel_linear as pl_module
        from comfy.distributed.parallel_linear import ParallelLinear
        from comfy.distributed.patcher import TP_HEAD_SPLIT_MODELS

        class FakeMesh:
            world_size = 2
            rank = 0
            current_device = "cpu"
        monkeypatch.setattr(patcher, "get_mesh", lambda: FakeMesh())
        monkeypatch.setattr(pl_module, "get_mesh", lambda: FakeMesh())

        dim, heads, kvheads, headdim = 64, 8, 2, 8
        assert "SingleStreamDiT" in TP_HEAD_SPLIT_MODELS

        class Attention(nn.Module):
            def __init__(self):
                super().__init__()
                self.heads = heads
                self.kvheads = kvheads
                self.headdim = headdim
                self.wq = nn.Linear(dim, heads * headdim, bias=False)
                self.wk = nn.Linear(dim, kvheads * headdim, bias=False)
                self.wv = nn.Linear(dim, kvheads * headdim, bias=False)
                self.gate = nn.Linear(dim, dim, bias=False)
                self.wo = nn.Linear(dim, dim, bias=False)

        class SwiGLU(nn.Module):
            def __init__(self):
                super().__init__()
                self.gate = nn.Linear(dim, dim * 2, bias=False)
                self.up = nn.Linear(dim, dim * 2, bias=False)
                self.down = nn.Linear(dim * 2, dim, bias=False)

        class SingleStreamBlock(nn.Module):
            def __init__(self):
                super().__init__()
                self.attn = Attention()
                self.mlp = SwiGLU()

        class TextFusionTransformer(nn.Module):
            def __init__(self):
                super().__init__()
                self.attn = Attention()

        class SingleStreamDiT(nn.Module):
            def __init__(self):
                super().__init__()
                self.heads = heads
                self.blocks = nn.ModuleList([SingleStreamBlock()])
                self.txtfusion = TextFusionTransformer()
                self.tproj = nn.Sequential(nn.GELU(), nn.Linear(dim, dim * 6))

        model = SingleStreamDiT()
        assert patcher.parallelize_model(model) is True
        attn = model.blocks[0].attn
        assert isinstance(attn.wq, ParallelLinear)
        assert attn.wq.mode == "colwise"
        assert isinstance(attn.wk, ParallelLinear)
        assert isinstance(attn.wv, ParallelLinear)
        assert isinstance(attn.gate, ParallelLinear)
        assert attn.gate.mode == "colwise"
        assert attn.wo.mode == "rowwise"
        assert attn.heads == heads // 2
        assert attn.kvheads == kvheads // 2
        assert isinstance(model.blocks[0].mlp.gate, ParallelLinear)
        assert model.blocks[0].mlp.down.mode == "rowwise"
        assert isinstance(model.txtfusion.attn.wq, nn.Linear)
        assert not isinstance(model.txtfusion.attn.wq, ParallelLinear)
        assert isinstance(model.tproj[1], nn.Linear)
        assert not isinstance(model.tproj[1], ParallelLinear)
        assert model.heads == heads


class TestMageFlowTP:
    """MageFlow wraps QwenImageTransformerBlock under transformer_blocks."""

    def test_qwen_blocks_head_split(self, monkeypatch):
        import torch.nn as nn
        from comfy.distributed import patcher
        from comfy.distributed import parallel_linear as pl_module
        from comfy.distributed.parallel_linear import ParallelLinear
        from comfy.distributed.patcher import TP_HEAD_SPLIT_MODELS

        class FakeMesh:
            world_size = 2
            rank = 0
            current_device = "cpu"
        monkeypatch.setattr(patcher, "get_mesh", lambda: FakeMesh())
        monkeypatch.setattr(pl_module, "get_mesh", lambda: FakeMesh())

        dim, heads = 64, 8
        assert "MageFlowTransformer2DModel" in TP_HEAD_SPLIT_MODELS

        class Attn(nn.Module):
            def __init__(self):
                super().__init__()
                self.heads = heads
                self.dim_head = dim // heads
                self.to_q = nn.Linear(dim, dim)
                self.to_k = nn.Linear(dim, dim)
                self.to_v = nn.Linear(dim, dim)
                self.to_out = nn.ModuleList([nn.Linear(dim, dim)])

        class Block(nn.Module):
            def __init__(self):
                super().__init__()
                self.attn = Attn()
                self.img_mod = nn.Sequential(nn.SiLU(), nn.Linear(dim, 6 * dim))
                self.img_mlp = nn.Module()
                self.img_mlp.net = nn.ModuleList([nn.Module(), nn.Identity(), nn.Linear(dim * 4, dim)])
                self.img_mlp.net[0].proj = nn.Linear(dim, dim * 4)

        class MageFlowTransformer2DModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.transformer_blocks = nn.ModuleList([Block()])
                self.img_in = nn.Linear(16, dim)

        model = MageFlowTransformer2DModel()
        assert patcher.parallelize_model(model) is True
        attn = model.transformer_blocks[0].attn
        assert isinstance(attn.to_q, ParallelLinear)
        assert attn.to_out[0].mode == "rowwise"
        assert attn.heads == heads // 2
        assert isinstance(model.transformer_blocks[0].img_mod[1], nn.Linear)
        assert not isinstance(model.transformer_blocks[0].img_mod[1], ParallelLinear)
        assert isinstance(model.img_in, nn.Linear)


class TestHunYuanDiTPlainTP:
    """Hunyuan3D HunYuanDiTPlain: unfused QKV + dense MLP; MoE last layers stay."""

    def test_unfused_attn_dense_mlp_moe_replicated(self, monkeypatch):
        import torch.nn as nn
        from comfy.distributed import patcher
        from comfy.distributed import parallel_linear as pl_module
        from comfy.distributed.parallel_linear import ParallelLinear
        from comfy.distributed.patcher import TP_HEAD_SPLIT_MODELS

        class FakeMesh:
            world_size = 2
            rank = 0
            current_device = "cpu"
        monkeypatch.setattr(patcher, "get_mesh", lambda: FakeMesh())
        monkeypatch.setattr(pl_module, "get_mesh", lambda: FakeMesh())

        dim, ctx, heads = 64, 32, 8
        assert "HunYuanDiTPlain" in TP_HEAD_SPLIT_MODELS

        class Attention(nn.Module):
            def __init__(self):
                super().__init__()
                self.num_heads = heads
                self.head_dim = dim // heads
                self.to_q = nn.Linear(dim, dim)
                self.to_k = nn.Linear(dim, dim)
                self.to_v = nn.Linear(dim, dim)
                self.out_proj = nn.Linear(dim, dim)

        class CrossAttention(nn.Module):
            def __init__(self):
                super().__init__()
                self.num_heads = heads
                self.head_dim = dim // heads
                self.to_q = nn.Linear(dim, dim)
                self.to_k = nn.Linear(ctx, dim)
                self.to_v = nn.Linear(ctx, dim)
                self.out_proj = nn.Linear(dim, dim)

        class MLP(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc1 = nn.Linear(dim, dim * 4)
                self.fc2 = nn.Linear(dim * 4, dim)

        class MoEBlock(nn.Module):
            def __init__(self):
                super().__init__()
                self.experts = nn.ModuleList([nn.Linear(dim, dim * 4), nn.Linear(dim, dim * 4)])
                self.shared_experts = nn.Linear(dim, dim * 4)

        class HunYuanDiTBlock(nn.Module):
            def __init__(self, use_moe=False, skip=False):
                super().__init__()
                self.attn1 = Attention()
                self.attn2 = CrossAttention()
                if skip:
                    self.skip_linear = nn.Linear(dim * 2, dim)
                if use_moe:
                    self.moe = MoEBlock()
                else:
                    self.mlp = MLP()

        class HunYuanDiTPlain(nn.Module):
            def __init__(self):
                super().__init__()
                self.num_heads = heads
                self.x_embedder = nn.Linear(16, dim)
                self.blocks = nn.ModuleList([
                    HunYuanDiTBlock(use_moe=False, skip=False),
                    HunYuanDiTBlock(use_moe=True, skip=True),
                ])

        model = HunYuanDiTPlain()
        assert patcher.parallelize_model(model) is True
        dense = model.blocks[0]
        moe = model.blocks[1]
        assert isinstance(dense.attn1.to_q, ParallelLinear)
        assert dense.attn1.to_q.mode == "colwise"
        assert dense.attn1.out_proj.mode == "rowwise"
        assert dense.attn1.num_heads == heads // 2
        assert isinstance(dense.attn2.to_k, ParallelLinear)
        assert dense.attn2.out_proj.mode == "rowwise"
        assert dense.attn2.num_heads == heads // 2
        assert isinstance(dense.mlp.fc1, ParallelLinear)
        assert dense.mlp.fc1.mode == "colwise"
        assert dense.mlp.fc2.mode == "rowwise"
        assert isinstance(moe.moe.experts[0], nn.Linear)
        assert not isinstance(moe.moe.experts[0], ParallelLinear)
        assert isinstance(moe.moe.shared_experts, nn.Linear)
        assert not isinstance(moe.moe.shared_experts, ParallelLinear)
        assert isinstance(moe.skip_linear, nn.Linear)
        assert not isinstance(moe.skip_linear, ParallelLinear)
        assert moe.attn1.num_heads == heads // 2
        assert isinstance(model.x_embedder, nn.Linear)
        assert not isinstance(model.x_embedder, ParallelLinear)
        assert model.num_heads == heads


class TestCogVideoXTP:
    """CogVideoX: unfused joint QKV, attn_out/ff_out rowwise, adaLN 6-way chunk stays."""

    def test_unfused_qkv_and_ff_modes(self, monkeypatch):
        import torch.nn as nn
        from comfy.distributed import patcher
        from comfy.distributed import parallel_linear as pl_module
        from comfy.distributed.parallel_linear import ParallelLinear
        from comfy.distributed.patcher import TP_HEAD_SPLIT_MODELS

        class FakeMesh:
            world_size = 2
            rank = 0
            current_device = "cpu"
        monkeypatch.setattr(patcher, "get_mesh", lambda: FakeMesh())
        monkeypatch.setattr(pl_module, "get_mesh", lambda: FakeMesh())

        dim, heads, head_dim = 64, 8, 8
        assert "CogVideoXTransformer3DModel" in TP_HEAD_SPLIT_MODELS

        class CogVideoXBlock(nn.Module):
            def __init__(self):
                super().__init__()
                self.num_heads = heads
                self.head_dim = head_dim
                self.q = nn.Linear(dim, dim)
                self.k = nn.Linear(dim, dim)
                self.v = nn.Linear(dim, dim)
                self.attn_out = nn.Linear(dim, dim)
                self.ff_proj = nn.Linear(dim, dim * 4)
                self.ff_out = nn.Linear(dim * 4, dim)
                self.norm1 = nn.Module()
                self.norm1.linear = nn.Linear(32, 6 * dim)
                self.norm2 = nn.Module()
                self.norm2.linear = nn.Linear(32, 6 * dim)

        class CogVideoXTransformer3DModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.num_attention_heads = heads
                self.blocks = nn.ModuleList([CogVideoXBlock()])
                self.proj_out = nn.Linear(dim, 16)

        model = CogVideoXTransformer3DModel()
        assert patcher.parallelize_model(model) is True
        block = model.blocks[0]
        assert isinstance(block.q, ParallelLinear)
        assert block.q.mode == "colwise"
        assert isinstance(block.k, ParallelLinear)
        assert isinstance(block.v, ParallelLinear)
        assert block.attn_out.mode == "rowwise"
        assert isinstance(block.ff_proj, ParallelLinear)
        assert block.ff_proj.mode == "colwise"
        assert block.ff_out.mode == "rowwise"
        assert block.num_heads == heads // 2
        assert isinstance(block.norm1.linear, nn.Linear)
        assert not isinstance(block.norm1.linear, ParallelLinear)
        assert isinstance(block.norm2.linear, nn.Linear)
        assert not isinstance(block.norm2.linear, ParallelLinear)
        assert isinstance(model.proj_out, nn.Linear)
        assert not isinstance(model.proj_out, ParallelLinear)
        assert model.num_attention_heads == heads


class TestNaDiTTP:
    """SeedVR2 NaDiT: packed MMModule QKV (vid/txt), MLP proj_in colwise, ada is Parameter."""

    def test_packed_qkv_mmmodule_and_mlp(self, monkeypatch):
        import torch.nn as nn
        from comfy.distributed import patcher
        from comfy.distributed import parallel_linear as pl_module
        from comfy.distributed.parallel_linear import ParallelLinear
        from comfy.distributed.patcher import TP_HEAD_SPLIT_MODELS

        class FakeMesh:
            world_size = 2
            rank = 0
            current_device = "cpu"
        monkeypatch.setattr(patcher, "get_mesh", lambda: FakeMesh())
        monkeypatch.setattr(pl_module, "get_mesh", lambda: FakeMesh())

        dim, heads, head_dim = 64, 8, 8
        assert "NaDiT" in TP_HEAD_SPLIT_MODELS

        class MMLinear(nn.Module):
            def __init__(self, in_f, out_f):
                super().__init__()
                self.vid = nn.Linear(in_f, out_f)
                self.txt = nn.Linear(in_f, out_f)

        class MLP(nn.Module):
            def __init__(self):
                super().__init__()
                self.proj_in = nn.Linear(dim, dim * 4)
                self.proj_out = nn.Linear(dim * 4, dim)

        class MMMLP(nn.Module):
            def __init__(self):
                super().__init__()
                self.vid = MLP()
                self.txt = MLP()

        class NaMMAttention(nn.Module):
            def __init__(self):
                super().__init__()
                self.heads = heads
                self.head_dim = head_dim
                self.proj_qkv = MMLinear(dim, dim * 3)
                self.proj_out = MMLinear(dim, dim)

        class Block(nn.Module):
            def __init__(self):
                super().__init__()
                self.attn = NaMMAttention()
                self.mlp = MMMLP()

        class NaDiT(nn.Module):
            def __init__(self):
                super().__init__()
                self.blocks = nn.ModuleList([Block()])
                self.txt_in = nn.Linear(32, dim)

        model = NaDiT()
        assert patcher.parallelize_model(model) is True
        attn = model.blocks[0].attn
        assert isinstance(attn.proj_qkv.vid, ParallelLinear)
        assert attn.proj_qkv.vid.mode == "colwise"
        assert attn.proj_qkv.vid.pack_count == 3
        assert isinstance(attn.proj_qkv.txt, ParallelLinear)
        assert attn.proj_qkv.txt.pack_count == 3
        assert attn.proj_out.vid.mode == "rowwise"
        assert attn.heads == heads // 2
        mlp = model.blocks[0].mlp.vid
        assert isinstance(mlp.proj_in, ParallelLinear)
        assert mlp.proj_in.mode == "colwise"
        assert mlp.proj_out.mode == "rowwise"
        assert isinstance(model.txt_in, nn.Linear)
        assert not isinstance(model.txt_in, ParallelLinear)


class TestAuraFlowTP:
    """AuraFlow MMDiT: unfused w1q/w2q. w2* must not hit Megatron w2 rowwise; c_fc2 is colwise."""

    def test_double_single_and_swiglu_modes(self, monkeypatch):
        import torch.nn as nn
        from comfy.distributed import patcher
        from comfy.distributed import parallel_linear as pl_module
        from comfy.distributed.parallel_linear import ParallelLinear
        from comfy.distributed.patcher import TP_HEAD_SPLIT_MODELS

        class FakeMesh:
            world_size = 2
            rank = 0
            current_device = "cpu"
        monkeypatch.setattr(patcher, "get_mesh", lambda: FakeMesh())
        monkeypatch.setattr(pl_module, "get_mesh", lambda: FakeMesh())

        dim, heads = 64, 8
        assert "MMDiT" in TP_HEAD_SPLIT_MODELS

        class DoubleAttention(nn.Module):
            def __init__(self):
                super().__init__()
                self.n_heads = heads
                self.head_dim = dim // heads
                self.w1q = nn.Linear(dim, dim, bias=False)
                self.w1k = nn.Linear(dim, dim, bias=False)
                self.w1v = nn.Linear(dim, dim, bias=False)
                self.w1o = nn.Linear(dim, dim, bias=False)
                self.w2q = nn.Linear(dim, dim, bias=False)
                self.w2k = nn.Linear(dim, dim, bias=False)
                self.w2v = nn.Linear(dim, dim, bias=False)
                self.w2o = nn.Linear(dim, dim, bias=False)

        class SingleAttention(nn.Module):
            def __init__(self):
                super().__init__()
                self.n_heads = heads
                self.head_dim = dim // heads
                self.w1q = nn.Linear(dim, dim, bias=False)
                self.w1k = nn.Linear(dim, dim, bias=False)
                self.w1v = nn.Linear(dim, dim, bias=False)
                self.w1o = nn.Linear(dim, dim, bias=False)

        class MLP(nn.Module):
            def __init__(self):
                super().__init__()
                self.c_fc1 = nn.Linear(dim, dim * 2, bias=False)
                self.c_fc2 = nn.Linear(dim, dim * 2, bias=False)
                self.c_proj = nn.Linear(dim * 2, dim, bias=False)

        class MMDiTBlock(nn.Module):
            def __init__(self):
                super().__init__()
                self.attn = DoubleAttention()
                self.mlpX = MLP()
                self.mlpC = MLP()
                self.modC = nn.Sequential(nn.SiLU(), nn.Linear(dim, 6 * dim, bias=False))
                self.modX = nn.Sequential(nn.SiLU(), nn.Linear(dim, 6 * dim, bias=False))

        class DiTBlock(nn.Module):
            def __init__(self):
                super().__init__()
                self.attn = SingleAttention()
                self.mlp = MLP()
                self.modCX = nn.Sequential(nn.SiLU(), nn.Linear(dim, 6 * dim, bias=False))

        class MMDiT(nn.Module):
            def __init__(self):
                super().__init__()
                self.double_layers = nn.ModuleList([MMDiTBlock()])
                self.single_layers = nn.ModuleList([DiTBlock()])
                self.init_x_linear = nn.Linear(16, dim)
                self.final_linear = nn.Linear(dim, 16, bias=False)

        model = MMDiT()
        assert patcher.parallelize_model(model) is True
        dattn = model.double_layers[0].attn
        assert isinstance(dattn.w1q, ParallelLinear)
        assert dattn.w1q.mode == "colwise"
        assert dattn.w1o.mode == "rowwise"
        assert isinstance(dattn.w2q, ParallelLinear)
        assert dattn.w2q.mode == "colwise"
        assert dattn.w2o.mode == "rowwise"
        assert dattn.n_heads == heads // 2
        mlp = model.double_layers[0].mlpX
        assert mlp.c_fc1.mode == "colwise"
        assert mlp.c_fc2.mode == "colwise"
        assert mlp.c_proj.mode == "rowwise"
        assert isinstance(model.double_layers[0].modC[1], nn.Linear)
        assert not isinstance(model.double_layers[0].modC[1], ParallelLinear)
        sattn = model.single_layers[0].attn
        assert isinstance(sattn.w1q, ParallelLinear)
        assert sattn.w1o.mode == "rowwise"
        assert sattn.n_heads == heads // 2
        assert isinstance(model.init_x_linear, nn.Linear)
        assert not isinstance(model.init_x_linear, ParallelLinear)


class TestHunyuan3Dv2TP:
    """Hunyuan3Dv2 reuses Flux DoubleStreamBlock; packed QKV, fused linear1 stays."""

    def test_packed_qkv_and_head_split(self, monkeypatch):
        import torch.nn as nn
        from comfy.distributed import patcher
        from comfy.distributed import parallel_linear as pl_module
        from comfy.distributed.parallel_linear import ParallelLinear
        from comfy.distributed.patcher import TP_HEAD_SPLIT_MODELS

        class FakeMesh:
            world_size = 2
            rank = 0
            current_device = "cpu"
        monkeypatch.setattr(patcher, "get_mesh", lambda: FakeMesh())
        monkeypatch.setattr(pl_module, "get_mesh", lambda: FakeMesh())

        dim = 128
        assert "Hunyuan3Dv2" in TP_HEAD_SPLIT_MODELS

        class SelfAttention(nn.Module):
            def __init__(self):
                super().__init__()
                self.num_heads = 8
                self.qkv = nn.Linear(dim, dim * 3, bias=False)
                self.proj = nn.Linear(dim, dim)

        class DoubleStreamBlock(nn.Module):
            def __init__(self):
                super().__init__()
                self.num_heads = 8
                self.img_attn = SelfAttention()
                self.txt_attn = SelfAttention()
                self.img_mlp = nn.Sequential(nn.Linear(dim, dim * 4), nn.GELU(), nn.Linear(dim * 4, dim))
                self.txt_mlp = nn.Sequential(nn.Linear(dim, dim * 4), nn.GELU(), nn.Linear(dim * 4, dim))

        class SingleStreamBlock(nn.Module):
            def __init__(self):
                super().__init__()
                self.num_heads = 8
                self.linear1 = nn.Linear(dim, dim * 3 + dim * 4)
                self.linear2 = nn.Linear(dim + dim * 4, dim)

        class Hunyuan3Dv2(nn.Module):
            def __init__(self):
                super().__init__()
                self.num_heads = 8
                self.latent_in = nn.Linear(64, dim)
                self.double_blocks = nn.ModuleList([DoubleStreamBlock()])
                self.single_blocks = nn.ModuleList([SingleStreamBlock()])
                self.final_layer = nn.Linear(dim, 64)

        model = Hunyuan3Dv2()
        assert patcher.parallelize_model(model) is True
        qkv = model.double_blocks[0].img_attn.qkv
        assert isinstance(qkv, ParallelLinear)
        assert qkv.mode == "colwise"
        assert qkv.pack_count == 3
        assert qkv.local_out_features == 3 * (dim // 2)
        assert isinstance(model.double_blocks[0].img_attn.proj, ParallelLinear)
        assert model.double_blocks[0].img_attn.proj.mode == "rowwise"
        txt_attn = model.double_blocks[0].txt_attn
        assert isinstance(txt_attn.qkv, ParallelLinear)
        assert txt_attn.qkv.pack_count == 3
        assert isinstance(model.single_blocks[0].linear1, nn.Linear)
        assert isinstance(model.double_blocks[0].img_mlp[0], ParallelLinear)
        assert model.double_blocks[0].img_mlp[0].mode == "colwise"
        assert isinstance(model.double_blocks[0].img_mlp[2], ParallelLinear)
        assert model.double_blocks[0].img_mlp[2].mode == "rowwise"
        assert isinstance(model.latent_in, nn.Linear)
        assert not isinstance(model.latent_in, ParallelLinear)
        assert isinstance(model.final_layer, nn.Linear)
        assert not isinstance(model.final_layer, ParallelLinear)
        assert model.double_blocks[0].img_attn.num_heads == 4
        assert model.double_blocks[0].num_heads == 4
        assert model.single_blocks[0].num_heads == 8
        assert model.num_heads == 8


class TestKandinsky5TP:
    """Kandinsky5: unfused to_query. Attn/FF out_layer rowwise; modulation.out_layer stays."""

    def test_unfused_qkv_and_modulation_stays(self, monkeypatch):
        import torch.nn as nn
        from comfy.distributed import patcher
        from comfy.distributed import parallel_linear as pl_module
        from comfy.distributed.parallel_linear import ParallelLinear
        from comfy.distributed.patcher import TP_HEAD_SPLIT_MODELS

        class FakeMesh:
            world_size = 2
            rank = 0
            current_device = "cpu"
        monkeypatch.setattr(patcher, "get_mesh", lambda: FakeMesh())
        monkeypatch.setattr(pl_module, "get_mesh", lambda: FakeMesh())

        dim, heads, head_dim, ffn = 128, 8, 16, 256
        assert "Kandinsky5" in TP_HEAD_SPLIT_MODELS

        class SelfAttention(nn.Module):
            def __init__(self):
                super().__init__()
                self.num_heads = heads
                self.head_dim = head_dim
                self.to_query = nn.Linear(dim, dim)
                self.to_key = nn.Linear(dim, dim)
                self.to_value = nn.Linear(dim, dim)
                self.query_norm = nn.RMSNorm(head_dim)
                self.key_norm = nn.RMSNorm(head_dim)
                self.out_layer = nn.Linear(dim, dim)

        class FeedForward(nn.Module):
            def __init__(self):
                super().__init__()
                self.in_layer = nn.Linear(dim, ffn, bias=False)
                self.out_layer = nn.Linear(ffn, dim, bias=False)

        class Modulation(nn.Module):
            def __init__(self, num_params):
                super().__init__()
                self.out_layer = nn.Linear(64, num_params * dim)

        class TransformerEncoderBlock(nn.Module):
            def __init__(self):
                super().__init__()
                self.text_modulation = Modulation(6)
                self.self_attention = SelfAttention()
                self.feed_forward = FeedForward()

        class TransformerDecoderBlock(nn.Module):
            def __init__(self):
                super().__init__()
                self.visual_modulation = Modulation(9)
                self.self_attention = SelfAttention()
                self.cross_attention = SelfAttention()
                self.feed_forward = FeedForward()

        class Kandinsky5(nn.Module):
            def __init__(self):
                super().__init__()
                self.time_embeddings = nn.Linear(dim, 64)
                self.text_transformer_blocks = nn.ModuleList([TransformerEncoderBlock()])
                self.visual_transformer_blocks = nn.ModuleList([TransformerDecoderBlock()])
                self.out_layer = nn.Linear(dim, 16)

        model = Kandinsky5()
        assert patcher.parallelize_model(model) is True
        enc = model.text_transformer_blocks[0]
        attn = enc.self_attention
        assert isinstance(attn.to_query, ParallelLinear)
        assert attn.to_query.mode == "colwise"
        assert attn.to_key.mode == "colwise"
        assert attn.to_value.mode == "colwise"
        assert isinstance(attn.out_layer, ParallelLinear)
        assert attn.out_layer.mode == "rowwise"
        assert attn.num_heads == heads // 2
        assert tuple(attn.query_norm.weight.shape) == (head_dim,)
        ff = enc.feed_forward
        assert isinstance(ff.in_layer, ParallelLinear)
        assert ff.in_layer.mode == "colwise"
        assert isinstance(ff.out_layer, ParallelLinear)
        assert ff.out_layer.mode == "rowwise"
        assert isinstance(enc.text_modulation.out_layer, nn.Linear)
        assert not isinstance(enc.text_modulation.out_layer, ParallelLinear)
        dec = model.visual_transformer_blocks[0]
        assert isinstance(dec.self_attention.to_query, ParallelLinear)
        assert dec.self_attention.out_layer.mode == "rowwise"
        assert isinstance(dec.cross_attention.to_query, ParallelLinear)
        assert dec.cross_attention.out_layer.mode == "rowwise"
        assert isinstance(dec.visual_modulation.out_layer, nn.Linear)
        assert not isinstance(dec.visual_modulation.out_layer, ParallelLinear)
        assert isinstance(model.time_embeddings, nn.Linear)
        assert not isinstance(model.time_embeddings, ParallelLinear)
        assert isinstance(model.out_layer, nn.Linear)
        assert not isinstance(model.out_layer, ParallelLinear)


class TestErnieImageTP:
    """Ernie Image: unfused to_q; SwiGLU gate/up colwise; linear_fc2 rowwise; root adaLN stays."""

    def test_unfused_qkv_and_swiglu(self, monkeypatch):
        import torch.nn as nn
        from comfy.distributed import patcher
        from comfy.distributed import parallel_linear as pl_module
        from comfy.distributed.parallel_linear import ParallelLinear
        from comfy.distributed.patcher import TP_HEAD_SPLIT_MODELS

        class FakeMesh:
            world_size = 2
            rank = 0
            current_device = "cpu"
        monkeypatch.setattr(patcher, "get_mesh", lambda: FakeMesh())
        monkeypatch.setattr(pl_module, "get_mesh", lambda: FakeMesh())

        dim, heads, head_dim, ffn = 128, 8, 16, 256
        assert "ErnieImageModel" in TP_HEAD_SPLIT_MODELS

        class Attention(nn.Module):
            def __init__(self):
                super().__init__()
                self.heads = heads
                self.head_dim = head_dim
                self.to_q = nn.Linear(dim, dim, bias=False)
                self.to_k = nn.Linear(dim, dim, bias=False)
                self.to_v = nn.Linear(dim, dim, bias=False)
                self.norm_q = nn.RMSNorm(head_dim)
                self.norm_k = nn.RMSNorm(head_dim)
                self.to_out = nn.ModuleList([nn.Linear(dim, dim, bias=False)])

        class FeedForward(nn.Module):
            def __init__(self):
                super().__init__()
                self.gate_proj = nn.Linear(dim, ffn, bias=False)
                self.up_proj = nn.Linear(dim, ffn, bias=False)
                self.linear_fc2 = nn.Linear(ffn, dim, bias=False)

        class Block(nn.Module):
            def __init__(self):
                super().__init__()
                self.self_attention = Attention()
                self.mlp = FeedForward()

        class ErnieImageModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.num_heads = heads
                self.x_embedder = nn.Linear(16, dim)
                self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(dim, 6 * dim))
                self.layers = nn.ModuleList([Block()])
                self.final_linear = nn.Linear(dim, 16)

        model = ErnieImageModel()
        assert patcher.parallelize_model(model) is True
        attn = model.layers[0].self_attention
        assert isinstance(attn.to_q, ParallelLinear)
        assert attn.to_q.mode == "colwise"
        assert attn.to_k.mode == "colwise"
        assert attn.to_v.mode == "colwise"
        assert isinstance(attn.to_out[0], ParallelLinear)
        assert attn.to_out[0].mode == "rowwise"
        assert attn.heads == heads // 2
        assert tuple(attn.norm_q.weight.shape) == (head_dim,)
        mlp = model.layers[0].mlp
        assert isinstance(mlp.gate_proj, ParallelLinear)
        assert mlp.gate_proj.mode == "colwise"
        assert mlp.up_proj.mode == "colwise"
        assert isinstance(mlp.linear_fc2, ParallelLinear)
        assert mlp.linear_fc2.mode == "rowwise"
        assert isinstance(model.adaLN_modulation[1], nn.Linear)
        assert not isinstance(model.adaLN_modulation[1], ParallelLinear)
        assert isinstance(model.x_embedder, nn.Linear)
        assert not isinstance(model.x_embedder, ParallelLinear)
        assert isinstance(model.final_linear, nn.Linear)
        assert not isinstance(model.final_linear, ParallelLinear)
        assert model.num_heads == heads


class TestHiDreamO1TP:
    """HiDreamO1: LLM GQA 32/8 + vision packed QKV. Embeddings / merger / patch embed stay."""

    def test_llm_gqa_and_vision_packed(self, monkeypatch):
        import torch.nn as nn
        from comfy.distributed import patcher
        from comfy.distributed import parallel_linear as pl_module
        from comfy.distributed.parallel_linear import ParallelLinear
        from comfy.distributed.patcher import TP_HEAD_SPLIT_MODELS, TP_UNSUPPORTED

        class FakeMesh:
            world_size = 2
            rank = 0
            current_device = "cpu"
        monkeypatch.setattr(patcher, "get_mesh", lambda: FakeMesh())
        monkeypatch.setattr(pl_module, "get_mesh", lambda: FakeMesh())

        hidden, q_heads, kv_heads, head_dim = 128, 8, 4, 16
        vis_dim, vis_heads = 64, 8
        assert "HiDreamO1Transformer" in TP_HEAD_SPLIT_MODELS
        assert "HiDreamO1Transformer" not in TP_UNSUPPORTED

        class Attention(nn.Module):
            def __init__(self):
                super().__init__()
                self.num_heads = q_heads
                self.num_kv_heads = kv_heads
                self.head_dim = head_dim
                self.q_proj = nn.Linear(hidden, q_heads * head_dim, bias=False)
                self.k_proj = nn.Linear(hidden, kv_heads * head_dim, bias=False)
                self.v_proj = nn.Linear(hidden, kv_heads * head_dim, bias=False)
                self.o_proj = nn.Linear(q_heads * head_dim, hidden, bias=False)
                self.q_norm = nn.RMSNorm(head_dim)
                self.k_norm = nn.RMSNorm(head_dim)

        class MLP(nn.Module):
            def __init__(self):
                super().__init__()
                self.gate_proj = nn.Linear(hidden, hidden * 2, bias=False)
                self.up_proj = nn.Linear(hidden, hidden * 2, bias=False)
                self.down_proj = nn.Linear(hidden * 2, hidden, bias=False)

        class TransformerBlock(nn.Module):
            def __init__(self):
                super().__init__()
                self.self_attn = Attention()
                self.mlp = MLP()

        class Llama2_(nn.Module):
            def __init__(self):
                super().__init__()
                self.embed_tokens = nn.Embedding(32, hidden)
                self.layers = nn.ModuleList([TransformerBlock()])
                self.norm = nn.RMSNorm(hidden)

        class VisionAttention(nn.Module):
            def __init__(self):
                super().__init__()
                self.num_heads = vis_heads
                self.head_dim = vis_dim // vis_heads
                self.qkv = nn.Linear(vis_dim, vis_dim * 3, bias=True)
                self.proj = nn.Linear(vis_dim, vis_dim)

        class VisionBlock(nn.Module):
            def __init__(self):
                super().__init__()
                self.attn = VisionAttention()
                self.mlp = nn.Module()
                self.mlp.linear_fc1 = nn.Linear(vis_dim, vis_dim * 2)
                self.mlp.linear_fc2 = nn.Linear(vis_dim * 2, vis_dim)

        class Qwen35VisionModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.num_heads = vis_heads
                self.patch_embed = nn.Linear(16, vis_dim)
                self.blocks = nn.ModuleList([VisionBlock()])
                self.merger = nn.Linear(vis_dim, hidden)

        class BottleneckPatchEmbed(nn.Module):
            def __init__(self):
                super().__init__()
                self.proj1 = nn.Linear(32, hidden // 4, bias=False)
                self.proj2 = nn.Linear(hidden // 4, hidden)

        class HiDreamO1Transformer(nn.Module):
            def __init__(self):
                super().__init__()
                self.visual = Qwen35VisionModel()
                self.language_model = Llama2_()
                self.x_embedder = BottleneckPatchEmbed()
                self.final_layer2 = nn.Linear(hidden, 32)

        model = HiDreamO1Transformer()
        assert patcher.parallelize_model(model) is True
        attn = model.language_model.layers[0].self_attn
        assert isinstance(attn.q_proj, ParallelLinear)
        assert attn.q_proj.mode == "colwise"
        assert attn.k_proj.mode == "colwise"
        assert attn.v_proj.mode == "colwise"
        assert attn.o_proj.mode == "rowwise"
        assert attn.num_heads == q_heads // 2
        assert attn.num_kv_heads == kv_heads // 2
        assert tuple(attn.q_norm.weight.shape) == (head_dim,)
        mlp = model.language_model.layers[0].mlp
        assert mlp.gate_proj.mode == "colwise"
        assert mlp.up_proj.mode == "colwise"
        assert mlp.down_proj.mode == "rowwise"
        assert not isinstance(model.language_model.embed_tokens, ParallelLinear)
        vattn = model.visual.blocks[0].attn
        assert isinstance(vattn.qkv, ParallelLinear)
        assert vattn.qkv.mode == "colwise"
        assert vattn.qkv.pack_count == 3
        assert vattn.qkv.local_out_features == 3 * (vis_dim // 2)
        assert vattn.proj.mode == "rowwise"
        assert vattn.num_heads == vis_heads // 2
        assert model.visual.blocks[0].mlp.linear_fc1.mode == "colwise"
        assert model.visual.blocks[0].mlp.linear_fc2.mode == "rowwise"
        assert isinstance(model.visual.patch_embed, nn.Linear)
        assert not isinstance(model.visual.patch_embed, ParallelLinear)
        assert isinstance(model.visual.merger, nn.Linear)
        assert not isinstance(model.visual.merger, ParallelLinear)
        assert model.visual.num_heads == vis_heads
        assert isinstance(model.x_embedder.proj1, nn.Linear)
        assert not isinstance(model.x_embedder.proj1, ParallelLinear)
        assert isinstance(model.final_layer2, nn.Linear)
        assert not isinstance(model.final_layer2, ParallelLinear)


class TestTripoSplatTP:
    """TripoSplat: packed attn.qkv; attn.out rowwise; MultiHeadRMSNorm/RePo heads follow shard."""

    def test_packed_qkv_repo_and_mh_rmsnorm(self, monkeypatch):
        import torch
        import torch.nn as nn
        from comfy.distributed import patcher
        from comfy.distributed import parallel_linear as pl_module
        from comfy.distributed.parallel_linear import ParallelLinear
        from comfy.distributed.patcher import TP_HEAD_SPLIT_MODELS

        class FakeMesh:
            world_size = 2
            rank = 0
            current_device = "cpu"
        monkeypatch.setattr(patcher, "get_mesh", lambda: FakeMesh())
        monkeypatch.setattr(pl_module, "get_mesh", lambda: FakeMesh())

        dim, heads, head_dim = 128, 8, 16
        assert "LatentSeqMMFlowModel" in TP_HEAD_SPLIT_MODELS

        class MultiHeadRMSNorm(nn.Module):
            def __init__(self):
                super().__init__()
                self.gamma = nn.Parameter(torch.ones(heads, head_dim))

        class RopeMultiHeadAttention(nn.Module):
            def __init__(self):
                super().__init__()
                self.num_heads = heads
                self.head_dim = head_dim
                self.qkv = nn.Linear(dim, dim * 3, bias=False)
                self.q_norm = MultiHeadRMSNorm()
                self.k_norm = MultiHeadRMSNorm()
                self.out = nn.Linear(dim, dim)

        class MLP(nn.Module):
            def __init__(self):
                super().__init__()
                self.mlp = nn.Sequential(
                    nn.Linear(dim, dim * 4),
                    nn.GELU(),
                    nn.Linear(dim * 4, dim),
                )

        class UnifiedTransformerBlock(nn.Module):
            def __init__(self, modulation=True):
                super().__init__()
                self.attn = RopeMultiHeadAttention()
                self.mlp = MLP()
                if modulation:
                    self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(dim, 6 * dim))

        class RePo3DRotaryEmbedding(nn.Module):
            def __init__(self):
                super().__init__()
                self.num_heads = heads
                self.head_dim = head_dim
                self.gate_map = nn.Linear(dim, dim // 8, bias=False)
                self.content_map = nn.Linear(dim, dim // 8, bias=False)
                self.final_map = nn.Linear(dim // 8, 3 * heads, bias=False)

        class LatentSeqMMFlowModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.num_heads = heads
                self.input_layer = nn.Linear(16, dim)
                self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(dim, 6 * dim))
                self.noise_refiner = nn.ModuleList([UnifiedTransformerBlock()])
                self.context_refiner = nn.ModuleList([UnifiedTransformerBlock(modulation=False)])
                self.blocks = nn.ModuleList([UnifiedTransformerBlock()])
                self.noise_repo_layers = nn.ModuleList([RePo3DRotaryEmbedding()])
                self.context_repo_layers = nn.ModuleList([RePo3DRotaryEmbedding()])
                self.repo_layers = nn.ModuleList([RePo3DRotaryEmbedding()])
                self.out_layer = nn.Linear(dim, 16)

        model = LatentSeqMMFlowModel()
        assert patcher.parallelize_model(model) is True
        attn = model.blocks[0].attn
        assert isinstance(attn.qkv, ParallelLinear)
        assert attn.qkv.mode == "colwise"
        assert attn.qkv.pack_count == 3
        assert attn.qkv.local_out_features == 3 * (dim // 2)
        assert isinstance(attn.out, ParallelLinear)
        assert attn.out.mode == "rowwise"
        assert attn.num_heads == heads // 2
        assert tuple(attn.q_norm.gamma.shape) == (heads // 2, head_dim)
        assert tuple(attn.k_norm.gamma.shape) == (heads // 2, head_dim)
        mlp = model.blocks[0].mlp.mlp
        assert isinstance(mlp[0], ParallelLinear)
        assert mlp[0].mode == "colwise"
        assert isinstance(mlp[2], ParallelLinear)
        assert mlp[2].mode == "rowwise"
        assert isinstance(model.blocks[0].adaLN_modulation[1], nn.Linear)
        assert not isinstance(model.blocks[0].adaLN_modulation[1], ParallelLinear)
        repo = model.repo_layers[0]
        assert isinstance(repo.gate_map, nn.Linear)
        assert not isinstance(repo.gate_map, ParallelLinear)
        assert isinstance(repo.content_map, nn.Linear)
        assert not isinstance(repo.content_map, ParallelLinear)
        assert isinstance(repo.final_map, ParallelLinear)
        assert repo.final_map.mode == "colwise"
        assert repo.final_map.pack_count == 1
        assert repo.final_map.local_out_features == 3 * (heads // 2)
        assert repo.num_heads == heads // 2
        assert model.noise_refiner[0].attn.qkv.pack_count == 3
        assert model.context_refiner[0].attn.out.mode == "rowwise"
        assert isinstance(model.input_layer, nn.Linear)
        assert not isinstance(model.input_layer, ParallelLinear)
        assert isinstance(model.out_layer, nn.Linear)
        assert not isinstance(model.out_layer, ParallelLinear)
        assert model.num_heads == heads

    def test_mh_rmsnorm_gamma_reloaded_from_full_sd(self, monkeypatch):
        import torch
        import torch.nn as nn
        from comfy.distributed import patcher
        from comfy.distributed import parallel_linear as pl_module

        class FakeMesh:
            world_size = 2
            rank = 0
            current_device = "cpu"
        monkeypatch.setattr(patcher, "get_mesh", lambda: FakeMesh())
        monkeypatch.setattr(pl_module, "get_mesh", lambda: FakeMesh())

        heads, head_dim, dim = 8, 16, 128

        class MultiHeadRMSNorm(nn.Module):
            def __init__(self):
                super().__init__()
                self.gamma = nn.Parameter(torch.arange(heads * head_dim, dtype=torch.float32).reshape(heads, head_dim))

        class RopeMultiHeadAttention(nn.Module):
            def __init__(self):
                super().__init__()
                self.num_heads = heads
                self.head_dim = head_dim
                self.qkv = nn.Linear(dim, dim * 3, bias=False)
                self.q_norm = MultiHeadRMSNorm()
                self.out = nn.Linear(dim, dim)

        class LatentSeqMMFlowModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.blocks = nn.ModuleList([nn.Module()])
                self.blocks[0].attn = RopeMultiHeadAttention()

        model = LatentSeqMMFlowModel()
        full_gamma = model.blocks[0].attn.q_norm.gamma.data.clone()
        assert patcher.parallelize_model(model) is True
        torch.testing.assert_close(model.blocks[0].attn.q_norm.gamma.data, full_gamma[: heads // 2])
        sd = {"blocks.0.attn.q_norm.gamma": full_gamma.clone()}
        patcher.load_tp_shards(model, sd)
        torch.testing.assert_close(model.blocks[0].attn.q_norm.gamma.data, full_gamma[: heads // 2])


class TestPackedColwise:
    """Fused QKV / SwiGLU: shard each pack, then concat — not a naive out-dim cut."""

    def test_qkv_pack_slice_rank0_and_rank1(self, monkeypatch):
        import torch
        from comfy.distributed import parallel_linear as pl_module

        pack, heads, head_dim, hidden, ws = 3, 8, 16, 32, 2
        inner = heads * head_dim
        full_out = pack * inner

        def run(rank_id):
            class FakeMesh:
                world_size = ws
                current_device = "cpu"
            FakeMesh.rank = rank_id

            monkeypatch.setattr(pl_module, "get_mesh", lambda: FakeMesh())
            layer = pl_module.ParallelLinear(hidden, full_out, bias=True, mode="colwise", pack_count=pack)
            assert layer.pack_count == pack
            assert layer.local_out_features == pack * (inner // ws)
            full_w = torch.arange(full_out * hidden, dtype=torch.float32).reshape(full_out, hidden)
            full_b = torch.arange(full_out, dtype=torch.float32)
            layer.load_shard(full_w)
            layer.load_bias_shard(full_b)
            local_inner = inner // ws
            expected = []
            for p in range(pack):
                start = p * inner + rank_id * local_inner
                expected.append(full_w[start:start + local_inner])
            torch.testing.assert_close(layer.weight.data, torch.cat(expected, dim=0))
            expected_b = []
            for p in range(pack):
                start = p * inner + rank_id * local_inner
                expected_b.append(full_b[start:start + local_inner])
            torch.testing.assert_close(layer.bias.data, torch.cat(expected_b, dim=0))

        run(0)
        run(1)

    def test_packed_colwise_forward_matches_per_pack_head_slice(self, monkeypatch):
        import torch
        from comfy.distributed import parallel_linear as pl_module

        pack, inner, hidden, ws, rank_id = 3, 8, 4, 2, 0
        full_out = pack * inner

        class FakeMesh:
            world_size = ws
            current_device = "cpu"
        FakeMesh.rank = rank_id
        monkeypatch.setattr(pl_module, "get_mesh", lambda: FakeMesh())

        torch.manual_seed(0)
        full_w = torch.randn(full_out, hidden)
        x = torch.randn(2, hidden)
        full_out_t = x @ full_w.t()
        layer = pl_module.ParallelLinear(hidden, full_out, bias=False, mode="colwise", pack_count=pack)
        layer.load_shard(full_w)
        got = layer(x)
        local = inner // ws
        expected = []
        for p in range(pack):
            start = p * inner + rank_id * local
            expected.append(full_out_t[..., start:start + local])
        torch.testing.assert_close(got, torch.cat(expected, dim=-1), atol=1e-5, rtol=1e-5)


class TestUnequalPackedColwise:
    """GQA fused QKV: packs are [Q heads | K kv | V kv], not equal thirds."""

    def test_gqa_pack_sizes_rank0_and_rank1(self, monkeypatch):
        import torch
        from comfy.distributed import parallel_linear as pl_module

        hidden, ws = 32, 2
        q_size, kv_size = 128, 64
        pack_sizes = (q_size, kv_size, kv_size)
        full_out = sum(pack_sizes)

        def run(rank_id):
            class FakeMesh:
                world_size = ws
                current_device = "cpu"
            FakeMesh.rank = rank_id
            monkeypatch.setattr(pl_module, "get_mesh", lambda: FakeMesh())
            layer = pl_module.ParallelLinear(
                hidden, full_out, bias=True, mode="colwise", pack_sizes=pack_sizes,
            )
            assert layer.pack_count == 3
            assert layer.pack_sizes == pack_sizes
            assert layer.local_out_features == (q_size + kv_size + kv_size) // ws
            full_w = torch.arange(full_out * hidden, dtype=torch.float32).reshape(full_out, hidden)
            layer.load_shard(full_w)
            expected = []
            offset = 0
            for size in pack_sizes:
                local = size // ws
                start = offset + rank_id * local
                expected.append(full_w[start:start + local])
                offset += size
            torch.testing.assert_close(layer.weight.data, torch.cat(expected, dim=0))
            naive = full_w[: layer.local_out_features]
            assert not torch.equal(layer.weight.data, naive)

        run(0)
        run(1)


class TestPackedColwiseLoRA:
    """LoRA diffs on packed QKV must follow load_shard, not a naive row cut."""

    def test_shard_like_tp_weight_packed_not_naive(self, monkeypatch):
        import torch
        from comfy.distributed import parallel_linear as pl_module

        pack, inner, hidden, ws, rank_id = 3, 8, 4, 2, 0
        full_out = pack * inner

        class FakeMesh:
            world_size = ws
            current_device = "cpu"
        FakeMesh.rank = rank_id
        monkeypatch.setattr(pl_module, "get_mesh", lambda: FakeMesh())

        layer = pl_module.ParallelLinear(hidden, full_out, bias=False, mode="colwise", pack_count=pack)
        full = torch.arange(full_out * hidden, dtype=torch.float32).reshape(full_out, hidden)
        layer.load_shard(full)
        packed = layer.weight.data.clone()
        naive = full[: packed.shape[0]]
        assert not torch.equal(packed, naive)

        from comfy.distributed.parallel_linear import shard_like_tp_weight, stamp_tp_shard_meta
        stamp_tp_shard_meta(layer.weight, layer)
        got = shard_like_tp_weight(full, layer.weight)
        torch.testing.assert_close(got, packed)

    def test_shard_like_tp_weight_gqa_pack_sizes(self, monkeypatch):
        import torch
        from comfy.distributed import parallel_linear as pl_module
        from comfy.distributed.parallel_linear import shard_like_tp_weight, stamp_tp_shard_meta

        hidden, ws, rank_id = 4, 2, 0
        pack_sizes = (128, 64, 64)
        full_out = sum(pack_sizes)

        class FakeMesh:
            world_size = ws
            current_device = "cpu"
        FakeMesh.rank = rank_id
        monkeypatch.setattr(pl_module, "get_mesh", lambda: FakeMesh())

        layer = pl_module.ParallelLinear(
            hidden, full_out, bias=False, mode="colwise", pack_sizes=pack_sizes,
        )
        full = torch.arange(full_out * hidden, dtype=torch.float32).reshape(full_out, hidden)
        layer.load_shard(full)
        packed = layer.weight.data.clone()
        naive = full[: packed.shape[0]]
        assert not torch.equal(packed, naive)
        stamp_tp_shard_meta(layer.weight, layer)
        got = shard_like_tp_weight(full, layer.weight)
        torch.testing.assert_close(got, packed)

    def test_lora_adapter_calculate_weight_uses_packed_slice(self, monkeypatch):
        import torch
        from comfy.distributed import parallel_linear as pl_module
        from comfy.weight_adapter.lora import LoRAAdapter

        pack, inner, hidden, ws, rank_id = 3, 8, 4, 2, 0
        full_out = pack * inner

        class FakeMesh:
            world_size = ws
            current_device = "cpu"
        FakeMesh.rank = rank_id
        monkeypatch.setattr(pl_module, "get_mesh", lambda: FakeMesh())
        monkeypatch.setattr("comfy.distributed.utils.is_tp_active", lambda: True)

        layer = pl_module.ParallelLinear(hidden, full_out, bias=False, mode="colwise", pack_count=pack)
        full = torch.arange(full_out * hidden, dtype=torch.float32).reshape(full_out, hidden)
        layer.load_shard(full)
        packed = layer.weight.data.clone()

        # mm(eye, full) == full LoRA diff, then TP-sliced onto a zero shard.
        up = torch.eye(full_out, dtype=torch.float32)
        adapter = LoRAAdapter(set(), (up, full, None, None, None, None))
        shard = torch.zeros_like(layer.weight.data)
        from comfy.distributed.parallel_linear import stamp_tp_shard_meta
        stamp_tp_shard_meta(shard, layer)
        out = adapter.calculate_weight(
            shard, "img_attn.qkv.weight", 1.0, 1.0, None, lambda x: x,
        )
        torch.testing.assert_close(out, packed)
        naive = full[: packed.shape[0]]
        assert not torch.equal(out, naive)

    def test_packed_colwise_activation_slice(self, monkeypatch):
        import torch
        from comfy.distributed import parallel_linear as pl_module
        from comfy.distributed.parallel_linear import slice_colwise_activation

        pack, inner, hidden, ws, rank_id = 3, 8, 4, 2, 0
        full_out = pack * inner

        class FakeMesh:
            world_size = ws
            current_device = "cpu"
        FakeMesh.rank = rank_id
        monkeypatch.setattr(pl_module, "get_mesh", lambda: FakeMesh())

        layer = pl_module.ParallelLinear(hidden, full_out, bias=False, mode="colwise", pack_count=pack)
        h_out = torch.arange(full_out, dtype=torch.float32).reshape(1, full_out)
        got = slice_colwise_activation(h_out, layer)
        local = inner // ws
        expected = []
        for p in range(pack):
            start = p * inner + rank_id * local
            expected.append(h_out[..., start:start + local])
        torch.testing.assert_close(got, torch.cat(expected, dim=-1))


class TestMiniMaxH3TP:
    """MiniMax H3: packed qkv_proj + packed SwiGLU fc1; adaLN stays; heads follow shard."""

    def test_packed_qkv_swiglu_and_head_split(self, monkeypatch):
        import torch.nn as nn
        from comfy.distributed import patcher
        from comfy.distributed import parallel_linear as pl_module
        from comfy.distributed.parallel_linear import ParallelLinear

        class FakeMesh:
            world_size = 2
            rank = 0
            current_device = "cpu"
        monkeypatch.setattr(patcher, "get_mesh", lambda: FakeMesh())
        monkeypatch.setattr(pl_module, "get_mesh", lambda: FakeMesh())

        dim, heads, head_dim, ffn = 128, 8, 16, 128
        inner = heads * head_dim

        class Attention(nn.Module):
            def __init__(self):
                super().__init__()
                self.heads = heads
                self.head_dim = head_dim
                self.qkv_proj = nn.Linear(dim, inner * 3, bias=False)
                self.out_proj = nn.Linear(inner, dim, bias=False)
                self.q_norm = nn.RMSNorm(head_dim)
                self.k_norm = nn.RMSNorm(head_dim)

        class MLP(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc1 = nn.Linear(dim, ffn * 2, bias=False)
                self.fc2 = nn.Linear(ffn, dim, bias=False)

        class AdalnProj(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(64, 6 * dim * 3)

        class DiTBlock(nn.Module):
            def __init__(self):
                super().__init__()
                self.attn = Attention()
                self.mlp = MLP()
                self.adaln_proj = AdalnProj()

        class MiniMaxH3Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.blocks = nn.ModuleList([DiTBlock()])

        model = MiniMaxH3Model()
        assert patcher.parallelize_model(model) is True
        attn = model.blocks[0].attn
        assert isinstance(attn.qkv_proj, ParallelLinear)
        assert attn.qkv_proj.mode == "colwise"
        assert attn.qkv_proj.pack_count == 3
        assert attn.qkv_proj.local_out_features == 3 * (inner // 2)
        assert attn.heads == heads // 2
        assert tuple(attn.q_norm.weight.shape) == (head_dim,)
        assert isinstance(attn.out_proj, ParallelLinear)
        assert attn.out_proj.mode == "rowwise"
        mlp = model.blocks[0].mlp
        assert isinstance(mlp.fc1, ParallelLinear)
        assert mlp.fc1.pack_count == 2
        assert mlp.fc1.local_out_features == 2 * (ffn // 2)
        assert isinstance(mlp.fc2, ParallelLinear)
        assert mlp.fc2.mode == "rowwise"
        assert isinstance(model.blocks[0].adaln_proj.linear, nn.Linear)


class TestParallelizeModelLoRA:
    """LoRA diffs must follow ParallelLinear shards after parallelize_model.

    Packed-colwise LoRA is already covered at ParallelLinear. These tests
    run the real patcher on unfused Qwen/Wan/Kandinsky/MiniTrainDIT/ACE 1.5,
    packed HiDreamO1 vision / NextDiT GQA / Ideogram / Flux / MiniMax H3 QKV,
    then apply LoRAAdapter.calculate_weight to the resulting shards — the
    path live LoRA e2e still needs.
    """

    def _mesh(self, monkeypatch, rank=0, world_size=2):
        from comfy.distributed import patcher
        from comfy.distributed import parallel_linear as pl_module

        class FakeMesh:
            current_device = "cpu"
        FakeMesh.world_size = world_size
        FakeMesh.rank = rank
        monkeypatch.setattr(patcher, "get_mesh", lambda: FakeMesh())
        monkeypatch.setattr(pl_module, "get_mesh", lambda: FakeMesh())
        # Production LoRA slicing keys off Parameter._tp_mode stamped by
        # parallelize_model, not a global is_tp_active monkeypatch.

    def _identity_lora(self, layer, full_weight, key):
        import torch
        from comfy.weight_adapter.lora import LoRAAdapter
        from comfy.distributed.parallel_linear import copy_tp_shard_meta, shard_like_tp_weight

        assert getattr(layer.weight, "_tp_mode", None) == layer.mode
        full_out = full_weight.shape[0]
        up = torch.eye(full_out, dtype=torch.float32)
        adapter = LoRAAdapter(set(), (up, full_weight.detach().float().cpu(), None, None, None, None))
        shard = torch.zeros_like(layer.weight.data)
        copy_tp_shard_meta(layer.weight, shard)
        got = adapter.calculate_weight(
            shard, key, 1.0, 1.0, None, lambda x: x,
        )
        expected = shard_like_tp_weight(full_weight.detach().float().cpu(), layer.weight)
        return got, expected

    def test_qwen_unfused_to_q_lora_rank1(self, monkeypatch):
        import torch
        import torch.nn as nn
        from comfy.distributed import patcher
        from comfy.distributed.parallel_linear import ParallelLinear

        self._mesh(monkeypatch, rank=1)
        dim = 64

        class QwenBlock(nn.Module):
            def __init__(self):
                super().__init__()
                self.attn = nn.Module()
                self.attn.to_q = nn.Linear(dim, dim, bias=False)
                self.attn.to_k = nn.Linear(dim, dim, bias=False)
                self.attn.to_v = nn.Linear(dim, dim, bias=False)
                self.attn.to_out = nn.ModuleList([nn.Linear(dim, dim, bias=False)])

        class QwenImageTransformer2DModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.transformer_blocks = nn.ModuleList([QwenBlock()])

        model = QwenImageTransformer2DModel()
        full_q = model.transformer_blocks[0].attn.to_q.weight.data.clone()
        assert patcher.parallelize_model(model) is True
        layer = model.transformer_blocks[0].attn.to_q
        assert isinstance(layer, ParallelLinear)
        assert layer.mode == "colwise"
        got, expected = self._identity_lora(layer, full_q, "transformer_blocks.0.attn.to_q.weight")
        torch.testing.assert_close(got, expected)
        naive = full_q[: layer.weight.shape[0]]
        assert not torch.equal(got, naive)

    def test_wan_unfused_q_lora_rank1(self, monkeypatch):
        import torch
        import torch.nn as nn
        from comfy.distributed import patcher
        from comfy.distributed.parallel_linear import ParallelLinear

        self._mesh(monkeypatch, rank=1)
        dim, n_heads, head_dim = 64, 8, 8

        class SelfAttn(nn.Module):
            def __init__(self):
                super().__init__()
                self.num_heads = n_heads
                self.head_dim = head_dim
                self.q = nn.Linear(dim, dim, bias=False)
                self.k = nn.Linear(dim, dim, bias=False)
                self.v = nn.Linear(dim, dim, bias=False)
                self.o = nn.Linear(dim, dim, bias=False)

        class Block(nn.Module):
            def __init__(self):
                super().__init__()
                self.self_attn = SelfAttn()
                self.ffn = nn.ModuleList([
                    nn.Linear(dim, dim * 4, bias=False),
                    nn.GELU(),
                    nn.Linear(dim * 4, dim, bias=False),
                ])

        class WanModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.blocks = nn.ModuleList([Block()])

        model = WanModel()
        full_q = model.blocks[0].self_attn.q.weight.data.clone()
        assert patcher.parallelize_model(model) is True
        layer = model.blocks[0].self_attn.q
        assert isinstance(layer, ParallelLinear)
        got, expected = self._identity_lora(layer, full_q, "blocks.0.self_attn.q.weight")
        torch.testing.assert_close(got, expected)
        naive = full_q[: layer.weight.shape[0]]
        assert not torch.equal(got, naive)

    def test_hidream_o1_vision_packed_qkv_lora(self, monkeypatch):
        import torch
        import torch.nn as nn
        from comfy.distributed import patcher
        from comfy.distributed.parallel_linear import ParallelLinear

        self._mesh(monkeypatch, rank=0)
        hidden, q_heads, kv_heads, head_dim = 64, 8, 4, 8
        vis_dim, vis_heads = 64, 8

        class Attention(nn.Module):
            def __init__(self):
                super().__init__()
                self.num_heads = q_heads
                self.num_kv_heads = kv_heads
                self.head_dim = head_dim
                self.q_proj = nn.Linear(hidden, q_heads * head_dim, bias=False)
                self.k_proj = nn.Linear(hidden, kv_heads * head_dim, bias=False)
                self.v_proj = nn.Linear(hidden, kv_heads * head_dim, bias=False)
                self.o_proj = nn.Linear(q_heads * head_dim, hidden, bias=False)

        class TransformerBlock(nn.Module):
            def __init__(self):
                super().__init__()
                self.self_attn = Attention()

        class Llama2_(nn.Module):
            def __init__(self):
                super().__init__()
                self.layers = nn.ModuleList([TransformerBlock()])

        class VisionAttention(nn.Module):
            def __init__(self):
                super().__init__()
                self.num_heads = vis_heads
                self.head_dim = vis_dim // vis_heads
                self.qkv = nn.Linear(vis_dim, vis_dim * 3, bias=False)
                self.proj = nn.Linear(vis_dim, vis_dim, bias=False)

        class VisionBlock(nn.Module):
            def __init__(self):
                super().__init__()
                self.attn = VisionAttention()

        class Qwen35VisionModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.blocks = nn.ModuleList([VisionBlock()])

        class HiDreamO1Transformer(nn.Module):
            def __init__(self):
                super().__init__()
                self.visual = Qwen35VisionModel()
                self.language_model = Llama2_()

        model = HiDreamO1Transformer()
        full_qkv = model.visual.blocks[0].attn.qkv.weight.data.clone()
        assert patcher.parallelize_model(model) is True
        layer = model.visual.blocks[0].attn.qkv
        assert isinstance(layer, ParallelLinear)
        assert layer.pack_count == 3
        assert getattr(layer.weight, "_tp_pack_count", None) == 3
        got, expected = self._identity_lora(layer, full_qkv, "visual.blocks.0.attn.qkv.weight")
        torch.testing.assert_close(got, expected)
        naive = full_qkv[: layer.weight.shape[0]]
        assert not torch.equal(got, naive)

    def test_kandinsky_to_query_lora_rank1(self, monkeypatch):
        import torch
        import torch.nn as nn
        from comfy.distributed import patcher
        from comfy.distributed.parallel_linear import ParallelLinear

        self._mesh(monkeypatch, rank=1)
        dim, heads, head_dim = 64, 8, 8

        class SelfAttention(nn.Module):
            def __init__(self):
                super().__init__()
                self.num_heads = heads
                self.head_dim = head_dim
                self.to_query = nn.Linear(dim, dim, bias=False)
                self.to_key = nn.Linear(dim, dim, bias=False)
                self.to_value = nn.Linear(dim, dim, bias=False)
                self.out_layer = nn.Linear(dim, dim, bias=False)

        class TransformerEncoderBlock(nn.Module):
            def __init__(self):
                super().__init__()
                self.self_attention = SelfAttention()

        class Kandinsky5(nn.Module):
            def __init__(self):
                super().__init__()
                self.text_transformer_blocks = nn.ModuleList([TransformerEncoderBlock()])
                self.visual_transformer_blocks = nn.ModuleList([TransformerEncoderBlock()])

        model = Kandinsky5()
        full_q = model.text_transformer_blocks[0].self_attention.to_query.weight.data.clone()
        assert patcher.parallelize_model(model) is True
        layer = model.text_transformer_blocks[0].self_attention.to_query
        assert isinstance(layer, ParallelLinear)
        got, expected = self._identity_lora(
            layer, full_q, "text_transformer_blocks.0.self_attention.to_query.weight",
        )
        torch.testing.assert_close(got, expected)
        naive = full_q[: layer.weight.shape[0]]
        assert not torch.equal(got, naive)

    def test_minitraindit_q_proj_lora_rank1(self, monkeypatch):
        import torch
        import torch.nn as nn
        from comfy.distributed import patcher
        from comfy.distributed.parallel_linear import ParallelLinear

        self._mesh(monkeypatch, rank=1)
        dim, heads, head_dim = 64, 8, 8

        class Attention(nn.Module):
            def __init__(self):
                super().__init__()
                self.n_heads = heads
                self.head_dim = head_dim
                self.q_proj = nn.Linear(dim, dim, bias=False)
                self.k_proj = nn.Linear(dim, dim, bias=False)
                self.v_proj = nn.Linear(dim, dim, bias=False)
                self.o_proj = nn.Linear(dim, dim, bias=False)

        class Block(nn.Module):
            def __init__(self):
                super().__init__()
                self.self_attn = Attention()
                self.mlp = nn.Module()
                self.mlp.layer1 = nn.Linear(dim, dim * 4, bias=False)
                self.mlp.layer2 = nn.Linear(dim * 4, dim, bias=False)

        class MiniTrainDIT(nn.Module):
            def __init__(self):
                super().__init__()
                self.blocks = nn.ModuleList([Block()])

        model = MiniTrainDIT()
        full_q = model.blocks[0].self_attn.q_proj.weight.data.clone()
        assert patcher.parallelize_model(model) is True
        layer = model.blocks[0].self_attn.q_proj
        assert isinstance(layer, ParallelLinear)
        got, expected = self._identity_lora(layer, full_q, "blocks.0.self_attn.q_proj.weight")
        torch.testing.assert_close(got, expected)
        naive = full_q[: layer.weight.shape[0]]
        assert not torch.equal(got, naive)

    def test_nextdit_packed_gqa_qkv_lora_rank1(self, monkeypatch):
        import torch
        import torch.nn as nn
        from comfy.distributed import patcher
        from comfy.distributed.parallel_linear import ParallelLinear

        self._mesh(monkeypatch, rank=1)
        dim, heads, kv_heads, head_dim = 128, 8, 4, 16
        qkv_out = (heads + kv_heads + kv_heads) * head_dim

        class JointAttention(nn.Module):
            def __init__(self):
                super().__init__()
                self.n_local_heads = heads
                self.n_local_kv_heads = kv_heads
                self.n_kv_heads = kv_heads
                self.head_dim = head_dim
                self.qkv = nn.Linear(dim, qkv_out, bias=False)
                self.out = nn.Linear(heads * head_dim, dim, bias=False)

        class JointTransformerBlock(nn.Module):
            def __init__(self):
                super().__init__()
                self.attention = JointAttention()
                self.feed_forward = nn.Module()
                self.feed_forward.w1 = nn.Linear(dim, dim * 2, bias=False)
                self.feed_forward.w3 = nn.Linear(dim, dim * 2, bias=False)
                self.feed_forward.w2 = nn.Linear(dim * 2, dim, bias=False)

        class NextDiT(nn.Module):
            def __init__(self):
                super().__init__()
                self.n_heads = heads
                self.layers = nn.ModuleList([JointTransformerBlock()])

        model = NextDiT()
        full_qkv = model.layers[0].attention.qkv.weight.data.clone()
        assert patcher.parallelize_model(model) is True
        layer = model.layers[0].attention.qkv
        assert isinstance(layer, ParallelLinear)
        assert layer.pack_sizes == (heads * head_dim, kv_heads * head_dim, kv_heads * head_dim)
        assert getattr(layer.weight, "_tp_pack_sizes", None) == layer.pack_sizes
        got, expected = self._identity_lora(layer, full_qkv, "layers.0.attention.qkv.weight")
        torch.testing.assert_close(got, expected)
        naive = full_qkv[: layer.weight.shape[0]]
        assert not torch.equal(got, naive)

    def test_ideogram_packed_qkv_lora_rank1(self, monkeypatch):
        import torch
        import torch.nn as nn
        from comfy.distributed import patcher
        from comfy.distributed.parallel_linear import ParallelLinear

        self._mesh(monkeypatch, rank=1)
        dim, heads, head_dim = 128, 8, 16

        class Ideogram4Attention(nn.Module):
            def __init__(self):
                super().__init__()
                self.num_heads = heads
                self.head_dim = head_dim
                self.qkv = nn.Linear(dim, dim * 3, bias=False)
                self.o = nn.Linear(dim, dim, bias=False)

        class Ideogram4TransformerBlock(nn.Module):
            def __init__(self):
                super().__init__()
                self.attention = Ideogram4Attention()
                self.feed_forward = nn.Module()
                self.feed_forward.w1 = nn.Linear(dim, dim * 2, bias=False)
                self.feed_forward.w3 = nn.Linear(dim, dim * 2, bias=False)
                self.feed_forward.w2 = nn.Linear(dim * 2, dim, bias=False)
                self.adaln_modulation = nn.Linear(64, 4 * dim)

        class Ideogram4Transformer(nn.Module):
            def __init__(self):
                super().__init__()
                self.head_dim = head_dim
                self.layers = nn.ModuleList([Ideogram4TransformerBlock()])

        model = Ideogram4Transformer()
        full_qkv = model.layers[0].attention.qkv.weight.data.clone()
        assert patcher.parallelize_model(model) is True
        layer = model.layers[0].attention.qkv
        assert isinstance(layer, ParallelLinear)
        assert layer.pack_count == 3
        got, expected = self._identity_lora(layer, full_qkv, "layers.0.attention.qkv.weight")
        torch.testing.assert_close(got, expected)
        naive = full_qkv[: layer.weight.shape[0]]
        assert not torch.equal(got, naive)

    def test_flux_packed_double_stream_qkv_lora_rank1(self, monkeypatch):
        import torch
        import torch.nn as nn
        from comfy.distributed import patcher
        from comfy.distributed.parallel_linear import ParallelLinear

        self._mesh(monkeypatch, rank=1)
        dim = 128

        class SelfAttention(nn.Module):
            def __init__(self):
                super().__init__()
                self.num_heads = 8
                self.qkv = nn.Linear(dim, dim * 3, bias=False)
                self.proj = nn.Linear(dim, dim)

        class DoubleStreamBlock(nn.Module):
            def __init__(self):
                super().__init__()
                self.num_heads = 8
                self.img_attn = SelfAttention()
                self.txt_attn = SelfAttention()
                self.img_mlp = nn.Sequential(nn.Linear(dim, dim * 4), nn.GELU(), nn.Linear(dim * 4, dim))
                self.txt_mlp = nn.Sequential(nn.Linear(dim, dim * 4), nn.GELU(), nn.Linear(dim * 4, dim))

        class Flux(nn.Module):
            def __init__(self):
                super().__init__()
                self.num_heads = 8
                self.double_blocks = nn.ModuleList([DoubleStreamBlock()])

        model = Flux()
        full_qkv = model.double_blocks[0].img_attn.qkv.weight.data.clone()
        assert patcher.parallelize_model(model) is True
        layer = model.double_blocks[0].img_attn.qkv
        assert isinstance(layer, ParallelLinear)
        assert layer.pack_count == 3
        got, expected = self._identity_lora(layer, full_qkv, "double_blocks.0.img_attn.qkv.weight")
        torch.testing.assert_close(got, expected)
        naive = full_qkv[: layer.weight.shape[0]]
        assert not torch.equal(got, naive)

    def test_ace15_unfused_q_proj_lora_rank1(self, monkeypatch):
        import torch
        import torch.nn as nn
        from comfy.distributed import patcher
        from comfy.distributed.parallel_linear import ParallelLinear

        self._mesh(monkeypatch, rank=1)
        hidden, heads, kv_heads, head_dim = 128, 16, 8, 8
        q_dim = heads * head_dim
        kv_dim = kv_heads * head_dim

        class AceStepAttention(nn.Module):
            def __init__(self):
                super().__init__()
                self.num_heads = heads
                self.num_kv_heads = kv_heads
                self.head_dim = head_dim
                self.q_proj = nn.Linear(hidden, q_dim, bias=False)
                self.k_proj = nn.Linear(hidden, kv_dim, bias=False)
                self.v_proj = nn.Linear(hidden, kv_dim, bias=False)
                self.o_proj = nn.Linear(q_dim, hidden, bias=False)

        class AceStepDiTLayer(nn.Module):
            def __init__(self):
                super().__init__()
                self.self_attn = AceStepAttention()

        class AceStepConditionGenerationModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.decoder = nn.Module()
                self.decoder.layers = nn.ModuleList([AceStepDiTLayer()])
                self.lyric_encoder = nn.Module()
                self.lyric_encoder.q_proj = nn.Linear(hidden, q_dim, bias=False)

        model = AceStepConditionGenerationModel()
        full_q = model.decoder.layers[0].self_attn.q_proj.weight.data.clone()
        assert patcher.parallelize_model(model) is True
        layer = model.decoder.layers[0].self_attn.q_proj
        assert isinstance(layer, ParallelLinear)
        got, expected = self._identity_lora(
            layer, full_q, "decoder.layers.0.self_attn.q_proj.weight",
        )
        torch.testing.assert_close(got, expected)
        naive = full_q[: layer.weight.shape[0]]
        assert not torch.equal(got, naive)
        assert isinstance(model.lyric_encoder.q_proj, nn.Linear)

    def test_minimax_packed_qkv_lora_rank1(self, monkeypatch):
        import torch
        import torch.nn as nn
        from comfy.distributed import patcher
        from comfy.distributed.parallel_linear import ParallelLinear

        self._mesh(monkeypatch, rank=1)
        dim, heads, head_dim, ffn = 128, 8, 16, 128
        inner = heads * head_dim

        class Attention(nn.Module):
            def __init__(self):
                super().__init__()
                self.heads = heads
                self.head_dim = head_dim
                self.qkv_proj = nn.Linear(dim, inner * 3, bias=False)
                self.out_proj = nn.Linear(inner, dim, bias=False)

        class DiTBlock(nn.Module):
            def __init__(self):
                super().__init__()
                self.attn = Attention()
                self.mlp = nn.Module()
                self.mlp.fc1 = nn.Linear(dim, ffn * 2, bias=False)
                self.mlp.fc2 = nn.Linear(ffn, dim, bias=False)
                self.adaln_proj = nn.Module()
                self.adaln_proj.linear = nn.Linear(64, 6 * dim * 3)

        class MiniMaxH3Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.blocks = nn.ModuleList([DiTBlock()])

        model = MiniMaxH3Model()
        full_qkv = model.blocks[0].attn.qkv_proj.weight.data.clone()
        assert patcher.parallelize_model(model) is True
        layer = model.blocks[0].attn.qkv_proj
        assert isinstance(layer, ParallelLinear)
        assert layer.pack_count == 3
        got, expected = self._identity_lora(layer, full_qkv, "blocks.0.attn.qkv_proj.weight")
        torch.testing.assert_close(got, expected)
        naive = full_qkv[: layer.weight.shape[0]]
        assert not torch.equal(got, naive)

    def test_nadit_packed_proj_qkv_vid_lora_rank1(self, monkeypatch):
        import torch
        import torch.nn as nn
        from comfy.distributed import patcher
        from comfy.distributed.parallel_linear import ParallelLinear

        self._mesh(monkeypatch, rank=1)
        dim, heads, head_dim = 64, 8, 8

        class MMLinear(nn.Module):
            def __init__(self, in_f, out_f):
                super().__init__()
                self.vid = nn.Linear(in_f, out_f, bias=False)
                self.txt = nn.Linear(in_f, out_f, bias=False)

        class NaMMAttention(nn.Module):
            def __init__(self):
                super().__init__()
                self.heads = heads
                self.head_dim = head_dim
                self.proj_qkv = MMLinear(dim, dim * 3)
                self.proj_out = MMLinear(dim, dim)

        class Block(nn.Module):
            def __init__(self):
                super().__init__()
                self.attn = NaMMAttention()

        class NaDiT(nn.Module):
            def __init__(self):
                super().__init__()
                self.blocks = nn.ModuleList([Block()])

        model = NaDiT()
        full_qkv = model.blocks[0].attn.proj_qkv.vid.weight.data.clone()
        assert patcher.parallelize_model(model) is True
        layer = model.blocks[0].attn.proj_qkv.vid
        assert isinstance(layer, ParallelLinear)
        assert layer.pack_count == 3
        got, expected = self._identity_lora(
            layer, full_qkv, "blocks.0.attn.proj_qkv.vid.weight",
        )
        torch.testing.assert_close(got, expected)
        naive = full_qkv[: layer.weight.shape[0]]
        assert not torch.equal(got, naive)


class TestAnimaTP:
    """Anima is MiniTrainDIT plus an unreplicated llm_adapter outside `blocks`."""

    def test_blocks_head_split_adapter_stays(self, monkeypatch):
        import torch.nn as nn
        from comfy.distributed import patcher
        from comfy.distributed import parallel_linear as pl_module
        from comfy.distributed.parallel_linear import ParallelLinear

        class FakeMesh:
            world_size = 2
            rank = 0
            current_device = "cpu"
        monkeypatch.setattr(patcher, "get_mesh", lambda: FakeMesh())
        monkeypatch.setattr(pl_module, "get_mesh", lambda: FakeMesh())

        dim, heads, head_dim = 64, 8, 8

        class Attention(nn.Module):
            def __init__(self):
                super().__init__()
                self.n_heads = heads
                self.head_dim = head_dim
                self.q_proj = nn.Linear(dim, dim, bias=False)
                self.k_proj = nn.Linear(dim, dim, bias=False)
                self.v_proj = nn.Linear(dim, dim, bias=False)
                self.o_proj = nn.Linear(dim, dim, bias=False)

        class Block(nn.Module):
            def __init__(self):
                super().__init__()
                self.self_attn = Attention()
                self.cross_attn = Attention()
                self.mlp = nn.Module()
                self.mlp.layer1 = nn.Linear(dim, dim * 4, bias=False)
                self.mlp.layer2 = nn.Linear(dim * 4, dim, bias=False)
                self.adaln_modulation_self_attn = nn.Sequential(
                    nn.SiLU(), nn.Linear(dim, 3 * dim, bias=False)
                )

        class MiniTrainDIT(nn.Module):
            def __init__(self):
                super().__init__()
                self.blocks = nn.ModuleList([Block()])

        class LLMAdapter(nn.Module):
            def __init__(self):
                super().__init__()
                self.q_proj = nn.Linear(dim, dim, bias=False)
                self.n_heads = 16

        class Anima(MiniTrainDIT):
            def __init__(self):
                super().__init__()
                self.llm_adapter = LLMAdapter()

        model = Anima()
        assert patcher.parallelize_model(model) is True
        attn = model.blocks[0].self_attn
        assert isinstance(attn.q_proj, ParallelLinear)
        assert attn.q_proj.mode == "colwise"
        assert attn.k_proj.mode == "colwise"
        assert attn.v_proj.mode == "colwise"
        assert attn.o_proj.mode == "rowwise"
        assert attn.n_heads == heads // 2
        assert isinstance(model.blocks[0].mlp.layer1, ParallelLinear)
        assert model.blocks[0].mlp.layer1.mode == "colwise"
        assert model.blocks[0].mlp.layer2.mode == "rowwise"
        assert isinstance(model.blocks[0].adaln_modulation_self_attn[1], nn.Linear)
        assert not isinstance(model.blocks[0].adaln_modulation_self_attn[1], ParallelLinear)
        assert isinstance(model.llm_adapter.q_proj, nn.Linear)
        assert not isinstance(model.llm_adapter.q_proj, ParallelLinear)
        assert model.llm_adapter.n_heads == 16
