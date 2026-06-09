from __future__ import annotations

import gc
import importlib.util
import json
import os
import pathlib
import sys
import threading
from unittest.mock import MagicMock, patch

import pytest
from aiohttp.test_utils import make_mocked_request


INIT_PATH = pathlib.Path(__file__).parents[3] / "custom_nodes" / "beast-utils" / "__init__.py"


def load_beast_utils(monkeypatch, local_rank="0"):
    """Load beast-utils __init__.py with mocked ComfyUI dependencies."""
    monkeypatch.setenv("LOCAL_RANK", local_rank)

    mock_server = MagicMock()
    monkeypatch.setitem(sys.modules, "server", mock_server)
    monkeypatch.setitem(sys.modules, "comfy", MagicMock())
    monkeypatch.setitem(sys.modules, "comfy.model_management", MagicMock())

    spec = importlib.util.spec_from_file_location("beast_utils_mod", INIT_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod, mock_server


class TestFreeMemoryHandler:
    @pytest.mark.asyncio
    async def test_returns_success_json(self, monkeypatch):
        mod, _ = load_beast_utils(monkeypatch)
        request = make_mocked_request("POST", "/beast/free-memory")
        response = await mod.free_memory(request)
        assert response.status == 200
        data = json.loads(response.body)
        assert data["success"] is True
        assert "memory" in data["message"].lower()

    @pytest.mark.asyncio
    async def test_calls_unload_and_cache_clear(self, monkeypatch):
        mod, _ = load_beast_utils(monkeypatch)
        mm_mock = sys.modules["comfy.model_management"]
        with patch("gc.collect") as mock_gc:
            request = make_mocked_request("POST", "/beast/free-memory")
            await mod.free_memory(request)
        mm_mock.unload_all_models.assert_called_once()
        mm_mock.soft_empty_cache.assert_called_once()
        mock_gc.assert_called_once()


class TestRestartHandler:
    @pytest.mark.asyncio
    async def test_returns_success_immediately(self, monkeypatch):
        mod, _ = load_beast_utils(monkeypatch)
        with patch("os.kill"), patch("threading.Thread"):
            request = make_mocked_request("POST", "/beast/restart")
            response = await mod.restart(request)
        assert response.status == 200
        data = json.loads(response.body)
        assert data["success"] is True

    @pytest.mark.asyncio
    async def test_spawns_background_thread(self, monkeypatch):
        mod, _ = load_beast_utils(monkeypatch)
        with patch("threading.Thread") as mock_thread_cls:
            mock_thread = MagicMock()
            mock_thread_cls.return_value = mock_thread
            request = make_mocked_request("POST", "/beast/restart")
            await mod.restart(request)
        mock_thread_cls.assert_called_once()
        mock_thread.start.assert_called_once()

    @pytest.mark.asyncio
    async def test_kills_parent_with_sigterm(self, monkeypatch):
        import signal as signal_mod
        import time as real_time
        mod, _ = load_beast_utils(monkeypatch)
        kill_calls = []
        monkeypatch.setattr(mod.time, "sleep", lambda _: None)
        monkeypatch.setattr(mod.os, "kill", lambda pid, sig: kill_calls.append((pid, sig)))
        monkeypatch.setattr(mod.os, "getppid", lambda: 12345)
        request = make_mocked_request("POST", "/beast/restart")
        await mod.restart(request)
        real_time.sleep(0.05)
        assert kill_calls == [(12345, signal_mod.SIGTERM)]


class TestRankGuard:
    def test_registers_routes_on_rank_zero(self, monkeypatch):
        _, mock_server = load_beast_utils(monkeypatch, local_rank="0")
        assert mock_server.PromptServer.instance.app.router.add_post.call_count == 2

    def test_skips_routes_on_worker_rank(self, monkeypatch):
        _, mock_server = load_beast_utils(monkeypatch, local_rank="1")
        mock_server.PromptServer.instance.app.router.add_post.assert_not_called()
