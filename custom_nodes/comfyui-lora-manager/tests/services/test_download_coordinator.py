from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple
from unittest.mock import AsyncMock

from py.services.download_coordinator import DownloadCoordinator


@dataclass
class StubWebSocketManager:
    progress: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    broadcasts: List[Tuple[str, Dict[str, Any]]] = field(default_factory=list)

    def generate_download_id(self) -> str:
        return "generated"

    def get_download_progress(self, download_id: str) -> Dict[str, Any] | None:
        return self.progress.get(download_id)

    async def broadcast_download_progress(self, download_id: str, payload: Dict[str, Any]) -> None:
        self.broadcasts.append((download_id, payload))


async def test_pause_download_broadcasts_cached_state():
    ws_manager = StubWebSocketManager(
        progress={
            "dl": {
                "progress": 45,
                "bytes_downloaded": 1024,
                "total_bytes": 2048,
                "bytes_per_second": 256.0,
            }
        }
    )

    download_manager = AsyncMock()
    download_manager.pause_download = AsyncMock(return_value={"success": True})

    async def factory():
        return download_manager

    coordinator = DownloadCoordinator(ws_manager=ws_manager, download_manager_factory=factory)

    result = await coordinator.pause_download("dl")

    assert result == {"success": True}
    assert ws_manager.broadcasts == [
        (
            "dl",
            {
                "status": "paused",
                "progress": 45,
                "download_id": "dl",
                "message": "Download paused by user",
                "bytes_downloaded": 1024,
                "total_bytes": 2048,
                "bytes_per_second": 0.0,
            },
        )
    ]


async def test_resume_download_broadcasts_cached_state():
    ws_manager = StubWebSocketManager(
        progress={
            "dl": {
                "progress": 75,
                "bytes_downloaded": 2048,
                "total_bytes": 4096,
                "bytes_per_second": 512.0,
            }
        }
    )

    download_manager = AsyncMock()
    download_manager.resume_download = AsyncMock(return_value={"success": True})

    async def factory():
        return download_manager

    coordinator = DownloadCoordinator(ws_manager=ws_manager, download_manager_factory=factory)

    result = await coordinator.resume_download("dl")

    assert result == {"success": True}
    assert ws_manager.broadcasts == [
        (
            "dl",
            {
                "status": "downloading",
                "progress": 75,
                "download_id": "dl",
                "message": "Download resumed by user",
                "bytes_downloaded": 2048,
                "total_bytes": 4096,
                "bytes_per_second": 512.0,
            },
        )
    ]


async def test_schedule_download_returns_immediately_without_awaiting_download():
    """schedule_download must return before the download coroutine completes."""
    ws_manager = StubWebSocketManager()

    completed = False

    async def slow_download(**_kwargs):
        nonlocal completed
        await asyncio.sleep(0.05)
        completed = True
        return {"success": True}

    download_manager = AsyncMock()
    download_manager.download_from_civitai = slow_download

    async def factory():
        return download_manager

    coordinator = DownloadCoordinator(ws_manager=ws_manager, download_manager_factory=factory)

    result = await coordinator.schedule_download({"model_id": "1"})

    assert result == {"success": True, "download_id": "generated"}
    assert not completed, "download should still be running in background"

    # Let background task finish
    await asyncio.sleep(0.1)
    assert completed
    assert ws_manager.broadcasts == [("generated", {"status": "completed", "download_id": "generated"})]


async def test_schedule_download_broadcasts_failed_on_error():
    ws_manager = StubWebSocketManager()

    async def failing_download(**_kwargs):
        return {"success": False, "error": "network timeout"}

    download_manager = AsyncMock()
    download_manager.download_from_civitai = failing_download

    async def factory():
        return download_manager

    coordinator = DownloadCoordinator(ws_manager=ws_manager, download_manager_factory=factory)

    result = await coordinator.schedule_download({"model_id": "1"})

    assert result["success"] is True
    await asyncio.sleep(0)
    assert ws_manager.broadcasts == [
        ("generated", {"status": "failed", "download_id": "generated", "error": "network timeout"})
    ]


async def test_schedule_download_broadcasts_skipped_with_base_model():
    ws_manager = StubWebSocketManager()

    async def skipped_download(**_kwargs):
        return {"success": True, "skipped": True, "base_model": "SDXL 1.0"}

    download_manager = AsyncMock()
    download_manager.download_from_civitai = skipped_download

    async def factory():
        return download_manager

    coordinator = DownloadCoordinator(ws_manager=ws_manager, download_manager_factory=factory)

    result = await coordinator.schedule_download({"model_id": "1"})

    assert result["success"] is True
    await asyncio.sleep(0)
    assert ws_manager.broadcasts == [
        ("generated", {"status": "skipped", "download_id": "generated", "base_model": "SDXL 1.0"})
    ]


async def test_pause_download_does_not_broadcast_on_failure():
    ws_manager = StubWebSocketManager()

    download_manager = AsyncMock()
    download_manager.pause_download = AsyncMock(return_value={"success": False, "error": "nope"})

    async def factory():
        return download_manager

    coordinator = DownloadCoordinator(ws_manager=ws_manager, download_manager_factory=factory)

    result = await coordinator.pause_download("dl")

    assert result == {"success": False, "error": "nope"}
    assert ws_manager.broadcasts == []
