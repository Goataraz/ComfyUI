from __future__ import annotations

import gc
import importlib
import logging
import os
import signal
import threading
import time

from aiohttp import web

logger = logging.getLogger(__name__)

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}
WEB_DIRECTORY = "./js"

_IS_WORKER_RANK = os.environ.get("LOCAL_RANK", "0") != "0"


async def free_memory(request: web.Request) -> web.Response:
    mm = importlib.import_module("comfy.model_management")
    mm.unload_all_models()
    mm.soft_empty_cache()
    gc.collect()
    logger.info("[beast-utils] Memory freed: models unloaded, VRAM/RAM cache cleared")
    return web.json_response({"success": True, "message": "Memory freed"})


async def restart(request: web.Request) -> web.Response:
    def _kill_parent() -> None:
        time.sleep(0.5)
        parent_pid = os.getppid()
        logger.info("[beast-utils] Sending SIGTERM to parent process %d", parent_pid)
        os.kill(parent_pid, signal.SIGTERM)

    t = threading.Thread(target=_kill_parent, daemon=True)
    t.start()
    return web.json_response({"success": True})


if not _IS_WORKER_RANK:
    try:
        from server import PromptServer  # type: ignore
        PromptServer.instance.app.router.add_post("/beast/free-memory", free_memory)
        PromptServer.instance.app.router.add_post("/beast/restart", restart)
        logger.info("[beast-utils] Routes registered: /beast/free-memory, /beast/restart")
    except Exception as e:
        logger.error("[beast-utils] Failed to register routes: %s", e)
