# beast-utils Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Create a `beast-utils` custom node that adds two action bar buttons to ComfyUI — one to unload all models and clear VRAM/RAM, one to restart the service with auto-reconnect.

**Architecture:** A minimal custom node with no workflow nodes — just a Python `__init__.py` that registers two aiohttp API routes, and a single JS file that registers two action bar buttons via `app.registerExtension`. The restart mechanism sends SIGTERM to the torchrun parent process; systemd's `Restart=always` brings it back up in ~30s. The JS polls `/system_stats` until ComfyUI responds, then reloads.

**Tech Stack:** Python 3.13, aiohttp (via ComfyUI's PromptServer), vanilla ES module JS, ComfyUI frontend 1.46.x `actionBarButtons` API, pytest + unittest.mock for tests.

---

## File Map

| Path | Action | Responsibility |
|------|--------|----------------|
| `custom_nodes/beast-utils/__init__.py` | Create | Route registration, rank guard, free-memory + restart handlers |
| `custom_nodes/beast-utils/js/beast_utils.js` | Create | Action bar buttons, confirm dialogs, toast, restart overlay + polling |
| `tests-unit/comfy_test/beast_utils_test/__init__.py` | Create | Empty — makes directory a package |
| `tests-unit/comfy_test/beast_utils_test/test_init.py` | Create | Pytest tests for route handlers |

---

## Task 1: Python backend — route handlers + rank guard

**Files:**
- Create: `custom_nodes/beast-utils/__init__.py`
- Create: `tests-unit/comfy_test/beast_utils_test/__init__.py`
- Create: `tests-unit/comfy_test/beast_utils_test/test_init.py`

- [ ] **Step 1: Write failing tests**

Create `tests-unit/comfy_test/beast_utils_test/__init__.py` (empty file).

Create `tests-unit/comfy_test/beast_utils_test/test_init.py`:

```python
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
        with patch("comfy.model_management.unload_all_models") as mock_unload, \
             patch("comfy.model_management.soft_empty_cache") as mock_cache, \
             patch("gc.collect") as mock_gc:
            request = make_mocked_request("POST", "/beast/free-memory")
            await mod.free_memory(request)
        mock_unload.assert_called_once()
        mock_cache.assert_called_once()
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


class TestRankGuard:
    def test_registers_routes_on_rank_zero(self, monkeypatch):
        _, mock_server = load_beast_utils(monkeypatch, local_rank="0")
        assert mock_server.PromptServer.instance.app.router.add_post.call_count == 2

    def test_skips_routes_on_worker_rank(self, monkeypatch):
        _, mock_server = load_beast_utils(monkeypatch, local_rank="1")
        mock_server.PromptServer.instance.app.router.add_post.assert_not_called()
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
cd /home/blackthorn/ComfyUI
pytest tests-unit/comfy_test/beast_utils_test/test_init.py -v 2>&1 | head -20
```

Expected: fails because `custom_nodes/beast-utils/__init__.py` doesn't exist yet.

- [ ] **Step 3: Create the node directory and `__init__.py`**

```bash
mkdir -p /home/blackthorn/ComfyUI/custom_nodes/beast-utils/js
```

Create `custom_nodes/beast-utils/__init__.py`:

```python
from __future__ import annotations

import gc
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
    import comfy.model_management
    comfy.model_management.unload_all_models()
    comfy.model_management.soft_empty_cache()
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
```

- [ ] **Step 4: Run tests — expect all pass**

```bash
cd /home/blackthorn/ComfyUI
pytest tests-unit/comfy_test/beast_utils_test/test_init.py -v --asyncio-mode=auto
```

Expected:
```
PASSED test_returns_success_json
PASSED test_calls_unload_and_cache_clear
PASSED test_returns_success_immediately
PASSED test_spawns_background_thread
PASSED test_registers_routes_on_rank_zero
PASSED test_skips_routes_on_worker_rank
6 passed
```

If `--asyncio-mode=auto` is rejected, check `pytest.ini` in the repo root and add:
```ini
[pytest]
asyncio_mode = auto
```
or run with `-p no:asyncio` and mark async tests with `@pytest.mark.asyncio` explicitly.

- [ ] **Step 5: Commit**

```bash
cd /home/blackthorn/ComfyUI
git add custom_nodes/beast-utils/__init__.py \
        tests-unit/comfy_test/beast_utils_test/__init__.py \
        tests-unit/comfy_test/beast_utils_test/test_init.py
git commit -m "feat(beast-utils): add Python backend with free-memory and restart routes"
```

---

## Task 2: JS frontend — action bar buttons

**Files:**
- Create: `custom_nodes/beast-utils/js/beast_utils.js`

- [ ] **Step 1: Create the JS extension file**

Create `custom_nodes/beast-utils/js/beast_utils.js`:

```javascript
import { app } from "../../scripts/app.js";

const STYLE_ID = "beast-utils-styles";

function injectStyles() {
    if (document.getElementById(STYLE_ID)) return;
    const style = document.createElement("style");
    style.id = STYLE_ID;
    style.textContent = [
        "#beast-utils-toast {",
        "  position: fixed; top: 48px; left: 50%;",
        "  transform: translateX(-50%);",
        "  background: var(--comfy-menu-bg, #1a1a2e);",
        "  color: var(--input-text, #fff);",
        "  border: 1px solid var(--border-color, #444);",
        "  border-radius: 6px; padding: 8px 20px;",
        "  font-size: 14px; z-index: 99999;",
        "  pointer-events: none;",
        "  opacity: 1; transition: opacity 0.4s ease;",
        "}",
        "#beast-utils-toast.fade-out { opacity: 0; }",
        "#beast-utils-overlay {",
        "  position: fixed; inset: 0;",
        "  background: rgba(0,0,0,0.75);",
        "  display: flex; flex-direction: column;",
        "  align-items: center; justify-content: center;",
        "  z-index: 99999; color: #fff;",
        "  font-size: 18px; gap: 20px;",
        "}",
        ".beast-utils-spinner {",
        "  width: 48px; height: 48px;",
        "  border: 5px solid rgba(255,255,255,0.2);",
        "  border-top-color: #fff; border-radius: 50%;",
        "  animation: beast-spin 0.8s linear infinite;",
        "}",
        "@keyframes beast-spin { to { transform: rotate(360deg); } }",
    ].join("\n");
    document.head.appendChild(style);
}

function showToast(message, durationMs = 2000) {
    document.getElementById("beast-utils-toast")?.remove();
    const toast = document.createElement("div");
    toast.id = "beast-utils-toast";
    toast.textContent = message;
    document.body.appendChild(toast);
    setTimeout(() => {
        toast.classList.add("fade-out");
        setTimeout(() => toast.remove(), 400);
    }, durationMs);
}

function showRestartOverlay() {
    const overlay = document.createElement("div");
    overlay.id = "beast-utils-overlay";

    const spinner = document.createElement("div");
    spinner.className = "beast-utils-spinner";

    const label = document.createElement("div");
    label.textContent = "Restarting ComfyUI…";

    const sub = document.createElement("div");
    sub.style.cssText = "font-size:13px;opacity:0.6";
    sub.textContent = "Will reconnect automatically";

    overlay.appendChild(spinner);
    overlay.appendChild(label);
    overlay.appendChild(sub);
    document.body.appendChild(overlay);
}

function removeOverlay() {
    document.getElementById("beast-utils-overlay")?.remove();
}

async function pollUntilAlive() {
    while (true) {
        await new Promise(r => setTimeout(r, 2000));
        try {
            const res = await fetch("/system_stats", { signal: AbortSignal.timeout(3000) });
            if (res.ok) return;
        } catch {
            // still restarting — keep polling
        }
    }
}

async function onFreeMemory() {
    if (!confirm("Unload all models and free VRAM/RAM?\n\nThis clears everything currently in memory.")) return;
    try {
        const res = await fetch("/beast/free-memory", { method: "POST" });
        const data = await res.json();
        if (data.success) {
            showToast("Memory freed");
        } else {
            alert("Free memory failed: " + (data.error ?? "unknown error"));
        }
    } catch (err) {
        alert("Free memory request failed: " + err.message);
    }
}

async function onRestart() {
    if (!confirm("Restart ComfyUI?\n\nThe page will reconnect automatically (~30 seconds).")) return;
    try {
        const res = await fetch("/beast/restart", { method: "POST" });
        const data = await res.json();
        if (!data.success) {
            alert("Restart failed: " + (data.error ?? "unknown error"));
            return;
        }
    } catch (err) {
        alert("Restart request failed: " + err.message);
        return;
    }
    showRestartOverlay();
    await pollUntilAlive();
    removeOverlay();
    window.location.reload();
}

app.registerExtension({
    name: "BeastUtils.ActionBar",
    async setup() {
        injectStyles();
    },
    actionBarButtons: [
        {
            icon: "icon-[mdi--memory] size-4",
            tooltip: "Free VRAM & RAM (unload all models)",
            onClick: onFreeMemory,
        },
        {
            icon: "icon-[mdi--restart] size-4",
            tooltip: "Restart ComfyUI",
            onClick: onRestart,
        },
    ],
});
```

- [ ] **Step 2: Verify the file is in place**

```bash
ls -la /home/blackthorn/ComfyUI/custom_nodes/beast-utils/js/beast_utils.js
```

Expected: file exists, non-zero size.

- [ ] **Step 3: Commit**

```bash
cd /home/blackthorn/ComfyUI
git add custom_nodes/beast-utils/js/beast_utils.js
git commit -m "feat(beast-utils): add JS action bar buttons with toast and restart overlay"
```

---

## Task 3: Integration test

**Files:** None — manual browser verification.

- [ ] **Step 1: Restart the service**

```bash
sudo systemctl restart comfyui-tp.service
```

Wait ~45 seconds for both TP ranks to fully initialize.

- [ ] **Step 2: Verify the node loads cleanly**

```bash
journalctl -u comfyui-tp.service -n 50 --no-pager | grep -i "beast-utils\|IMPORT FAILED"
```

Expected: `[beast-utils] Routes registered: /beast/free-memory, /beast/restart` appears once (rank 0). No `IMPORT FAILED` line for beast-utils.

- [ ] **Step 3: Confirm buttons appear**

Open ComfyUI in a browser. Two new icons should appear in the action bar. Hover each to confirm tooltips: "Free VRAM & RAM (unload all models)" and "Restart ComfyUI".

- [ ] **Step 4: Test Free Memory button**

Click the memory chip icon → confirm → click OK → toast "Memory freed" appears at top-center and fades after 2 seconds.

Confirm in logs:
```bash
journalctl -u comfyui-tp.service -n 5 --no-pager | grep "beast-utils"
```
Expected: `[beast-utils] Memory freed: models unloaded, VRAM/RAM cache cleared`

- [ ] **Step 5: Test Restart button**

Click the restart icon → confirm → click OK → fullscreen dark overlay with spinner appears → after ~30–40 seconds overlay disappears and page reloads.

Confirm service is back:
```bash
systemctl is-active comfyui-tp.service
```
Expected: `active`

- [ ] **Step 6: Final commit**

```bash
cd /home/blackthorn/ComfyUI
git status
# if any final tweaks were made during testing:
git add custom_nodes/beast-utils/
git commit -m "feat(beast-utils): beast-utils node complete and integration-verified"
```
