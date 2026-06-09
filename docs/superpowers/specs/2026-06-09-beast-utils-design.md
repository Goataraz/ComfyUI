# beast-utils Custom Node — Design Spec

**Date:** 2026-06-09  
**Status:** Approved

## Overview

A minimal custom ComfyUI node (`custom_nodes/beast-utils/`) that adds two utility buttons to the ComfyUI action bar: one to unload all models and clear VRAM/RAM, and one to restart the ComfyUI service with auto-reconnect.

No workflow nodes are exposed — this is pure infrastructure utility.

---

## Architecture

### Files

```
custom_nodes/beast-utils/
├── __init__.py       # Python: registers API routes, sets WEB_DIRECTORY
└── js/
    └── beast_utils.js  # JS: registers action bar buttons + UI
```

### Python (`__init__.py`)

Registers two aiohttp routes on ComfyUI's PromptServer and exports `WEB_DIRECTORY = "./js"`. No `NODE_CLASS_MAPPINGS` — no workflow nodes.

**`POST /beast/free-memory`**
- Calls `comfy.model_management.unload_all_models()`
- Calls `comfy.model_management.soft_empty_cache()`
- Calls `gc.collect()`
- Returns `{"success": true, "message": "Memory freed"}`
- On rank 1 (TP worker), returns early with success — rank 0 handles the actual unload

**`POST /beast/restart`**
- Returns `{"success": true}` immediately
- Spawns a background thread that sleeps 0.5s then calls `os.kill(os.getppid(), signal.SIGTERM)`
- `os.getppid()` targets the torchrun parent process
- systemd's `Restart=always` (RestartSec=30) brings the service back up

### JS (`js/beast_utils.js`)

Registers a `ComfyUI` extension via `app.registerExtension`. Targets frontend ≥ 1.33.9 using the `actionBarButtons` API (same approach as LoRA Manager). No legacy fallback needed — Beast runs 1.46.11.

**Free Memory button**
- Icon: `icon-[mdi--memory]` 
- Tooltip: "Free VRAM & RAM (unload all models)"
- Click → browser `confirm()` dialog → `POST /beast/free-memory` → small auto-dismissing status div injected into the page: "Memory freed" (2s, top-center)

**Restart button**
- Icon: `icon-[mdi--restart]`
- Tooltip: "Restart ComfyUI"
- Click → browser `confirm()` dialog → `POST /beast/restart` → fullscreen overlay showing "Restarting ComfyUI…" spinner
- Overlay polls `GET /system_stats` every 2 seconds
- When `/system_stats` responds successfully, calls `window.location.reload()`

---

## Error Handling

- Free memory: if the fetch fails, show an alert with the error message
- Restart: if the initial POST fails, dismiss overlay and show alert
- Reconnect polling: no timeout — keeps polling indefinitely until the server responds (systemd will always bring it back)

---

## Constraints

- TP mode: rank 1 must not register routes (it has no real HTTP server). Guard by checking `os.environ.get("LOCAL_RANK", "0") != "0"` — if non-zero rank, skip route registration and set `NODE_CLASS_MAPPINGS = {}` only.
- No sudo required: `os.kill(os.getppid(), SIGTERM)` works because the torchrun parent is owned by the same user (`blackthorn`)
- No workflow nodes: `NODE_CLASS_MAPPINGS = {}` and `NODE_DISPLAY_NAME_MAPPINGS = {}`

---

## Out of Scope

- Configurable restart delay
- Selective model unloading (unloads all)
- Restart on workflow completion
