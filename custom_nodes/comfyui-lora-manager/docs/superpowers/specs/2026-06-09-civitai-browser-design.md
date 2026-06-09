# CivitAI Model Browser — Design Spec

**Date:** 2026-06-09  
**Status:** Approved

## Overview

Add a "Discover" tab to the LoRA Manager standalone web UI that lets users browse models from CivitAI (civitai.com and civitai.red) they don't yet have installed, and install them with a single click.

---

## Architecture

All CivitAI API calls are proxied through the Python backend — the API key never touches the frontend, and civitai.red URL resolution is handled transparently by the existing `_resolve_civitai_base_url()` logic in `civitai_client.py`.

New pieces:
- `CivitaiClient.browse_models()` — extends the existing client
- `DiscoverRouteRegistrar` — two new GET endpoints
- `static/js/discover.js` — the Discover tab frontend

The install flow reuses the existing `DownloadManager` modal entirely. No new version-picker UI is built.

---

## Backend

### `py/services/civitai_client.py`

New method added to `CivitaiClient`:

```python
async def browse_models(
    self, *,
    model_type: str | None = None,
    base_model: str | None = None,
    sort: str = "Trending",
    period: str = "Week",
    query: str | None = None,
    page: int = 1,
    limit: int = 20,
) -> dict:
```

Calls `GET /models` on the active CivitAI base URL with the provided params. Returns the raw response dict (`{ items: [...], metadata: { totalPages, currentPage, ... } }`).

`model_type` maps to CivitAI's `type` param values: `"LORA"`, `"Checkpoint"`, `"TextualInversion"`, `"Controlnet"`, `"Hypernetwork"`, `"DoRA"`, `"MotionModule"`, `"Poses"`, `"TextEncoder"`, `"Upscaler"`, `"VAE"`, `"Wildcards"`, `"Workflows"`, `"AestheticGradient"`, `"Detection"`, `"Other"`.

### `py/routes/discover_routes.py`

New `DiscoverRouteRegistrar` class following the existing `ModelRouteRegistrar` pattern.

**`GET /discover/browse`**

Query params: `type`, `baseModel`, `sort`, `period`, `page`, `limit`, `query`

Calls `civitai_client.browse_models()` with the forwarded params. Returns:

```json
{
  "items": [...],
  "metadata": { "totalPages": 10, "currentPage": 1 }
}
```

Returns `503` on CivitAI connectivity failure, `429` on rate limit, `401` on auth failure — each with a `{ "error": "..." }` body the frontend renders as an inline message.

**`GET /discover/installed-ids`**

Iterates the live caches of `LoraService`, `CheckpointService`, and `EmbeddingService`. For each cached entry that has `civitai.modelId` set, collects that integer ID. Returns:

```json
{ "ids": [12345, 67890] }
```

Failure is non-fatal — if this endpoint errors, the Discover tab still loads without badges.

### Wiring

`DiscoverRouteRegistrar` is instantiated in:
- `py/lora_manager.py` — alongside existing registrars
- `standalone.py` — for standalone server mode

---

## Frontend

### `static/js/discover.js`

Loaded when the "Discover" nav tab is active. Manages its own state (current filters, page, installed ID set).

**Layout:**

```
[ Search input ] [ Type ▾ ] [ Base Model ▾ ]
┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐
│ card │ │ card │ │ card │ │ card │
└──────┘ └──────┘ └──────┘ └──────┘
                 [ Load More ]
```

**Model cards** display:
- Preview image (falls back to a placeholder if none)
- Model name
- Creator username
- Base model tag
- Type badge
- "Installed" overlay badge if the model's ID is in the installed-ids set
- "Install" button (hidden if already installed)

**Data flow:**

1. Tab opened → `GET /discover/installed-ids` (store result), then `GET /discover/browse` (trending, all types, page 1)
2. Filter or search change → debounced 400 ms → re-fetch `/discover/browse` with new params, reset to page 1, replace grid
3. "Load More" clicked → fetch next page, append cards to grid
4. "Install" clicked → construct `https://civitai.com/models/{id}` → open existing `DownloadManager` modal

**Error states (inline, inside the grid area):**
- 503 / network failure: "Could not reach CivitAI — check your connection" + Retry button
- 429: "CivitAI rate limit hit — please wait a moment" + Retry button
- 401: "CivitAI API key invalid — check Settings"
- Empty results: "No models found — try different filters"

### Templates

- **`templates/components/header.html`** — add a "Discover" nav item following the existing pattern (currently has LoRAs, Recipes, Checkpoints, Embeddings, Statistics)
- **`templates/discover.html`** — new page template, same structure as `loras.html` (extends `base.html`, sets `page_id`, loads `discover.js`)

---

## Install Flow

Clicking "Install" on a card:

1. Constructs the canonical CivitAI model URL: `https://civitai.com/models/{id}`
2. Calls `downloadManager.showDownloadModal(url)` — see note below
3. `DownloadManager` auto-fetches versions for the pre-filled URL, shows which are already installed, lets the user pick one and start the download
4. On download completion the `DownloadManager`'s existing WebSocket progress flow handles progress display

**`DownloadManager` modification required:** `showDownloadModal()` currently takes no arguments and clears the URL input. It needs a `prefillUrl` parameter — when provided, it fills the URL input after `resetSteps()` and auto-triggers the version fetch, skipping directly to the version picker step.

No new download UI is written. The "Installed" badge on Discover cards is refreshed on next tab open (installed-ids is re-fetched each time the tab becomes active).

---

## Error Handling

| Scenario | Backend response | Frontend behavior |
|---|---|---|
| CivitAI unreachable | 503 `{ "error": "..." }` | Inline error + Retry |
| Rate limited | 429 `{ "error": "..." }` | Inline error + Retry |
| Invalid API key | 401 `{ "error": "..." }` | Inline error pointing to Settings |
| Empty results | 200, `items: []` | "No models found" empty state |
| `/discover/installed-ids` fails | — | Tab loads, no badges shown |

---

## Testing

### Backend (pytest)

- `test_browse_models_trending`: mock HTTP, verify default params forwarded
- `test_browse_models_with_filters`: verify type/baseModel/query params forwarded correctly
- `test_browse_models_pagination`: verify page param forwarded, metadata returned
- `test_browse_models_rate_limit`: mock 429 from CivitAI, verify handler returns 429
- `test_installed_ids_aggregates_all_types`: seed mock caches for lora/checkpoint/embedding, verify all three are included in response
- `test_installed_ids_skips_missing_civitai_data`: entries without `civitai.modelId` are not included

### Frontend (vitest)

- Installed badge appears when model ID is in installed set
- Installed badge absent when model ID is not in installed set
- Install button hidden when badge shown
- Filter change triggers debounced re-fetch (fake timers)
- "Load More" appends cards without replacing existing ones
- Error state renders for 503 response
- Empty state renders for empty items array
