# Discover Tab — Model Detail Modal

**Date:** 2026-06-09
**Status:** Approved

## Overview

When a user clicks a card in the Discover tab, a rich model detail modal opens — identical in appearance to the modal shown for installed LoRAs and Checkpoints. Non-installed models get an Install button in place of the Send-to-ComfyUI button. The Install button on the card itself is unchanged.

## Architecture

Three layers of new work:

1. **Backend proxy** — new route proxies CivitAI's full model detail endpoint, returning versions, images, trigger words, stats, and license data.
2. **Frontend entry point** — new function in `ModelModal.js` fetches from the proxy, normalizes the response, and feeds the existing render path with a `isDiscoverModel` flag.
3. **Discover card wiring** — delegated click listener on the grid opens the modal on card body clicks; Install button clicks pass through unchanged.

No new modal HTML element, no new CSS file. Reuses `#modelModal`.

## Backend

**New method on `CivitaiClient`:**
```
get_model_detail(model_id: str) -> dict
```
- Calls `GET https://civitai.com/api/v1/models/{model_id}` via `_make_request`
- Returns full CivitAI response dict on success
- Returns `{"error": str(result)}` dict on failure
- Re-raises `RateLimitError`

**New route in `discover_routes.py`:**
```
GET /api/lm/discover/model/{model_id}
```
- Calls `client.get_model_detail(model_id)`
- 404 from CivitAI (error string contains "404") → returns HTTP 404
- `RateLimitError` → returns HTTP 429
- Auth failure (error string contains "401") → returns HTTP 401
- Any other error → returns HTTP 503
- Registered in `setup_routes(app)`

## Frontend

**`showModelModalForCivitaiId(civitaiId, modelType)`** — exported from `ModelModal.js`:
1. Fetches `GET /api/lm/discover/model/{civitaiId}`
2. Normalizes CivitAI model detail response to the model shape existing render logic expects: `model_name`, `civitai`, `from_civitai: true`, no `file_path`
3. Calls existing render path with `isDiscoverModel: true`
4. `isDiscoverModel` flag:
   - Omit ALL management actions (Send-to-ComfyUI, delete, move, rename)
   - Hide Recipes tab (no local recipes for a non-installed model)
   - Render Install button in place of the management action area

**Install button in modal** calls the same `installModel(model)` already in `discover.js`, which opens `DownloadManager` for the model.

**Card wiring in `discover.js`:**
- `buildCard()` adds `data-civitai-id` and `data-model-type` attributes to each card element
- Single delegated `click` listener on `#discoverGrid`
- If click target is the Install button → pass through (existing behavior)
- Otherwise → call `showModelModalForCivitaiId(civitaiId, modelType)`
- Card shows a brief inline spinner while the fetch is in flight

## Error Handling

| Condition | User-visible result |
|---|---|
| Proxy returns 404 | Toast: "Model not found on CivitAI" |
| Proxy returns 429 | Toast: "CivitAI is rate limiting requests, try again in a moment" |
| Network / 503 | Toast: "Could not reach CivitAI — check your connection" |
| Loading | Card shows spinner; replaced by modal on success |

## Testing

**`tests/routes/test_discover_routes.py`** — 4 new cases for `get_model_detail`:
- Success returns full payload with HTTP 200
- CivitAI 404 error → HTTP 404
- `RateLimitError` → HTTP 429
- Generic error → HTTP 503

**`tests/services/test_civitai_browse.py`** — 3 new cases for `CivitaiClient.get_model_detail`:
- Success returns response dict
- `_make_request` failure → returns error dict
- `RateLimitError` re-raised

**`tests/frontend/pages/discover.test.js`** — 2 new cases:
- Card body click calls `showModelModalForCivitaiId` with correct `civitaiId` and `modelType`
- Install button click does NOT trigger the modal

## Files Changed

| File | Change |
|---|---|
| `py/services/civitai_client.py` | Add `get_model_detail(model_id)` method |
| `py/routes/discover_routes.py` | Add `get_model_detail` handler, register route |
| `static/js/components/shared/ModelModal.js` | Add `showModelModalForCivitaiId`, `isDiscoverModel` flag |
| `static/js/discover.js` | Add `data-civitai-id` attrs, delegated click listener, card spinner |
| `tests/routes/test_discover_routes.py` | 4 new test cases |
| `tests/services/test_civitai_browse.py` | 3 new test cases |
| `tests/frontend/pages/discover.test.js` | 2 new test cases |
