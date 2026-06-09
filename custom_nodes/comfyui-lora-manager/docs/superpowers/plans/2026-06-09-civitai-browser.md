# CivitAI Model Browser Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a "Discover" tab to the LoRA Manager standalone UI that lets users browse CivitAI models, see which ones are already installed, and one-click install any version via the existing DownloadManager.

**Architecture:** A new `DiscoverRoutes` class registers two GET endpoints — `/api/lm/discover/browse` (proxies CivitAI's `/models` API) and `/api/lm/discover/installed-ids` (aggregates CivitAI model IDs from the three local caches). `discover.js` renders a filterable card grid; clicking "Install" calls the already-existing `downloadManager.openForModelVersion(modelType, modelId)` which handles the version picker and download flow with no modifications needed.

**Tech Stack:** Python/aiohttp backend, vanilla JS ES modules frontend, Jinja2 templates, pytest + vitest for tests.

> **Note:** The spec described modifying `DownloadManager.showDownloadModal()` to accept a `prefillUrl` parameter. This is unnecessary — `DownloadManager` already has `openForModelVersion(modelType, modelId)` which does exactly what's needed. No DownloadManager changes are required.

> **Security:** All DOM construction uses `createElement`, `textContent`, `appendChild`. No `innerHTML` (except clearing via `replaceChildren()` which is safe). No user data is ever interpolated into HTML strings.

---

## File Map

| Action | Path | Responsibility |
|---|---|---|
| Modify | `py/services/civitai_client.py` | Add `browse_models()` method |
| Create | `py/routes/discover_routes.py` | `DiscoverRoutes` class with page + API handlers |
| Modify | `py/lora_manager.py` | Wire `DiscoverRoutes` into plugin mode |
| Modify | `standalone.py` | Wire `DiscoverRoutes` into standalone mode |
| Modify | `templates/components/header.html` | Add Discover nav item |
| Create | `templates/discover.html` | Discover page template |
| Create | `static/js/discover.js` | Discover tab JS (card grid, filters, install trigger) |
| Create | `static/css/discover.css` | Discover page styles |
| Create | `tests/services/test_civitai_browse.py` | Unit tests for `browse_models()` |
| Create | `tests/routes/test_discover_routes.py` | Unit tests for route handlers |
| Create | `tests/frontend/pages/discover.test.js` | Frontend unit tests |

---

## Task 1: `CivitaiClient.browse_models()`

**Files:**
- Modify: `py/services/civitai_client.py` (append after `get_user_models`, around line 812)
- Create: `tests/services/test_civitai_browse.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/services/test_civitai_browse.py`:

```python
from unittest.mock import AsyncMock

import pytest

from py.services import civitai_client as civitai_client_module
from py.services.civitai_client import CivitaiClient


@pytest.fixture(autouse=True)
def reset_singleton():
    CivitaiClient._instance = None
    yield
    CivitaiClient._instance = None


@pytest.fixture
def downloader(monkeypatch):
    d = AsyncMock()
    d.make_request = AsyncMock(
        return_value=(True, {"items": [], "metadata": {"totalPages": 1, "currentPage": 1}})
    )
    monkeypatch.setattr(
        civitai_client_module, "get_downloader", AsyncMock(return_value=d)
    )
    return d


@pytest.mark.asyncio
async def test_browse_models_default_params(downloader):
    client = await CivitaiClient.get_instance()
    result = await client.browse_models()
    params = downloader.make_request.call_args.kwargs["params"]
    assert params["sort"] == "Trending"
    assert params["period"] == "Week"
    assert params["page"] == 1
    assert params["limit"] == 20
    assert "types" not in params
    assert "baseModels" not in params
    assert result["items"] == []


@pytest.mark.asyncio
async def test_browse_models_with_type_filter(downloader):
    client = await CivitaiClient.get_instance()
    await client.browse_models(model_type="LORA")
    params = downloader.make_request.call_args.kwargs["params"]
    assert params["types"] == "LORA"


@pytest.mark.asyncio
async def test_browse_models_with_base_model_filter(downloader):
    client = await CivitaiClient.get_instance()
    await client.browse_models(base_model="Flux.1 D")
    params = downloader.make_request.call_args.kwargs["params"]
    assert params["baseModels"] == "Flux.1 D"


@pytest.mark.asyncio
async def test_browse_models_with_query_and_page(downloader):
    client = await CivitaiClient.get_instance()
    await client.browse_models(query="anime girl", page=3, limit=10)
    params = downloader.make_request.call_args.kwargs["params"]
    assert params["query"] == "anime girl"
    assert params["page"] == 3
    assert params["limit"] == 10


@pytest.mark.asyncio
async def test_browse_models_returns_error_dict_on_failure(downloader):
    downloader.make_request = AsyncMock(return_value=(False, "server error"))
    client = await CivitaiClient.get_instance()
    result = await client.browse_models()
    assert "error" in result
    assert result["items"] == []
    assert result["metadata"] == {}
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd /home/blackthorn/ComfyUI/custom_nodes/comfyui-lora-manager
pytest tests/services/test_civitai_browse.py -v
```

Expected: All 5 tests FAIL with `AttributeError: 'CivitaiClient' object has no attribute 'browse_models'`

- [ ] **Step 3: Implement `browse_models()`**

Open `py/services/civitai_client.py` and add this method after `get_user_models()` (after line 811):

```python
    async def browse_models(
        self,
        *,
        model_type: str | None = None,
        base_model: str | None = None,
        sort: str = "Trending",
        period: str = "Week",
        query: str | None = None,
        page: int = 1,
        limit: int = 20,
    ) -> dict:
        """Browse CivitAI models with optional filtering."""
        params: dict = {"sort": sort, "period": period, "page": page, "limit": limit}
        if model_type:
            params["types"] = model_type
        if base_model:
            params["baseModels"] = base_model
        if query:
            params["query"] = query

        try:
            success, result = await self._make_request(
                "GET",
                f"{self.base_url}/models",
                use_auth=True,
                params=params,
            )
            if not success:
                if is_expected_offline_error(result):
                    logger.info("browse_models skipped: %s", OFFLINE_FRIENDLY_MESSAGE)
                return {"error": str(result), "items": [], "metadata": {}}
            return result if isinstance(result, dict) else {"items": [], "metadata": {}}
        except RateLimitError:
            raise
        except Exception as exc:
            logger.error("Error in browse_models: %s", exc)
            return {"error": str(exc), "items": [], "metadata": {}}
```

`is_expected_offline_error`, `OFFLINE_FRIENDLY_MESSAGE`, and `RateLimitError` are already imported (they're used in `get_user_models`).

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/services/test_civitai_browse.py -v
```

Expected: All 5 tests PASS

- [ ] **Step 5: Commit**

```bash
git -C /home/blackthorn/ComfyUI add \
  custom_nodes/comfyui-lora-manager/py/services/civitai_client.py \
  custom_nodes/comfyui-lora-manager/tests/services/test_civitai_browse.py
git -C /home/blackthorn/ComfyUI commit -m "feat(discover): add CivitaiClient.browse_models() with filtering and pagination"
```

---

## Task 2: Backend Route Handlers

**Files:**
- Create: `py/routes/discover_routes.py`
- Create: `tests/routes/test_discover_routes.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/routes/test_discover_routes.py`:

```python
import json
from unittest.mock import AsyncMock

import pytest
from aiohttp.test_utils import make_mocked_request

from py.routes import discover_routes as discover_module
from py.routes.discover_routes import DiscoverRoutes


class FakeCache:
    def __init__(self, raw_data):
        self.raw_data = raw_data


class FakeScanner:
    def __init__(self, raw_data):
        self._cache = FakeCache(raw_data)

    async def get_cached_data(self):
        return self._cache


@pytest.fixture
def routes(monkeypatch):
    lora_data = [
        {"sha256": "a", "civitai": {"modelId": 100, "id": 200}},
        {"sha256": "b", "civitai": {"modelId": 101, "id": 201}},
        {"sha256": "c"},  # no civitai key
    ]
    checkpoint_data = [{"sha256": "d", "civitai": {"modelId": 200, "id": 300}}]
    embedding_data = [{"sha256": "e", "civitai": {"modelId": 300, "id": 400}}]

    async def fake_lora(cls):
        return FakeScanner(lora_data)

    async def fake_checkpoint(cls):
        return FakeScanner(checkpoint_data)

    async def fake_embedding(cls):
        return FakeScanner(embedding_data)

    monkeypatch.setattr(
        discover_module.ServiceRegistry, "get_lora_scanner", classmethod(fake_lora)
    )
    monkeypatch.setattr(
        discover_module.ServiceRegistry,
        "get_checkpoint_scanner",
        classmethod(fake_checkpoint),
    )
    monkeypatch.setattr(
        discover_module.ServiceRegistry,
        "get_embedding_scanner",
        classmethod(fake_embedding),
    )
    return DiscoverRoutes()


@pytest.mark.asyncio
async def test_installed_ids_aggregates_all_three_caches(routes):
    request = make_mocked_request("GET", "/api/lm/discover/installed-ids")
    response = await routes.get_installed_ids(request)
    data = json.loads(response.body)
    assert set(data["ids"]) == {100, 101, 200, 300}


@pytest.mark.asyncio
async def test_installed_ids_skips_entries_without_civitai(routes):
    request = make_mocked_request("GET", "/api/lm/discover/installed-ids")
    response = await routes.get_installed_ids(request)
    data = json.loads(response.body)
    # sha256="c" has no civitai dict — only 4 IDs total
    assert len(data["ids"]) == 4


@pytest.mark.asyncio
async def test_browse_forwards_params_to_client(monkeypatch):
    mock_client = AsyncMock()
    mock_client.browse_models = AsyncMock(
        return_value={"items": [], "metadata": {"totalPages": 1, "currentPage": 1}}
    )
    monkeypatch.setattr(
        discover_module.CivitaiClient, "get_instance", AsyncMock(return_value=mock_client)
    )
    routes = DiscoverRoutes()
    request = make_mocked_request(
        "GET", "/api/lm/discover/browse?type=LORA&baseModel=Flux.1+D&page=2&query=cat"
    )
    await routes.browse_models(request)
    mock_client.browse_models.assert_called_once_with(
        model_type="LORA",
        base_model="Flux.1 D",
        sort="Trending",
        period="Week",
        query="cat",
        page=2,
        limit=20,
    )


@pytest.mark.asyncio
async def test_browse_returns_civitai_payload(monkeypatch):
    payload = {"items": [{"id": 1}], "metadata": {"totalPages": 1, "currentPage": 1}}
    mock_client = AsyncMock()
    mock_client.browse_models = AsyncMock(return_value=payload)
    monkeypatch.setattr(
        discover_module.CivitaiClient, "get_instance", AsyncMock(return_value=mock_client)
    )
    routes = DiscoverRoutes()
    request = make_mocked_request("GET", "/api/lm/discover/browse")
    response = await routes.browse_models(request)
    assert response.status == 200
    assert json.loads(response.body) == payload


@pytest.mark.asyncio
async def test_browse_returns_503_on_error(monkeypatch):
    mock_client = AsyncMock()
    mock_client.browse_models = AsyncMock(
        return_value={"error": "connection refused", "items": [], "metadata": {}}
    )
    monkeypatch.setattr(
        discover_module.CivitaiClient, "get_instance", AsyncMock(return_value=mock_client)
    )
    routes = DiscoverRoutes()
    request = make_mocked_request("GET", "/api/lm/discover/browse")
    response = await routes.browse_models(request)
    assert response.status == 503
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/routes/test_discover_routes.py -v
```

Expected: All 5 tests FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Create `py/routes/discover_routes.py`**

```python
from __future__ import annotations

import logging

import jinja2
from aiohttp import web

from ..config import config
from ..services.civitai_client import CivitaiClient
from ..services.errors import RateLimitError
from ..services.server_i18n import server_i18n
from ..services.service_registry import ServiceRegistry

logger = logging.getLogger(__name__)


class DiscoverRoutes:
    """Route handlers for the Discover (CivitAI browser) page."""

    def __init__(self) -> None:
        self._template_env = jinja2.Environment(
            loader=jinja2.FileSystemLoader(config.templates_path),
            autoescape=True,
        )

    def setup_routes(self, app: web.Application) -> None:
        app.router.add_get("/discover", self.handle_discover_page)
        app.router.add_get("/api/lm/discover/browse", self.browse_models)
        app.router.add_get("/api/lm/discover/installed-ids", self.get_installed_ids)

    async def handle_discover_page(self, request: web.Request) -> web.Response:
        try:
            user_language = "en"
            try:
                from ..services.settings_manager import get_settings_manager
                user_language = get_settings_manager().get("language", "en")
            except Exception:
                pass
            server_i18n.set_locale(user_language)
            if not hasattr(self._template_env, "_i18n_filter_added"):
                self._template_env.filters["t"] = server_i18n.create_template_filter()
                self._template_env._i18n_filter_added = True
            rendered = self._template_env.get_template("discover.html").render(
                is_initializing=False,
                request=request,
                t=server_i18n.get_translation,
            )
            return web.Response(text=rendered, content_type="text/html")
        except Exception as exc:
            logger.error("Error rendering discover page: %s", exc, exc_info=True)
            return web.Response(text="Error loading discover page", status=500)

    async def browse_models(self, request: web.Request) -> web.Response:
        q = request.rel_url.query
        model_type = q.get("type") or None
        base_model = q.get("baseModel") or None
        sort = q.get("sort", "Trending")
        period = q.get("period", "Week")
        query = q.get("query") or None
        try:
            page = int(q.get("page", "1"))
            limit = int(q.get("limit", "20"))
        except ValueError:
            page, limit = 1, 20

        try:
            client = await CivitaiClient.get_instance()
            result = await client.browse_models(
                model_type=model_type,
                base_model=base_model,
                sort=sort,
                period=period,
                query=query,
                page=page,
                limit=limit,
            )
        except RateLimitError:
            return web.json_response({"error": "Rate limited by CivitAI"}, status=429)

        if "error" in result:
            return web.json_response({"error": result["error"]}, status=503)
        return web.json_response(result)

    async def get_installed_ids(self, request: web.Request) -> web.Response:
        ids: set[int] = set()
        for getter in (
            ServiceRegistry.get_lora_scanner,
            ServiceRegistry.get_checkpoint_scanner,
            ServiceRegistry.get_embedding_scanner,
        ):
            try:
                scanner = await getter()
                cache = await scanner.get_cached_data()
                for entry in cache.raw_data or []:
                    civitai = entry.get("civitai") if isinstance(entry, dict) else None
                    if isinstance(civitai, dict):
                        mid = civitai.get("modelId")
                        if mid is not None:
                            try:
                                ids.add(int(mid))
                            except (TypeError, ValueError):
                                pass
            except Exception as exc:
                logger.warning("Could not collect installed IDs from scanner: %s", exc)
        return web.json_response({"ids": list(ids)})
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/routes/test_discover_routes.py -v
```

Expected: All 5 tests PASS

- [ ] **Step 5: Commit**

```bash
git -C /home/blackthorn/ComfyUI add \
  custom_nodes/comfyui-lora-manager/py/routes/discover_routes.py \
  custom_nodes/comfyui-lora-manager/tests/routes/test_discover_routes.py
git -C /home/blackthorn/ComfyUI commit -m "feat(discover): add DiscoverRoutes with browse and installed-ids endpoints"
```

---

## Task 3: Wire Routes + Templates

**Files:**
- Modify: `py/lora_manager.py`
- Modify: `standalone.py`
- Modify: `templates/components/header.html`
- Create: `templates/discover.html`
- Create: `static/css/discover.css`

- [ ] **Step 1: Wire `DiscoverRoutes` into `py/lora_manager.py`**

Add import near line 27 (after the other route imports):

```python
from .routes.discover_routes import DiscoverRoutes
```

In `add_routes()`, after `MiscRoutes.setup_routes(app)` (around line 166):

```python
        discover_routes = DiscoverRoutes()
        discover_routes.setup_routes(app)
```

- [ ] **Step 2: Wire `DiscoverRoutes` into `standalone.py`**

Add import after `from py.routes.stats_routes import StatsRoutes` (around line 328):

```python
from py.routes.discover_routes import DiscoverRoutes
```

After `MiscRoutes.setup_routes(app)` (around line 342):

```python
        discover_routes = DiscoverRoutes()
        discover_routes.setup_routes(app)
```

- [ ] **Step 3: Add Discover nav item to `templates/components/header.html`**

In the `current_page` detection block (lines 12–22), add a new elif before `{% else %}`:

```html
      {% elif current_path.startswith('/discover') %}
      {% set current_page = 'discover' %}
```

In `<nav class="main-nav">`, add after the Statistics link (after line 42):

```html
        <a href="/discover" class="nav-item{% if current_path.startswith('/discover') %} active{% endif %}"
          id="discoverNavItem">
          <i class="fas fa-compass"></i> <span>Discover</span>
        </a>
```

Update the search-disabled condition (line 47) to also disable search on the Discover page:

Find:
```html
    {% set search_disabled = current_page == 'statistics' %}
```

Replace with:
```html
    {% set search_disabled = current_page == 'statistics' or current_page == 'discover' %}
```

- [ ] **Step 4: Create `templates/discover.html`**

```html
{% extends "base.html" %}

{% block title %}Discover — {{ t('header.appTitle') }}{% endblock %}
{% block page_id %}discover{% endblock %}

{% block content %}
<link rel="stylesheet" href="/loras_static/css/discover.css">
<div class="discover-container" id="discoverContainer">
    <div class="discover-filters" id="discoverFilters">
        <input type="text" id="discoverSearch" class="discover-search-input"
               placeholder="Search CivitAI models..." autocomplete="off">
        <select id="discoverTypeFilter" class="discover-select">
            <option value="">All Types</option>
        </select>
        <select id="discoverBaseModelFilter" class="discover-select">
            <option value="">All Base Models</option>
        </select>
    </div>
    <div class="discover-grid" id="discoverGrid"></div>
    <div class="discover-load-more" id="discoverLoadMore" style="display:none">
        <button id="loadMoreBtn" class="discover-load-more-btn">Load More</button>
    </div>
</div>
{% endblock %}

{% block main_script %}
<script type="module" src="/loras_static/js/discover.js"></script>
{% endblock %}
```

- [ ] **Step 5: Create `static/css/discover.css`**

```css
.discover-container {
    padding: 16px;
    max-width: 1600px;
    margin: 0 auto;
}

.discover-filters {
    display: flex;
    gap: 10px;
    margin-bottom: 18px;
    flex-wrap: wrap;
    align-items: center;
}

.discover-search-input {
    flex: 1 1 220px;
    min-width: 180px;
    padding: 8px 12px;
    border-radius: 6px;
    border: 1px solid var(--border-color, #ccc);
    background: var(--input-bg, #fff);
    color: var(--text-color, #222);
    font-size: 14px;
}

.discover-select {
    padding: 8px 10px;
    border-radius: 6px;
    border: 1px solid var(--border-color, #ccc);
    background: var(--input-bg, #fff);
    color: var(--text-color, #222);
    font-size: 14px;
    cursor: pointer;
}

.discover-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(180px, 1fr));
    gap: 14px;
}

.discover-card {
    position: relative;
    border-radius: 8px;
    overflow: hidden;
    background: var(--card-bg, #f5f5f5);
    box-shadow: 0 1px 4px rgba(0,0,0,0.12);
}

.discover-card-img {
    width: 100%;
    aspect-ratio: 2/3;
    object-fit: cover;
    display: block;
    background: var(--placeholder-bg, #ddd);
}

.discover-card-info {
    padding: 8px;
}

.discover-card-name {
    font-size: 13px;
    font-weight: 600;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
    margin-bottom: 2px;
}

.discover-card-creator {
    font-size: 11px;
    color: var(--text-muted, #888);
    margin-bottom: 4px;
}

.discover-card-tags {
    display: flex;
    flex-wrap: wrap;
    gap: 4px;
}

.discover-badge {
    font-size: 10px;
    padding: 2px 6px;
    border-radius: 4px;
    background: var(--badge-bg, #e0e0e0);
    color: var(--badge-color, #555);
}

.discover-badge-type {
    background: #3b5bdb22;
    color: #3b5bdb;
}

.discover-badge-base {
    background: #2f9e4422;
    color: #2f9e44;
}

.discover-installed-badge {
    position: absolute;
    top: 8px;
    right: 8px;
    background: #2f9e44;
    color: #fff;
    font-size: 11px;
    font-weight: 600;
    padding: 3px 8px;
    border-radius: 4px;
    pointer-events: none;
}

.discover-install-btn {
    position: absolute;
    bottom: 52px;
    right: 8px;
    background: #3b5bdb;
    color: #fff;
    border: none;
    border-radius: 5px;
    padding: 4px 10px;
    font-size: 12px;
    cursor: pointer;
    opacity: 0;
    transition: opacity 0.15s;
}

.discover-card:hover .discover-install-btn {
    opacity: 1;
}

.discover-error,
.discover-empty {
    grid-column: 1 / -1;
    text-align: center;
    padding: 48px 0;
    color: var(--text-muted, #888);
}

.discover-error p {
    margin-bottom: 12px;
    font-size: 15px;
}

.discover-error button,
.discover-load-more-btn {
    padding: 7px 18px;
    border-radius: 6px;
    border: 1px solid var(--border-color, #ccc);
    background: var(--btn-bg, #fff);
    cursor: pointer;
    font-size: 14px;
}

.discover-load-more {
    display: flex;
    justify-content: center;
    margin-top: 24px;
}
```

- [ ] **Step 6: Run the full backend test suite to catch any wiring regressions**

```bash
cd /home/blackthorn/ComfyUI/custom_nodes/comfyui-lora-manager && pytest -x -q 2>&1 | tail -20
```

Expected: All tests pass.

- [ ] **Step 7: Commit**

```bash
git -C /home/blackthorn/ComfyUI add \
  custom_nodes/comfyui-lora-manager/py/lora_manager.py \
  custom_nodes/comfyui-lora-manager/standalone.py \
  custom_nodes/comfyui-lora-manager/templates/components/header.html \
  custom_nodes/comfyui-lora-manager/templates/discover.html \
  custom_nodes/comfyui-lora-manager/static/css/discover.css
git -C /home/blackthorn/ComfyUI commit -m "feat(discover): wire routes, add nav item, template, and CSS"
```

---

## Task 4: `discover.js` — Card Grid + Install Trigger

**Files:**
- Create: `static/js/discover.js`
- Create: `tests/frontend/pages/discover.test.js`

- [ ] **Step 1: Write the failing frontend tests**

Create `tests/frontend/pages/discover.test.js`:

```js
import { beforeEach, describe, expect, it, vi } from 'vitest';

vi.mock('../../../static/js/managers/DownloadManager.js', () => ({
    downloadManager: { openForModelVersion: vi.fn() },
}));

let buildCard, installModel;

beforeEach(async () => {
    vi.resetModules();
    vi.mock('../../../static/js/managers/DownloadManager.js', () => ({
        downloadManager: { openForModelVersion: vi.fn() },
    }));
    const mod = await import('../../../static/js/discover.js');
    buildCard = mod.buildCard;
    installModel = mod.installModel;
    vi.clearAllMocks();
});

describe('buildCard', () => {
    it('renders card name and creator', () => {
        const model = {
            id: 100,
            name: 'My LoRA',
            type: 'LORA',
            creator: { username: 'artist123' },
            modelVersions: [{ baseModel: 'Flux.1 D', images: [] }],
        };
        const card = buildCard(model, new Set());
        expect(card.querySelector('.discover-card-name').textContent).toBe('My LoRA');
        expect(card.querySelector('.discover-card-creator').textContent).toBe('artist123');
    });

    it('shows installed badge when model ID is in installed set', () => {
        const model = {
            id: 100,
            name: 'Installed Model',
            type: 'LORA',
            creator: { username: 'u' },
            modelVersions: [],
        };
        const card = buildCard(model, new Set([100]));
        expect(card.querySelector('.discover-installed-badge')).not.toBeNull();
        expect(card.querySelector('.discover-install-btn')).toBeNull();
    });

    it('shows install button when model ID is NOT in installed set', () => {
        const model = {
            id: 999,
            name: 'New Model',
            type: 'LORA',
            creator: { username: 'u' },
            modelVersions: [],
        };
        const card = buildCard(model, new Set([100]));
        expect(card.querySelector('.discover-install-btn')).not.toBeNull();
        expect(card.querySelector('.discover-installed-badge')).toBeNull();
    });

    it('uses first non-video image from first version as preview', () => {
        const model = {
            id: 1,
            name: 'M',
            type: 'LORA',
            creator: { username: 'u' },
            modelVersions: [{
                baseModel: 'SDXL',
                images: [{ url: 'https://example.com/preview.jpg' }],
            }],
        };
        const card = buildCard(model, new Set());
        expect(card.querySelector('.discover-card-img').src).toContain('preview.jpg');
    });

    it('falls back to placeholder when no images available', () => {
        const model = {
            id: 2,
            name: 'M',
            type: 'LORA',
            creator: {},
            modelVersions: [{ images: [] }],
        };
        const card = buildCard(model, new Set());
        expect(card.querySelector('.discover-card-img').src).toContain('no-preview');
    });
});

describe('installModel', () => {
    it('maps LORA to "loras" and calls openForModelVersion', async () => {
        const { downloadManager } = await import('../../../static/js/managers/DownloadManager.js');
        installModel({ id: 42, type: 'LORA' });
        expect(downloadManager.openForModelVersion).toHaveBeenCalledWith('loras', 42);
    });

    it('maps Checkpoint to "checkpoints"', async () => {
        const { downloadManager } = await import('../../../static/js/managers/DownloadManager.js');
        installModel({ id: 99, type: 'Checkpoint' });
        expect(downloadManager.openForModelVersion).toHaveBeenCalledWith('checkpoints', 99);
    });

    it('maps TextualInversion to "embeddings"', async () => {
        const { downloadManager } = await import('../../../static/js/managers/DownloadManager.js');
        installModel({ id: 55, type: 'TextualInversion' });
        expect(downloadManager.openForModelVersion).toHaveBeenCalledWith('embeddings', 55);
    });

    it('falls back to "loras" for unmapped types', async () => {
        const { downloadManager } = await import('../../../static/js/managers/DownloadManager.js');
        installModel({ id: 77, type: 'Poses' });
        expect(downloadManager.openForModelVersion).toHaveBeenCalledWith('loras', 77);
    });
});
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd /home/blackthorn/ComfyUI/custom_nodes/comfyui-lora-manager
npm run test:js -- tests/frontend/pages/discover.test.js
```

Expected: FAIL — module not found

- [ ] **Step 3: Create `static/js/discover.js`**

All DOM construction uses `createElement`/`textContent`/`appendChild`. Element clearing uses `replaceChildren()`.

```js
import { downloadManager } from './managers/DownloadManager.js';

const MODEL_TYPE_MAP = {
    LORA: 'loras',
    LoCon: 'loras',
    DoRA: 'loras',
    Lycoris: 'loras',
    Checkpoint: 'checkpoints',
    CheckpointConfig: 'checkpoints',
    TextualInversion: 'embeddings',
    Controlnet: 'loras',
    Hypernetwork: 'loras',
    AestheticGradient: 'loras',
    Poses: 'loras',
    MotionModule: 'loras',
    TextEncoder: 'loras',
    Upscaler: 'loras',
    VAE: 'checkpoints',
    Wildcards: 'loras',
    Detection: 'loras',
    Other: 'loras',
};

const CIVITAI_TYPES = Object.keys(MODEL_TYPE_MAP);

const BASE_MODELS = [
    'Flux.1 D', 'Flux.1 S', 'SDXL 1.0', 'SDXL 0.9', 'SD 1.5',
    'SD 2.1', 'SD 3', 'SD 3.5', 'SD 3.5 Large', 'Pony',
    'Illustrious', 'NoobAI', 'Wan Video',
];

// ── Pure exported functions (testable) ──────────────────────────────────────

export function buildCard(model, installedIds) {
    const card = document.createElement('div');
    card.className = 'discover-card';
    card.dataset.modelId = model.id;

    const version = (model.modelVersions || [])[0] || {};
    const image = (version.images || []).find(img => img.url && !img.url.endsWith('.mp4'));
    const previewUrl = image ? image.url : '/loras_static/images/no-preview.png';

    const img = document.createElement('img');
    img.src = previewUrl;
    img.alt = '';
    img.className = 'discover-card-img';
    img.loading = 'lazy';
    card.appendChild(img);

    const info = document.createElement('div');
    info.className = 'discover-card-info';

    const nameEl = document.createElement('div');
    nameEl.className = 'discover-card-name';
    nameEl.textContent = model.name || '';
    info.appendChild(nameEl);

    const creatorEl = document.createElement('div');
    creatorEl.className = 'discover-card-creator';
    creatorEl.textContent = model.creator?.username || '';
    info.appendChild(creatorEl);

    const tagsEl = document.createElement('div');
    tagsEl.className = 'discover-card-tags';
    if (model.type) {
        const tb = document.createElement('span');
        tb.className = 'discover-badge discover-badge-type';
        tb.textContent = model.type;
        tagsEl.appendChild(tb);
    }
    if (version.baseModel) {
        const bb = document.createElement('span');
        bb.className = 'discover-badge discover-badge-base';
        bb.textContent = version.baseModel;
        tagsEl.appendChild(bb);
    }
    info.appendChild(tagsEl);
    card.appendChild(info);

    if (installedIds.has(model.id)) {
        const badge = document.createElement('div');
        badge.className = 'discover-installed-badge';
        badge.textContent = 'Installed';
        card.appendChild(badge);
    } else {
        const btn = document.createElement('button');
        btn.className = 'discover-install-btn';
        btn.textContent = 'Install';
        btn.addEventListener('click', (e) => {
            e.stopPropagation();
            installModel(model);
        });
        card.appendChild(btn);
    }

    return card;
}

export function installModel(model) {
    const modelType = MODEL_TYPE_MAP[model.type] || 'loras';
    downloadManager.openForModelVersion(modelType, model.id);
}

export function renderError(grid, status) {
    const wrap = document.createElement('div');
    wrap.className = 'discover-error';

    const msg = document.createElement('p');
    if (status === 429) {
        msg.textContent = 'CivitAI rate limit hit — please wait a moment';
    } else if (status === 401) {
        msg.textContent = 'CivitAI API key invalid — check Settings';
    } else {
        msg.textContent = 'Could not reach CivitAI — check your connection';
    }

    const retryBtn = document.createElement('button');
    retryBtn.textContent = 'Retry';
    retryBtn.addEventListener('click', () => loadPage(1, false));

    wrap.appendChild(msg);
    wrap.appendChild(retryBtn);
    grid.replaceChildren(wrap);
}

export function renderEmpty(grid) {
    const el = document.createElement('div');
    el.className = 'discover-empty';
    el.textContent = 'No models found — try different filters';
    grid.replaceChildren(el);
}

// ── Module state ─────────────────────────────────────────────────────────────

let installedIds = new Set();
let currentPage = 1;
let isLoading = false;
let debounceTimer = null;

async function fetchInstalledIds() {
    try {
        const res = await fetch('/api/lm/discover/installed-ids');
        if (!res.ok) return;
        const data = await res.json();
        installedIds = new Set(data.ids || []);
    } catch (_) {
        // non-fatal
    }
}

async function loadPage(page, append) {
    if (isLoading) return;
    isLoading = true;

    const grid = document.getElementById('discoverGrid');
    const typeVal = document.getElementById('discoverTypeFilter').value;
    const baseModelVal = document.getElementById('discoverBaseModelFilter').value;
    const queryVal = document.getElementById('discoverSearch').value.trim();

    const params = new URLSearchParams({ page, limit: 20 });
    if (typeVal) params.set('type', typeVal);
    if (baseModelVal) params.set('baseModel', baseModelVal);
    if (queryVal) params.set('query', queryVal);

    try {
        const res = await fetch(`/api/lm/discover/browse?${params}`);
        const data = await res.json();

        if (!res.ok) {
            renderError(grid, res.status);
            isLoading = false;
            return;
        }

        const items = data.items || [];

        if (!append) {
            grid.replaceChildren();
        }

        if (items.length === 0 && !append) {
            renderEmpty(grid);
            isLoading = false;
            return;
        }

        items.forEach(model => grid.appendChild(buildCard(model, installedIds)));

        const meta = data.metadata || {};
        const loadMoreEl = document.getElementById('discoverLoadMore');
        if (meta.currentPage < meta.totalPages) {
            currentPage = meta.currentPage;
            loadMoreEl.style.display = 'flex';
        } else {
            loadMoreEl.style.display = 'none';
        }
    } catch (_) {
        renderError(grid, 503);
    } finally {
        isLoading = false;
    }
}

function populateFilters() {
    const typeSelect = document.getElementById('discoverTypeFilter');
    CIVITAI_TYPES.forEach(t => {
        const opt = document.createElement('option');
        opt.value = t;
        opt.textContent = t;
        typeSelect.appendChild(opt);
    });

    const baseSelect = document.getElementById('discoverBaseModelFilter');
    BASE_MODELS.forEach(bm => {
        const opt = document.createElement('option');
        opt.value = bm;
        opt.textContent = bm;
        baseSelect.appendChild(opt);
    });
}

function onFilterChange() {
    currentPage = 1;
    loadPage(1, false);
}

function init() {
    populateFilters();
    document.getElementById('discoverTypeFilter').addEventListener('change', onFilterChange);
    document.getElementById('discoverBaseModelFilter').addEventListener('change', onFilterChange);
    document.getElementById('discoverSearch').addEventListener('input', () => {
        clearTimeout(debounceTimer);
        debounceTimer = setTimeout(onFilterChange, 400);
    });
    document.getElementById('loadMoreBtn').addEventListener('click', () => {
        loadPage(currentPage + 1, true);
    });
    fetchInstalledIds().then(() => loadPage(1, false));
}

// Guard: only auto-run when the Discover page DOM is present
if (document.getElementById('discoverGrid')) {
    init();
}
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
npm run test:js -- tests/frontend/pages/discover.test.js
```

Expected: All tests PASS

- [ ] **Step 5: Commit**

```bash
git -C /home/blackthorn/ComfyUI add \
  custom_nodes/comfyui-lora-manager/static/js/discover.js \
  custom_nodes/comfyui-lora-manager/tests/frontend/pages/discover.test.js
git -C /home/blackthorn/ComfyUI commit -m "feat(discover): add discover.js card grid, install trigger, and frontend tests"
```

---

## Task 5: Full Test Run + Manual Smoke Test

- [ ] **Step 1: Run all backend tests**

```bash
cd /home/blackthorn/ComfyUI/custom_nodes/comfyui-lora-manager && pytest -x -q 2>&1 | tail -20
```

Expected: All tests PASS

- [ ] **Step 2: Run all frontend tests**

```bash
cd /home/blackthorn/ComfyUI/custom_nodes/comfyui-lora-manager && npm test 2>&1 | tail -20
```

Expected: All tests PASS

- [ ] **Step 3: Start the standalone server**

```bash
cd /home/blackthorn/ComfyUI/custom_nodes/comfyui-lora-manager && python standalone.py --port 8199
```

- [ ] **Step 4: Manual verification checklist**

Open `http://localhost:8199` in a browser.

- [ ] "Discover" tab is visible in the nav bar
- [ ] Clicking the Discover tab loads the page with a filter bar and card grid
- [ ] Default load shows trending models (cards with images, names, type/base-model badges)
- [ ] Models already installed (CivitAI model ID in local cache) show a green "Installed" badge
- [ ] Models not installed show an "Install" button on hover
- [ ] Clicking "Install" opens the existing DownloadManager modal with versions pre-fetched
- [ ] Type dropdown filters results (e.g. selecting "LORA" shows only LoRAs)
- [ ] Base model dropdown filters results
- [ ] Typing in the search box waits ~400ms then updates the grid
- [ ] "Load More" button appears when there are additional pages and appends cards without clearing the grid
- [ ] Header search bar is disabled/greyed out on the Discover page (consistent with Statistics page)

- [ ] **Step 5: Commit any fixes found during smoke test**

```bash
git -C /home/blackthorn/ComfyUI add -p
git -C /home/blackthorn/ComfyUI commit -m "fix(discover): address issues found during smoke test"
```

Stop the standalone server with `Ctrl+C`.
