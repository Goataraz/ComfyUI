# Discover Tab — Model Detail Modal Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Clicking any Discover tab card opens the rich ModelModal showing full CivitAI metadata, with an Install CTA in place of management actions.

**Architecture:** A new backend proxy endpoint fetches the full CivitAI model detail response. A new `showModelModalForCivitaiId` function in `ModelModal.js` normalizes the response and feeds the existing render path via a new `isDiscoverModel` flag that swaps management UI for an Install button and hides the Recipes tab. Discover cards get `data-civitai-id`/`data-model-type` attributes and a delegated click listener on the grid.

**Tech Stack:** Python/aiohttp (backend), vanilla JS ES modules (frontend), pytest + vitest (tests). DOM manipulation uses only safe APIs — `createElement`, `textContent`, `appendChild`, `replaceChildren`. No direct HTML string injection into the DOM.

---

### Task 1: `CivitaiClient.get_model_detail` + tests

**Files:**
- Modify: `py/services/civitai_client.py` (append after line 849)
- Test: `tests/services/test_civitai_browse.py`

- [ ] **Step 1: Write three failing tests**

In `tests/services/test_civitai_browse.py`, append after the last test:

```python
@pytest.mark.asyncio
async def test_get_model_detail_success(downloader):
    downloader.make_request = AsyncMock(
        return_value=(True, {"id": 123, "name": "Test Model", "modelVersions": []})
    )
    client = await CivitaiClient.get_instance()
    result = await client.get_model_detail("123")
    assert result["id"] == 123
    assert result["name"] == "Test Model"


@pytest.mark.asyncio
async def test_get_model_detail_returns_error_dict_on_failure(downloader):
    downloader.make_request = AsyncMock(return_value=(False, "404 Not Found"))
    client = await CivitaiClient.get_instance()
    result = await client.get_model_detail("999")
    assert "error" in result
    assert "404" in result["error"]


@pytest.mark.asyncio
async def test_get_model_detail_reraises_rate_limit_error(downloader):
    from py.services.errors import RateLimitError
    downloader.make_request = AsyncMock(side_effect=RateLimitError("rate limited"))
    client = await CivitaiClient.get_instance()
    with pytest.raises(RateLimitError):
        await client.get_model_detail("123")
```

- [ ] **Step 2: Run tests to verify they fail**

```
cd /home/blackthorn/ComfyUI/custom_nodes/comfyui-lora-manager && pytest tests/services/test_civitai_browse.py::test_get_model_detail_success -v
```
Expected: FAIL with `AttributeError: 'CivitaiClient' object has no attribute 'get_model_detail'`

- [ ] **Step 3: Implement `get_model_detail` in `py/services/civitai_client.py`**

Append to the end of the file (after line 849):

```python

    async def get_model_detail(self, model_id: str) -> dict:
        try:
            success, result = await self._make_request(
                "GET",
                f"{self.base_url}/models/{model_id}",
                use_auth=True,
            )
            if not success:
                return {"error": str(result)}
            return result if isinstance(result, dict) else {"error": "Invalid response"}
        except RateLimitError:
            raise
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.error("Error in get_model_detail: %s", exc)
            return {"error": str(exc)}
```

- [ ] **Step 4: Run all tests in the file to verify they pass**

```
cd /home/blackthorn/ComfyUI/custom_nodes/comfyui-lora-manager && pytest tests/services/test_civitai_browse.py -v
```
Expected: all 9 tests pass

- [ ] **Step 5: Commit**

```bash
git -C /home/blackthorn/ComfyUI add -f \
    custom_nodes/comfyui-lora-manager/py/services/civitai_client.py \
    custom_nodes/comfyui-lora-manager/tests/services/test_civitai_browse.py
git -C /home/blackthorn/ComfyUI commit -m "feat: add CivitaiClient.get_model_detail method"
```

---

### Task 2: `/api/lm/discover/model/{model_id}` route + tests

**Files:**
- Modify: `py/routes/discover_routes.py`
- Test: `tests/routes/test_discover_routes.py`

- [ ] **Step 1: Write four failing tests**

In `tests/routes/test_discover_routes.py`, append after the last test:

```python
@pytest.mark.asyncio
async def test_get_model_detail_returns_payload(monkeypatch):
    payload = {"id": 123, "name": "Test Model", "modelVersions": []}
    mock_client = AsyncMock()
    mock_client.get_model_detail = AsyncMock(return_value=payload)
    monkeypatch.setattr(
        discover_module.CivitaiClient, "get_instance", AsyncMock(return_value=mock_client)
    )
    routes = DiscoverRoutes()
    request = make_mocked_request(
        "GET", "/api/lm/discover/model/123", match_info={"model_id": "123"}
    )
    response = await routes.get_model_detail(request)
    assert response.status == 200
    assert json.loads(response.body)["id"] == 123


@pytest.mark.asyncio
async def test_get_model_detail_returns_404_for_not_found(monkeypatch):
    mock_client = AsyncMock()
    mock_client.get_model_detail = AsyncMock(return_value={"error": "404 Not Found"})
    monkeypatch.setattr(
        discover_module.CivitaiClient, "get_instance", AsyncMock(return_value=mock_client)
    )
    routes = DiscoverRoutes()
    request = make_mocked_request(
        "GET", "/api/lm/discover/model/999", match_info={"model_id": "999"}
    )
    response = await routes.get_model_detail(request)
    assert response.status == 404


@pytest.mark.asyncio
async def test_get_model_detail_returns_429_on_rate_limit(monkeypatch):
    from py.services.errors import RateLimitError
    mock_client = AsyncMock()
    mock_client.get_model_detail = AsyncMock(side_effect=RateLimitError("rate limited"))
    monkeypatch.setattr(
        discover_module.CivitaiClient, "get_instance", AsyncMock(return_value=mock_client)
    )
    routes = DiscoverRoutes()
    request = make_mocked_request(
        "GET", "/api/lm/discover/model/123", match_info={"model_id": "123"}
    )
    response = await routes.get_model_detail(request)
    assert response.status == 429


@pytest.mark.asyncio
async def test_get_model_detail_returns_503_on_generic_error(monkeypatch):
    mock_client = AsyncMock()
    mock_client.get_model_detail = AsyncMock(return_value={"error": "connection refused"})
    monkeypatch.setattr(
        discover_module.CivitaiClient, "get_instance", AsyncMock(return_value=mock_client)
    )
    routes = DiscoverRoutes()
    request = make_mocked_request(
        "GET", "/api/lm/discover/model/123", match_info={"model_id": "123"}
    )
    response = await routes.get_model_detail(request)
    assert response.status == 503
```

- [ ] **Step 2: Run one test to confirm it fails**

```
cd /home/blackthorn/ComfyUI/custom_nodes/comfyui-lora-manager && pytest tests/routes/test_discover_routes.py::test_get_model_detail_returns_payload -v
```
Expected: FAIL with `AttributeError: 'DiscoverRoutes' object has no attribute 'get_model_detail'`

- [ ] **Step 3: Register the route in `setup_routes`**

In `py/routes/discover_routes.py`, inside the `setup_routes` method after the existing three `add_get` calls, add:

```python
        app.router.add_get("/api/lm/discover/model/{model_id}", self.get_model_detail)
```

- [ ] **Step 4: Add the `get_model_detail` handler method**

In `py/routes/discover_routes.py`, add after `get_installed_ids` (after line 115):

```python
    async def get_model_detail(self, request: web.Request) -> web.Response:
        model_id = request.match_info["model_id"]
        try:
            client = await CivitaiClient.get_instance()
            result = await client.get_model_detail(model_id)
        except RateLimitError:
            return web.json_response({"error": "Rate limited by CivitAI"}, status=429)

        if "error" in result:
            error_str = str(result["error"])
            if "401" in error_str:
                return web.json_response({"error": "Authentication required"}, status=401)
            if "404" in error_str:
                return web.json_response({"error": "Model not found"}, status=404)
            return web.json_response({"error": error_str}, status=503)
        return web.json_response(result)
```

- [ ] **Step 5: Run all route tests**

```
cd /home/blackthorn/ComfyUI/custom_nodes/comfyui-lora-manager && pytest tests/routes/test_discover_routes.py -v
```
Expected: all 12 tests pass

- [ ] **Step 6: Run full backend suite**

```
cd /home/blackthorn/ComfyUI/custom_nodes/comfyui-lora-manager && pytest -v 2>&1 | tail -20
```
Expected: only the pre-existing `test_bulk_metadata_refresh_emits_progress_and_updates_cache` failure; everything else passes.

- [ ] **Step 7: Commit**

```bash
git -C /home/blackthorn/ComfyUI add -f \
    custom_nodes/comfyui-lora-manager/py/routes/discover_routes.py \
    custom_nodes/comfyui-lora-manager/tests/routes/test_discover_routes.py
git -C /home/blackthorn/ComfyUI commit -m "feat: add GET /api/lm/discover/model/{model_id} proxy endpoint"
```

---

### Task 3: CSS additions

**Files:**
- Modify: `static/css/discover.css`
- Modify: `static/css/components/lora-modal/lora-modal.css`

- [ ] **Step 1: Add card spinner styles to `static/css/discover.css`**

Append to the end of the file:

```css
/* Card loading spinner while fetching model detail */
.discover-card {
    position: relative;
}

.discover-card-spinner {
    position: absolute;
    inset: 0;
    display: flex;
    align-items: center;
    justify-content: center;
    background: color-mix(in oklch, var(--lora-bg, #1a1a2e) 70%, transparent);
    border-radius: inherit;
    font-size: 1.5rem;
    color: var(--lora-accent);
    z-index: 2;
    pointer-events: none;
}
```

- [ ] **Step 2: Add Install button styles to `static/css/components/lora-modal/lora-modal.css`**

Find the `.modal-send-btn:active` rule (around line 953) and add after its closing brace `}`:

```css

/* Install button in Discover modal — replaces Send-to-ComfyUI */
.modal-install-btn {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 6px 14px;
    background: var(--lora-accent);
    border: 1px solid var(--lora-accent);
    border-radius: var(--border-radius-sm);
    color: #fff;
    cursor: pointer;
    font-weight: 600;
    font-size: 0.9em;
    transition: all 0.2s;
}

.modal-install-btn:hover {
    opacity: 0.88;
    transform: translateY(-1px);
}

.modal-install-btn:active {
    transform: translateY(0);
}
```

- [ ] **Step 3: Commit**

```bash
git -C /home/blackthorn/ComfyUI add -f \
    custom_nodes/comfyui-lora-manager/static/css/discover.css \
    custom_nodes/comfyui-lora-manager/static/css/components/lora-modal/lora-modal.css
git -C /home/blackthorn/ComfyUI commit -m "style: add discover card spinner and modal install button styles"
```

---

### Task 4: Extend `showModelModal` for Discover mode

**Files:**
- Modify: `static/js/components/shared/ModelModal.js`

Changes tested indirectly by Task 6. No standalone tests needed — the modifications are guarded branches, not new logic.

- [ ] **Step 1: Change `showModelModal` signature (line 242)**

Replace:
```js
export async function showModelModal(model, modelType) {
```
With:
```js
export async function showModelModal(model, modelType, { isDiscoverModel = false, onInstall = null } = {}) {
```

- [ ] **Step 2: Replace the Send-to-ComfyUI button block with a conditional**

Find this block (around lines 300–308):
```js
    // Add send to ComfyUI button for all model types
    const sendToWorkflowTitle = translate('modals.model.actions.sendToWorkflow', {}, 'Send to ComfyUI');
    const sendToWorkflowButton = `
        <button class="modal-send-btn" data-action="send-to-workflow" data-model-type="${modelType}" title="${sendToWorkflowTitle}">
            <i class="fas fa-paper-plane"></i>
            <span>${translate('modals.model.actions.sendToWorkflowText', {}, 'Send to ComfyUI')}</span>
        </button>
    `.trim();
    headerActionItems.push(indentMarkup(sendToWorkflowButton, 20));
```

Replace with:
```js
    if (!isDiscoverModel) {
        const sendToWorkflowTitle = translate('modals.model.actions.sendToWorkflow', {}, 'Send to ComfyUI');
        const sendToWorkflowButton = `
            <button class="modal-send-btn" data-action="send-to-workflow" data-model-type="${modelType}" title="${sendToWorkflowTitle}">
                <i class="fas fa-paper-plane"></i>
                <span>${translate('modals.model.actions.sendToWorkflowText', {}, 'Send to ComfyUI')}</span>
            </button>
        `.trim();
        headerActionItems.push(indentMarkup(sendToWorkflowButton, 20));
    } else {
        const installTitle = translate('modals.model.actions.install', {}, 'Install model');
        const installButton = `
            <button class="modal-install-btn" data-action="install-discover-model" title="${installTitle}">
                <i class="fas fa-download"></i>
                <span>${translate('modals.model.actions.installText', {}, 'Install')}</span>
            </button>
        `.trim();
        headerActionItems.push(indentMarkup(installButton, 20));
    }
```

- [ ] **Step 3: Force `supportsRecipesTab` to false for Discover models (line 359)**

Replace:
```js
    const supportsRecipesTab = modelType === 'loras' || modelType === 'checkpoints';
```
With:
```js
    const supportsRecipesTab = !isDiscoverModel && (modelType === 'loras' || modelType === 'checkpoints');
```

- [ ] **Step 4: Conditionally hide the edit pencil and navigation controls in the content template**

In the `content` template string, find:
```js
                    <div class="model-name-header">
                        <h2 class="model-name-content">${modalTitle}</h2>
                        <button class="edit-model-name-btn" title="${translate('modals.model.actions.editModelName', {}, 'Edit model name')}">
                            <i class="fas fa-pencil-alt"></i>
                        </button>
                    </div>

                    ${navigationControls}
```

Replace with:
```js
                    <div class="model-name-header">
                        <h2 class="model-name-content">${modalTitle}</h2>
                        ${!isDiscoverModel ? `<button class="edit-model-name-btn" title="${translate('modals.model.actions.editModelName', {}, 'Edit model name')}">
                            <i class="fas fa-pencil-alt"></i>
                        </button>` : ''}
                    </div>

                    ${isDiscoverModel ? '' : navigationControls}
```

- [ ] **Step 5: Guard all edit/management setup calls and update `setupEventHandlers` call**

Find and replace the post-render setup block (around lines 656–693).

Old block (from `setupTagTooltip()` through the `loadRecipesForModel` calls):
```js
    setupTagTooltip();
    setupTagEditMode(modelType);
    setupModelNameEditing(modelWithFullData.file_path);
    setupVersionNameEditing(modelWithFullData.file_path);
    setupBaseModelEditing(modelWithFullData.file_path);
    setupFileNameEditing(modelWithFullData.file_path);
    setupEventHandlers(modelWithFullData.file_path, modelType);
    setupNavigationShortcuts(modelType);
    updateNavigationControls();

    // Model-specific setup
    if (modelType === 'loras' || modelType === 'embeddings') {
        setupTriggerWordsEditMode();
    }

    if (modelType === 'loras') {
        loadRecipesForModel({
            modelKind: 'lora',
            displayName: modelWithFullData.model_name,
            sha256: modelWithFullData.sha256,
        });
    } else if (modelType === 'checkpoints') {
        loadRecipesForModel({
            modelKind: 'checkpoint',
            displayName: modelWithFullData.model_name,
            sha256: modelWithFullData.sha256,
        });
    }
```

New block:
```js
    setupTagTooltip();
    setupEventHandlers(modelWithFullData.file_path, modelType, { onInstall });
    if (!isDiscoverModel) {
        setupEditableFields(modelWithFullData.file_path, modelType);
        setupTagEditMode(modelType);
        setupModelNameEditing(modelWithFullData.file_path);
        setupVersionNameEditing(modelWithFullData.file_path);
        setupBaseModelEditing(modelWithFullData.file_path);
        setupFileNameEditing(modelWithFullData.file_path);
        setupNavigationShortcuts(modelType);
        updateNavigationControls();
        if (modelType === 'loras' || modelType === 'embeddings') {
            setupTriggerWordsEditMode();
        }
        if (modelType === 'loras') {
            loadRecipesForModel({
                modelKind: 'lora',
                displayName: modelWithFullData.model_name,
                sha256: modelWithFullData.sha256,
            });
        } else if (modelType === 'checkpoints') {
            loadRecipesForModel({
                modelKind: 'checkpoint',
                displayName: modelWithFullData.model_name,
                sha256: modelWithFullData.sha256,
            });
        }
    }
```

Also remove the standalone `setupEditableFields` call that appears just before `setupTagTooltip` (around line 656), if it exists — it is now inside the guard above.

- [ ] **Step 6: Add `onInstall` parameter to `setupEventHandlers` and wire the new action**

Change the function signature (line ~747):
```js
function setupEventHandlers(filePath, modelType, { onInstall = null } = {}) {
```

Inside the `switch (action)` block, add a new case after `'send-to-workflow'`:
```js
            case 'install-discover-model':
                if (onInstall) onInstall();
                break;
```

- [ ] **Step 7: Commit**

```bash
git -C /home/blackthorn/ComfyUI add -f \
    custom_nodes/comfyui-lora-manager/static/js/components/shared/ModelModal.js
git -C /home/blackthorn/ComfyUI commit -m "feat: extend showModelModal with isDiscoverModel flag and onInstall callback"
```

---

### Task 5: `showModelModalForCivitaiId` + normalization helper

**Files:**
- Modify: `static/js/components/shared/ModelModal.js` (insert before the exports block at end of file)

- [ ] **Step 1: Add `normalizeCivitaiModelDetail` and `showModelModalForCivitaiId`**

Find the exports block at the very end of `ModelModal.js`:
```js
const modelModal = {
    show: showModelModal,
    toggleShowcase,
    scrollToTop
};

export { modelModal };
```

Insert the following two functions **before** that block:

```js
function normalizeCivitaiModelDetail(civitaiModel) {
    const firstVersion = (civitaiModel.modelVersions || [])[0] || {};
    const tags = (civitaiModel.tags || [])
        .map(t => (typeof t === 'string' ? t : t?.name || ''))
        .filter(Boolean);
    return {
        model_name: civitaiModel.name || '',
        file_path: null,
        from_civitai: true,
        base_model: firstVersion.baseModel || '',
        sha256: null,
        tags,
        update_available: false,
        civitai: {
            modelId: civitaiModel.id,
            id: firstVersion.id,
            creator: civitaiModel.creator,
            trainedWords: firstVersion.trainedWords || [],
            images: firstVersion.images || [],
            description: firstVersion.description || civitaiModel.description || '',
            model: {
                allowCommercialUse: civitaiModel.allowCommercialUse,
                allowDerivatives: civitaiModel.allowDerivatives,
                allowNoCredit: civitaiModel.allowNoCredit,
                allowDifferentLicense: civitaiModel.allowDifferentLicense,
            },
        },
    };
}

export async function showModelModalForCivitaiId(civitaiId, modelType, { onInstall = null } = {}) {
    const card = document.querySelector(`.discover-card[data-civitai-id="${civitaiId}"]`);
    let spinner = null;
    if (card) {
        spinner = document.createElement('div');
        spinner.className = 'discover-card-spinner';
        const icon = document.createElement('i');
        icon.className = 'fas fa-spinner fa-spin';
        spinner.appendChild(icon);
        card.appendChild(spinner);
    }

    try {
        const res = await fetch(`/api/lm/discover/model/${civitaiId}`);
        const data = await res.json();

        if (!res.ok) {
            const errorMap = {
                404: ['discover.errors.notFound', 'Model not found on CivitAI'],
                429: ['discover.errors.rateLimit', 'CivitAI is rate limiting requests, try again in a moment'],
            };
            const [key, fallback] = errorMap[res.status] || ['discover.errors.network', 'Could not reach CivitAI — check your connection'];
            showToast(key, {}, 'error', fallback);
            return;
        }

        const model = normalizeCivitaiModelDetail(data);
        await showModelModal(model, modelType, { isDiscoverModel: true, onInstall });
    } catch (_) {
        showToast('discover.errors.network', {}, 'error', 'Could not reach CivitAI — check your connection');
    } finally {
        if (spinner) spinner.remove();
    }
}
```

- [ ] **Step 2: Commit**

```bash
git -C /home/blackthorn/ComfyUI add -f \
    custom_nodes/comfyui-lora-manager/static/js/components/shared/ModelModal.js
git -C /home/blackthorn/ComfyUI commit -m "feat: add showModelModalForCivitaiId and normalizeCivitaiModelDetail"
```

---

### Task 6: Discover card wiring + tests

**Files:**
- Modify: `static/js/discover.js`
- Test: `tests/frontend/pages/discover.test.js`

- [ ] **Step 1: Write failing tests**

In `tests/frontend/pages/discover.test.js`:

Add at the **top of the file** (after the existing top-level `vi.mock` for DownloadManager):
```js
vi.mock('../../../static/js/components/shared/ModelModal.js', () => ({
    showModelModalForCivitaiId: vi.fn(),
}));
```

Add at the **end of the file** (a new top-level `describe` block):
```js
describe('card click behavior', () => {
    beforeEach(async () => {
        vi.resetModules();
        vi.mock('../../../static/js/managers/DownloadManager.js', () => ({
            downloadManager: { openForModelVersion: vi.fn() },
        }));
        vi.mock('../../../static/js/components/shared/ModelModal.js', () => ({
            showModelModalForCivitaiId: vi.fn(),
        }));
        global.fetch = vi.fn().mockResolvedValue({
            ok: true,
            json: async () => ({ ids: [], items: [], metadata: { currentPage: 1, totalPages: 1 } }),
        });
        document.body.innerHTML = `
            <div id="discoverGrid"></div>
            <select id="discoverTypeFilter"></select>
            <select id="discoverBaseModelFilter"></select>
            <input id="discoverSearch">
            <div id="discoverLoadMore" style="display:none"><button id="loadMoreBtn"></button></div>
        `;
    });

    it('card body click calls showModelModalForCivitaiId with correct args', async () => {
        const { showModelModalForCivitaiId } = await import('../../../static/js/components/shared/ModelModal.js');
        const { buildCard } = await import('../../../static/js/discover.js');

        const model = {
            id: 555,
            name: 'ClickMe',
            type: 'LORA',
            creator: { username: 'u' },
            modelVersions: [],
        };
        const card = buildCard(model, new Set());
        document.getElementById('discoverGrid').appendChild(card);

        card.querySelector('.discover-card-info').click();

        expect(showModelModalForCivitaiId).toHaveBeenCalledWith(
            '555',
            'loras',
            expect.objectContaining({ onInstall: expect.any(Function) })
        );
    });

    it('install button click does NOT call showModelModalForCivitaiId', async () => {
        const { showModelModalForCivitaiId } = await import('../../../static/js/components/shared/ModelModal.js');
        const { buildCard } = await import('../../../static/js/discover.js');

        const model = {
            id: 777,
            name: 'NewModel',
            type: 'LORA',
            creator: { username: 'u' },
            modelVersions: [],
        };
        const card = buildCard(model, new Set());
        document.getElementById('discoverGrid').appendChild(card);

        card.querySelector('.discover-install-btn').click();

        expect(showModelModalForCivitaiId).not.toHaveBeenCalled();
    });
});
```

- [ ] **Step 2: Run tests to verify they fail**

```
cd /home/blackthorn/ComfyUI/custom_nodes/comfyui-lora-manager && npm run test:js -- --reporter verbose 2>&1 | grep -A 10 "card click behavior"
```
Expected: both new tests FAIL — `showModelModalForCivitaiId` not called because the listener is not yet wired.

- [ ] **Step 3: Add import at the top of `discover.js`**

Replace the first line of `static/js/discover.js`:
```js
import { downloadManager } from './managers/DownloadManager.js';
```
With:
```js
import { downloadManager } from './managers/DownloadManager.js';
import { showModelModalForCivitaiId } from './components/shared/ModelModal.js';
```

- [ ] **Step 4: Add `data-civitai-id` and `data-model-type` attributes in `buildCard`**

In `buildCard`, find the line:
```js
    card.dataset.modelId = model.id;
```
Add directly after it:
```js
    card.dataset.civitaiId = model.id;
    card.dataset.modelType = model.type;
```

- [ ] **Step 5: Add delegated click listener in `init()`**

In `init()`, add after `document.getElementById('loadMoreBtn').addEventListener(...)`:
```js
    document.getElementById('discoverGrid').addEventListener('click', (e) => {
        if (e.target.closest('.discover-install-btn')) return;
        const card = e.target.closest('.discover-card');
        if (!card || !card.dataset.civitaiId) return;
        const modelType = MODEL_TYPE_MAP[card.dataset.modelType] || 'loras';
        showModelModalForCivitaiId(card.dataset.civitaiId, modelType, {
            onInstall: () => installModel({
                id: Number(card.dataset.civitaiId),
                type: card.dataset.modelType,
            }),
        });
    });
```

- [ ] **Step 6: Run frontend tests**

```
cd /home/blackthorn/ComfyUI/custom_nodes/comfyui-lora-manager && npm run test:js -- --reporter verbose
```
Expected: all tests pass including the two new card click tests.

- [ ] **Step 7: Run full test suite**

```
cd /home/blackthorn/ComfyUI/custom_nodes/comfyui-lora-manager && npm test && pytest 2>&1 | tail -20
```
Expected: all frontend tests pass; backend: only the pre-existing bulk-refresh failure.

- [ ] **Step 8: Commit**

```bash
git -C /home/blackthorn/ComfyUI add -f \
    custom_nodes/comfyui-lora-manager/static/js/discover.js \
    custom_nodes/comfyui-lora-manager/tests/frontend/pages/discover.test.js
git -C /home/blackthorn/ComfyUI commit -m "feat: wire Discover card clicks to model detail modal"
```
