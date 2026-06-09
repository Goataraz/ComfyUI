import { downloadManager } from './managers/DownloadManager.js';
import { showModelModalForCivitaiId } from './components/shared/ModelModal.js';

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

// Pure exported functions (testable)

export function buildCard(model, installedIds) {
    const card = document.createElement('div');
    card.className = 'discover-card';
    card.dataset.modelId = model.id;
    card.dataset.civitaiId = model.id;
    card.dataset.modelType = model.type;

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

// Module state

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
    fetchInstalledIds().then(() => loadPage(1, false));
}

// Guard: only auto-run when the Discover page DOM is present
if (document.getElementById('discoverGrid')) {
    init();
}
