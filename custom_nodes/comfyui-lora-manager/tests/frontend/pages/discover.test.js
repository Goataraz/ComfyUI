import { beforeEach, describe, expect, it, vi } from 'vitest';

vi.mock('../../../static/js/managers/DownloadManager.js', () => ({
    downloadManager: { openForModelVersion: vi.fn() },
}));

vi.mock('../../../static/js/components/shared/ModelModal.js', () => ({
    showModelModalForCivitaiId: vi.fn(),
}));

vi.mock('../../../static/js/core.js', () => ({
    appCore: { initialize: vi.fn().mockResolvedValue(undefined) },
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

describe('card click behavior', () => {
    beforeEach(async () => {
        vi.resetModules();
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
        document.dispatchEvent(new Event('DOMContentLoaded'));
        await new Promise(resolve => setTimeout(resolve, 0));

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
        document.dispatchEvent(new Event('DOMContentLoaded'));
        await new Promise(resolve => setTimeout(resolve, 0));

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
