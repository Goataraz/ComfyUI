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
        sort="Most Downloaded",
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


@pytest.mark.asyncio
async def test_browse_returns_429_on_rate_limit(monkeypatch):
    from py.services.errors import RateLimitError
    mock_client = AsyncMock()
    mock_client.browse_models = AsyncMock(side_effect=RateLimitError("rate limited"))
    monkeypatch.setattr(
        discover_module.CivitaiClient, "get_instance", AsyncMock(return_value=mock_client)
    )
    routes = DiscoverRoutes()
    request = make_mocked_request("GET", "/api/lm/discover/browse")
    response = await routes.browse_models(request)
    assert response.status == 429


@pytest.mark.asyncio
async def test_installed_ids_handles_scanner_failure_gracefully(monkeypatch):
    """If one scanner raises, the others still contribute."""
    good_data = [{"civitai": {"modelId": 999}}]

    async def fake_lora_fail(cls):
        raise RuntimeError("scanner unavailable")

    async def fake_checkpoint(cls):
        return FakeScanner(good_data)

    async def fake_embedding(cls):
        return FakeScanner([])

    monkeypatch.setattr(
        discover_module.ServiceRegistry, "get_lora_scanner", classmethod(fake_lora_fail)
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
    routes = DiscoverRoutes()
    request = make_mocked_request("GET", "/api/lm/discover/installed-ids")
    response = await routes.get_installed_ids(request)
    import json
    data = json.loads(response.body)
    assert data["ids"] == [999]
    assert response.status == 200


@pytest.mark.asyncio
async def test_browse_returns_401_on_auth_failure(monkeypatch):
    mock_client = AsyncMock()
    mock_client.browse_models = AsyncMock(
        return_value={"error": "401 Unauthorized: Invalid API key", "items": [], "metadata": {}}
    )
    monkeypatch.setattr(
        discover_module.CivitaiClient, "get_instance", AsyncMock(return_value=mock_client)
    )
    routes = DiscoverRoutes()
    request = make_mocked_request("GET", "/api/lm/discover/browse")
    response = await routes.browse_models(request)
    assert response.status == 401
