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
