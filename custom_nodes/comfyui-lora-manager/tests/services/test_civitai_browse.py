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
    assert params["sort"] == "Most Downloaded"
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
    assert result["error"] == "server error"
    assert result["items"] == []
    assert result["metadata"] == {}


@pytest.mark.asyncio
async def test_browse_models_reraises_rate_limit_error(downloader):
    from py.services.errors import RateLimitError
    downloader.make_request = AsyncMock(side_effect=RateLimitError("rate limited"))
    client = await CivitaiClient.get_instance()
    with pytest.raises(RateLimitError):
        await client.browse_models()


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
