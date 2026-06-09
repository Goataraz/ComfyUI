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
