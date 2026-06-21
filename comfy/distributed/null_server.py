"""
    This file is part of ComfyUI.
    Copyright (C) 2024 Comfy

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""

from __future__ import annotations


class NullProxy:
    """Safe stand-in for chained attribute access on worker ranks.

    Handles patterns like server.app.router.frozen or
    await PromptServer.instance.send(...) without crashing.
    Any attribute access returns another NullProxy; any call is a no-op.
    """
    def __getattr__(self, name):
        return NullProxy()

    def __call__(self, *args, **kwargs):
        return NullProxy()

    def __await__(self):
        async def _noop(*a, **kw):
            return None
        return _noop().__await__()

    def __bool__(self):
        return False

    def __int__(self):
        return 0

    def __iadd__(self, other):
        return self

    def __setattr__(self, name, value):
        pass

    def __iter__(self):
        return iter([])

    def __contains__(self, item):
        return False

    def __len__(self):
        return 0


class NullQueue:
    """No-op queue stand-in for TP worker ranks."""
    def __init__(self):
        self.currently_running = False

    def get_flags(self):
        return {}

    def set_flag(self, *args, **kwargs):
        pass

    def task_done(self, *args, **kwargs):
        pass

    def put(self, *args, **kwargs):
        pass

    def get(self, *args, **kwargs):
        return None

    def __getattr__(self, name):
        return NullProxy()


class NullRoutes:
    """No-op routes stand-in that swallows decorator-based route registrations."""
    def get(self, path, **kwargs):
        def decorator(func):
            return func
        return decorator

    def post(self, path, **kwargs):
        def decorator(func):
            return func
        return decorator

    def put(self, path, **kwargs):
        def decorator(func):
            return func
        return decorator

    def delete(self, path, **kwargs):
        def decorator(func):
            return func
        return decorator


class NullServer:
    """No-op server stand-in for TP worker ranks that don't run the HTTP server.

    Explicit attributes and methods mirror the PromptServer interface.
    Any unknown attribute returns a NullProxy that safely handles chained
    access and calls, so custom nodes that reference PromptServer.instance
    won't crash.
    """
    def __init__(self):
        self.client_id = None
        self.last_prompt_id = None
        self.last_node_id = None
        self.number = 0
        self.on_prompt_handlers = []
        self.prompt_queue = NullQueue()
        self.routes = NullRoutes()
        self.app = NullProxy()
        self.loop = NullProxy()
        self.client_session = None
        self.messages = NullProxy()
        self.node_replace_manager = NullProxy()
        self.supports = set()
        self.instance = self

    def send_sync(self, *args, **kwargs):
        pass

    def queue_updated(self):
        pass

    def send_progress_text(self, text, node_id, sid=None):
        pass

    def add_on_prompt_handler(self, handler):
        pass

    async def send(self, event, data, sid=None):
        pass

    def get_queue_info(self):
        return {}

    def trigger_on_prompt(self, json_data):
        return json_data

    def __getattr__(self, name):
        return NullProxy()