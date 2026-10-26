"""Pytest configuration and test environment setup.

Ensures safe execution in restricted sandboxes / headless CI without network access.
"""

import asyncio
import os

# Disable telemetry and online checks in test environments
os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"

# In isolated sandbox environments without loopback socket binding permission,
# patch Gradio lock and event helpers to avoid socket.socketpair() loopback calls.
try:
    import gradio.utils as _gu
    import gradio.queueing as _gq
    import gradio.routes as _grr

    _gu.safe_get_lock = _gq.safe_get_lock = lambda: asyncio.Lock()
    _gu.safe_get_stop_event = _grr.safe_get_stop_event = lambda: asyncio.Event()
except ImportError:
    pass
