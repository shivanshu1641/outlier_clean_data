"""FastAPI application for the Data Cleaning OpenEnv environment."""

from __future__ import annotations

import sys
import os

# Ensure root is on path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import ActionWrapper, DataCleaningObservation
from server.environment import DataCleaningEnvironment
from openenv.core import create_app

import gradio as gr

# Patch TabbedInterface so "Custom" tab (benchmark dashboard) loads first
_OrigTabbedInterface = gr.TabbedInterface

def _swapped_tabbed_interface(interfaces, tab_names=None, **kwargs):
    if tab_names and len(tab_names) == 2 and tab_names[0] == "Playground":
        interfaces = list(reversed(interfaces))
        tab_names = list(reversed(tab_names))
    return _OrigTabbedInterface(interfaces, tab_names=tab_names, **kwargs)

gr.TabbedInterface = _swapped_tabbed_interface


def _gradio_builder(*args, **kwargs):
    """Custom Gradio UI builder for the openenv /web endpoint."""
    from ui.app import create_app as create_ui_app
    return create_ui_app()


app = create_app(
    env=DataCleaningEnvironment,
    action_cls=ActionWrapper,
    observation_cls=DataCleaningObservation,
    env_name="data_cleaning_env",
    gradio_builder=_gradio_builder,
)


@app.get("/")
def health():
    return {"status": "ok", "environment": "data_cleaning_env"}


def main():
    import uvicorn
    port = int(os.environ.get("PORT", "7860"))
    uvicorn.run(app, host="0.0.0.0", port=port, ws_ping_interval=60, ws_ping_timeout=120)


if __name__ == "__main__":
    main()
