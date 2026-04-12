"""Gradio app entry point — benchmark dashboard.

Usage:
    python -m ui.app                           # default port 7861
    python -m ui.app --port 7862               # custom port
    python -m ui.app --benchmark-dir results/  # custom results directory
"""
from __future__ import annotations

import argparse
import os

import gradio as gr

from ui.leaderboard import create_leaderboard_tab
from ui.explorer import create_explorer_tab
from ui.catalog_view import create_catalog_tab
from ui.theme import CUSTOM_CSS

_HEADER_HTML = """
<div style="padding:32px 36px;background:linear-gradient(135deg,#0f172a 0%,#0d2137 55%,#0f2d2a 100%);
            border-radius:12px;margin-bottom:4px;position:relative;overflow:hidden;">
  <div style="position:absolute;inset:0;opacity:0.04;pointer-events:none;
              background-image:linear-gradient(#fff 1px,transparent 1px),linear-gradient(90deg,#fff 1px,transparent 1px);
              background-size:28px 28px;"></div>
  <div style="position:relative;">
    <div style="font-size:11px;font-weight:700;letter-spacing:.18em;color:#0d9488;
                text-transform:uppercase;margin-bottom:12px;">
      Open Benchmark &nbsp;·&nbsp; 6 categories &nbsp;·&nbsp; 3 difficulty levels
    </div>
    <h1 style="margin:0 0 12px 0;font-size:32px;font-weight:800;line-height:1.1;letter-spacing:-0.5px;
               background:linear-gradient(90deg,#5eead4 0%,#818cf8 100%);
               -webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;">
      Data Cleaning Benchmark
    </h1>
    <p style="margin:0;font-size:14px;color:#64748b;line-height:1.75;max-width:580px;">
      A standardized benchmark for measuring how well AI agents clean corrupted tabular data.<br>
      Six damage categories, three difficulty levels, fully reproducible episodes.
    </p>
  </div>
</div>
"""


def create_app(
    benchmark_dir: str = "outputs/benchmark",
    episodes_dir: str = "outputs/benchmark/episodes",
) -> gr.Blocks:
    """Create the full Gradio app with all tabs."""
    with gr.Blocks(title="Data Cleaning Benchmark") as app:
        gr.HTML(_HEADER_HTML)

        with gr.Tabs():
            with gr.Tab("Leaderboard"):
                create_leaderboard_tab(benchmark_dir)
            with gr.Tab("Episode Explorer"):
                create_explorer_tab(episodes_dir)
            with gr.Tab("Dataset Catalog"):
                create_catalog_tab()

    return app


def main():
    parser = argparse.ArgumentParser(description="Launch benchmark dashboard")
    parser.add_argument("--port", type=int, default=int(os.environ.get("UI_PORT", "7861")))
    parser.add_argument("--benchmark-dir", default="outputs/benchmark")
    parser.add_argument("--episodes-dir", default="outputs/benchmark/episodes")
    parser.add_argument("--share", action="store_true", help="Create public Gradio link")
    args = parser.parse_args()

    app = create_app(benchmark_dir=args.benchmark_dir, episodes_dir=args.episodes_dir)
    app.launch(
        server_port=args.port,
        share=args.share,
        css=CUSTOM_CSS,
    )


if __name__ == "__main__":
    main()
