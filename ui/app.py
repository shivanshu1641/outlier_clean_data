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


def create_app(
    benchmark_dir: str = "outputs/benchmark",
    episodes_dir: str = "outputs/benchmark/episodes",
) -> gr.Blocks:
    """Create the full Gradio app with all tabs."""
    with gr.Blocks(title="Data Cleaning Benchmark") as app:
        gr.HTML(f"<style>{CUSTOM_CSS}</style>")
        gr.Markdown(
            "# Data Cleaning Environment\n"
            "Benchmark dashboard for evaluating AI agents on tabular data cleaning tasks "
            "across 6 skill categories."
        )

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
