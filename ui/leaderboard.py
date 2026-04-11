# ui/leaderboard.py
"""Benchmark leaderboard tab for the Gradio UI."""
from __future__ import annotations

import gradio as gr
import pandas as pd
import plotly.express as px

from ui.data_loader import load_benchmark_summary


def _build_pivot(df: pd.DataFrame, difficulty: str | None = None) -> pd.DataFrame:
    """Build Model × Category pivot table of average rewards."""
    if df.empty:
        return pd.DataFrame()
    filtered = df.copy()
    if difficulty and difficulty != "All":
        filtered = filtered[filtered["difficulty"] == difficulty]
    if filtered.empty:
        return pd.DataFrame()
    pivot = filtered.pivot_table(
        index="model", columns="category", values="reward", aggfunc="mean"
    ).round(3)
    pivot["Overall"] = pivot.mean(axis=1).round(3)
    pivot = pivot.sort_values("Overall", ascending=False)
    return pivot.reset_index()


def _build_bar_chart(df: pd.DataFrame, difficulty: str | None = None):
    """Build a grouped bar chart of model performance by category."""
    if df.empty:
        return None
    filtered = df.copy()
    if difficulty and difficulty != "All":
        filtered = filtered[filtered["difficulty"] == difficulty]
    if filtered.empty:
        return None
    agg = filtered.groupby(["model", "category"])["reward"].mean().reset_index()
    fig = px.bar(
        agg, x="category", y="reward", color="model",
        barmode="group", title="Model Performance by Category",
        labels={"reward": "Average Reward", "category": "Category"},
    )
    fig.update_layout(yaxis_range=[0, 1])
    return fig


def create_leaderboard_tab(benchmark_dir: str = "outputs/benchmark") -> gr.Blocks:
    """Create the leaderboard tab component."""
    df = load_benchmark_summary(benchmark_dir)
    difficulties = ["All"] + sorted(df["difficulty"].unique().tolist()) if not df.empty else ["All"]

    with gr.Blocks() as tab:
        gr.Markdown("## Benchmark Leaderboard")
        gr.Markdown("*These are pre-computed benchmark results.*")

        if df.empty:
            gr.Markdown("**No benchmark results found.** Run the benchmark first:\n"
                        "```\npython -m tools.benchmark_runner\n```")
            return tab

        difficulty_dropdown = gr.Dropdown(
            choices=difficulties, value="All", label="Filter by Difficulty"
        )
        pivot_table = gr.DataFrame(
            value=_build_pivot(df), label="Model × Category Scores (average reward)",
        )
        bar_chart = gr.Plot(value=_build_bar_chart(df), label="Performance Comparison")

        total_tasks = len(df)
        total_models = df["model"].nunique()
        total_datasets = df["dataset_id"].nunique()
        gr.Markdown(
            f"**Stats:** {total_tasks} benchmark runs | "
            f"{total_models} models | {total_datasets} datasets"
        )

        def update_views(difficulty):
            return _build_pivot(df, difficulty), _build_bar_chart(df, difficulty)

        difficulty_dropdown.change(
            fn=update_views, inputs=[difficulty_dropdown],
            outputs=[pivot_table, bar_chart],
        )

    return tab
