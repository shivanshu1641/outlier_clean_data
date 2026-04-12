"""Model Comparison tab — radar chart, difficulty breakdown, reward distribution."""
from __future__ import annotations

import gradio as gr
import pandas as pd
import plotly.graph_objects as go

from ui.data_loader import load_results, load_best_per_model_task
from ui.theme import CATEGORY_META, MODEL_COLORS


def _build_radar_chart(df: pd.DataFrame) -> go.Figure:
    """Each model as a trace across 6 benchmark categories."""
    fig = go.Figure()
    if df.empty or "category" not in df.columns:
        return fig

    best = load_best_per_model_task(df)
    categories = list(CATEGORY_META.keys())
    pivot = best.pivot_table(index="model", columns="category", values="reward", aggfunc="mean")

    for i, model in enumerate(pivot.index):
        values = [float(pivot.loc[model, cat]) if cat in pivot.columns and pd.notna(pivot.loc[model, cat]) else 0.0
                  for cat in categories]
        # Close the polygon
        fig.add_trace(go.Scatterpolar(
            r=values + [values[0]],
            theta=categories + [categories[0]],
            fill="toself",
            name=model,
            line_color=MODEL_COLORS[i % len(MODEL_COLORS)],
            opacity=0.65,
        ))

    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1], tickformat=".0%")),
        showlegend=True,
        title=dict(text="Category Performance Radar", font=dict(size=14, color="#1e293b")),
        plot_bgcolor="#fff",
        paper_bgcolor="#fff",
        height=420,
        margin=dict(l=40, r=40, t=60, b=40),
        font=dict(color="#475569", size=12),
        legend=dict(orientation="h", yanchor="top", y=-0.05),
    )
    return fig


def _build_difficulty_chart(df: pd.DataFrame) -> go.Figure:
    """Grouped bar: easy / medium / hard avg reward per model."""
    fig = go.Figure()
    if df.empty:
        return fig

    best = load_best_per_model_task(df)
    models = sorted(best["model"].unique().tolist())
    diff_colors = {"easy": "#059669", "medium": "#d97706", "hard": "#dc2626"}

    for diff in ["easy", "medium", "hard"]:
        if "difficulty" not in best.columns:
            continue
        sub = best[best["difficulty"] == diff]
        if sub.empty:
            continue
        avg = sub.groupby("model")["reward"].mean()
        vals = [float(avg.get(m, 0)) for m in models]
        fig.add_trace(go.Bar(
            name=diff.capitalize(),
            x=models,
            y=vals,
            marker_color=diff_colors[diff],
            text=[f"{v:.0%}" for v in vals],
            textposition="outside",
            cliponaxis=False,
        ))

    fig.update_layout(
        barmode="group",
        title=dict(text="Reward by Difficulty per Model", font=dict(size=14, color="#1e293b")),
        yaxis=dict(range=[0, 1.2], tickformat=".0%", showgrid=True, gridcolor="#e2e8f0"),
        xaxis=dict(showgrid=False),
        plot_bgcolor="#fff",
        paper_bgcolor="#fff",
        height=350,
        margin=dict(l=10, r=10, t=44, b=10),
        font=dict(color="#475569", size=12),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def _build_box_plot(df: pd.DataFrame) -> go.Figure:
    """Reward distribution box plot per model."""
    fig = go.Figure()
    if df.empty:
        return fig

    best = load_best_per_model_task(df)
    models = sorted(best["model"].unique().tolist())

    for i, model in enumerate(models):
        rewards = best[best["model"] == model]["reward"].tolist()
        fig.add_trace(go.Box(
            y=rewards,
            name=model,
            marker_color=MODEL_COLORS[i % len(MODEL_COLORS)],
            boxmean=True,
            line_width=1.5,
        ))

    fig.update_layout(
        title=dict(text="Reward Distribution per Model", font=dict(size=14, color="#1e293b")),
        yaxis=dict(range=[-0.05, 1.05], tickformat=".0%", showgrid=True, gridcolor="#e2e8f0"),
        xaxis=dict(showgrid=False),
        plot_bgcolor="#fff",
        paper_bgcolor="#fff",
        height=350,
        margin=dict(l=10, r=10, t=44, b=10),
        font=dict(color="#475569", size=12),
        showlegend=False,
    )
    return fig


def create_comparison_tab(benchmark_dir: str = "outputs/benchmark") -> gr.Blocks:
    df = load_results(benchmark_dir)

    with gr.Blocks() as tab:
        gr.Markdown("## Model Comparison")

        if df.empty:
            gr.Markdown(
                "**No results found.** Run the benchmark first:\n"
                "```\npython -m tools.benchmark_runner\n```"
            )
            return tab

        n_models = df["model"].nunique()
        n_tasks = df["task_id"].nunique() if "task_id" in df.columns else len(df)
        gr.Markdown(f"*{n_models} models · {n_tasks} tasks*")

        with gr.Row():
            with gr.Column(scale=1):
                gr.Plot(value=_build_radar_chart(df), label="Category Radar")
            with gr.Column(scale=1):
                gr.Plot(value=_build_difficulty_chart(df), label="Difficulty Breakdown")

        with gr.Row():
            with gr.Column():
                gr.Plot(value=_build_box_plot(df), label="Reward Distribution")

    return tab
