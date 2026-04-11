# ui/explorer.py
"""Episode explorer tab — step-by-step replay of agent cleaning episodes."""
from __future__ import annotations

import gradio as gr
import pandas as pd
import plotly.express as px

from ui.data_loader import load_benchmark_summary, load_episode_log


def _format_step(step: dict) -> str:
    step_num = step.get("step", "?")
    action_type = step.get("type", "unknown")
    summary = step.get("summary", "")
    reward = step.get("reward_after", 0.0)
    fixed = step.get("errors_fixed", 0)
    header = f"### Step {step_num}: `{action_type}`"
    code_block = f"\n```python\n{summary}\n```" if summary else ""
    return f"{header}{code_block}\n\nReward: **{reward:.4f}** | Errors fixed: **{fixed}**\n\n---"


def _build_reward_chart(steps: list[dict]):
    if not steps:
        return None
    df = pd.DataFrame(steps)
    if "reward_after" not in df.columns:
        return None
    fig = px.line(
        df, x="step", y="reward_after", title="Score Progression",
        labels={"reward_after": "Reward", "step": "Step"}, markers=True,
    )
    fig.update_layout(yaxis_range=[0, 1])
    return fig


def create_explorer_tab(benchmark_dir: str = "outputs/benchmark") -> gr.Blocks:
    df = load_benchmark_summary(benchmark_dir)

    with gr.Blocks() as tab:
        gr.Markdown("## Episode Explorer")
        gr.Markdown("*Step-by-step replay of pre-computed agent episodes.*")

        if df.empty:
            gr.Markdown("**No episodes found.** Run the benchmark first.")
            return tab

        with gr.Row():
            model_dd = gr.Dropdown(choices=sorted(df["model"].unique().tolist()), label="Model")
            dataset_dd = gr.Dropdown(choices=sorted(df["dataset_id"].unique().tolist()), label="Dataset")
            category_dd = gr.Dropdown(choices=sorted(df["category"].unique().tolist()), label="Category")
            difficulty_dd = gr.Dropdown(choices=sorted(df["difficulty"].unique().tolist()), label="Difficulty")

        load_btn = gr.Button("Load Episode")
        episode_summary = gr.Markdown("Select filters and click Load Episode.")
        reward_chart = gr.Plot(label="Score Progression")
        steps_display = gr.Markdown("")

        def load_episode(model, dataset, category, difficulty):
            if not all([model, dataset, category, difficulty]):
                return "Please select all filters.", None, ""
            matches = df[
                (df["model"] == model) & (df["dataset_id"] == dataset) &
                (df["category"] == category) & (df["difficulty"] == difficulty)
            ]
            if matches.empty:
                return "No matching episode found.", None, ""
            row = matches.iloc[0]
            ep_path = row.get("episode_log_path", "")
            steps = load_episode_log(ep_path) if ep_path else []
            summary = (
                f"**{dataset}** | {category} | {difficulty} | {model}\n\n"
                f"Final reward: **{row['reward']:.4f}** | Steps: **{row['steps']}** | "
                f"Time: **{row['elapsed_s']:.1f}s**"
            )
            chart = _build_reward_chart(steps)
            steps_md = "\n\n".join(_format_step(s) for s in steps) if steps else "No step data available."
            return summary, chart, steps_md

        load_btn.click(
            fn=load_episode,
            inputs=[model_dd, dataset_dd, category_dd, difficulty_dd],
            outputs=[episode_summary, reward_chart, steps_display],
        )

    return tab
