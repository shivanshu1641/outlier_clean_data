"""Episode explorer tab — step-by-step replay of agent cleaning episodes."""
from __future__ import annotations

import gradio as gr
import pandas as pd
import plotly.graph_objects as go

from ui.data_loader import list_episode_files, load_episode_log


def _build_reward_chart(events: list[dict]) -> go.Figure | None:
    steps = [e for e in events if e.get("event") == "step"]
    if not steps:
        return None

    step_nums = [s["step"] for s in steps]
    rewards = [s.get("reward", 0.0) for s in steps]
    action_types = [s.get("action_type", "") for s in steps]

    color_map = {
        "transform": "#0d9488",
        "explore": "#2563eb",
        "validate": "#d97706",
        "done": "#059669",
        "undo": "#dc2626",
    }
    colors = [color_map.get(a, "#94a3b8") for a in action_types]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=step_nums, y=rewards, mode="lines+markers",
        line=dict(color="#0d9488", width=2),
        marker=dict(color=colors, size=10, line=dict(color="#fff", width=1)),
        hovertext=[f"Step {n}: {a}<br>Reward: {r:.4f}" for n, a, r in zip(step_nums, action_types, rewards)],
        hoverinfo="text",
    ))
    fig.update_layout(
        plot_bgcolor="#fff",
        paper_bgcolor="#fff",
        font=dict(color="#475569"),
        margin=dict(l=10, r=10, t=40, b=10),
        height=250,
        xaxis=dict(title="Step", showgrid=False, zeroline=False),
        yaxis=dict(title="Reward", range=[0, 1.05], showgrid=True, gridcolor="#e2e8f0"),
        title=dict(text="Score Progression", font=dict(size=14, color="#1e293b")),
    )
    return fig


def _format_steps_html(events: list[dict]) -> str:
    steps = [e for e in events if e.get("event") == "step"]
    if not steps:
        return "<p style='color:#94a3b8'>No step data available.</p>"

    color_map = {
        "transform": "#0d9488",
        "explore": "#2563eb",
        "validate": "#d97706",
        "done": "#059669",
        "undo": "#dc2626",
    }

    html_parts = []
    for s in steps:
        action = s.get("action_type", "unknown")
        color = color_map.get(action, "#94a3b8")
        step_num = s.get("step", "?")
        reward = s.get("reward", 0.0)
        fixed = s.get("errors_fixed", 0)
        total = s.get("errors_total", 0)
        content = s.get("action_content", "")

        code_block = ""
        if content and action == "transform":
            escaped = content.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
            code_block = (
                f'<pre style="background:#f1f5f9;border:1px solid #e2e8f0;border-radius:6px;'
                f'padding:10px;margin:8px 0;font-size:12px;color:#334155;overflow-x:auto">{escaped}</pre>'
            )

        html_parts.append(
            f'<div style="background:#f8fafc;border:1px solid #e2e8f0;border-radius:10px;padding:18px;margin-bottom:10px">'
            f'<div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:6px">'
            f'<span style="font-weight:600;color:#1e293b">Step {step_num}: '
            f'<span style="color:{color}">{action}</span></span>'
            f'<span style="font-size:13px;color:#64748b">'
            f'Reward: <b style="color:#1e293b">{reward:.4f}</b>'
            f' &nbsp;|&nbsp; Fixed: <b style="color:#059669">{fixed}</b>/{total}'
            f'</span></div>'
            f'{code_block}</div>'
        )

    return "\n".join(html_parts)


def create_explorer_tab(log_dir: str = "outputs/episodes") -> gr.Blocks:
    episodes = list_episode_files(log_dir)

    with gr.Blocks() as tab:
        gr.Markdown("## Episode Explorer")
        gr.Markdown("Step-by-step replay of agent cleaning episodes.")

        if not episodes:
            gr.Markdown(
                f"**No episodes found in `{log_dir}/`.** "
                "Copy JSONL logs there to browse them:\n"
                "```\nmkdir -p outputs/episodes\n"
                "cp outputs/logs/run_*.jsonl outputs/episodes/\n```"
            )
            return tab

        choices = []
        for ep in episodes:
            label = f"{ep['model']} | {ep['task_id']} | reward={ep['final_reward']:.3f}"
            choices.append((label, ep["path"]))

        # Pick best episode (highest reward) as default
        best_ep = max(episodes, key=lambda e: e.get("final_reward", 0))
        default_path = best_ep["path"]

        episode_dd = gr.Dropdown(
            choices=choices,
            label="▼ Select Episode",
            value=default_path,
            info=f"Click to browse {len(episodes)} episodes — switch between models and tasks",
            elem_classes=["interactive-dropdown"],
        )

        def _load_episode(path):
            if not path:
                return "Select an episode.", gr.update(visible=False), ""

            events = load_episode_log(path)
            if not events:
                return "Failed to load episode.", gr.update(visible=False), ""

            start = next((e for e in events if e.get("event") == "task_start"), {})
            end = next((e for e in events if e.get("event") == "task_end"), {})

            # Fall back to filename metadata if event fields missing
            model = start.get("model") or "unknown"
            task_id = start.get("task_id") or "unknown"
            final_reward = end.get("final_reward", end.get("reward", 0.0))
            total_steps = end.get("total_steps", len([e for e in events if e.get("event") == "step"]))
            elapsed = end.get("elapsed_s", 0.0)

            summary = (
                f"### {task_id}\n"
                f"**Model:** {model} &nbsp;|&nbsp; "
                f"**Final reward:** {final_reward:.4f} &nbsp;|&nbsp; "
                f"**Steps:** {total_steps} &nbsp;|&nbsp; "
                f"**Time:** {elapsed:.1f}s"
            )

            chart = _build_reward_chart(events)
            steps = _format_steps_html(events)

            return summary, gr.update(value=chart, visible=chart is not None), steps

        # Auto-load default episode
        _events = load_episode_log(default_path)
        _start = next((e for e in _events if e.get("event") == "task_start"), {})
        _end = next((e for e in _events if e.get("event") == "task_end"), {})
        _model = _start.get("model") or "unknown"
        _task_id = _start.get("task_id") or "unknown"
        _final = _end.get("final_reward", _end.get("reward", 0.0))
        _steps_n = _end.get("total_steps", len([e for e in _events if e.get("event") == "step"]))
        _elapsed = _end.get("elapsed_s", 0.0)
        _default_summary = (
            f"### {_task_id}\n**Model:** {_model} &nbsp;|&nbsp; "
            f"**Final reward:** {_final:.4f} &nbsp;|&nbsp; "
            f"**Steps:** {_steps_n} &nbsp;|&nbsp; **Time:** {_elapsed:.1f}s"
        )
        _default_chart = _build_reward_chart(_events)
        _default_steps_html = _format_steps_html(_events)

        summary_md = gr.Markdown(value=_default_summary)
        reward_chart = gr.Plot(value=_default_chart, visible=_default_chart is not None)
        steps_html = gr.HTML(value=_default_steps_html)

        episode_dd.change(
            fn=_load_episode,
            inputs=[episode_dd],
            outputs=[summary_md, reward_chart, steps_html],
        )

    return tab
