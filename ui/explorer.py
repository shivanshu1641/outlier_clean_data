"""Episode explorer tab — step-by-step replay of agent cleaning episodes."""
from __future__ import annotations

import gradio as gr
import pandas as pd
import plotly.graph_objects as go

from ui.data_loader import list_episode_files, load_episode_log


def _parse_episode_meta(ep: dict) -> dict:
    """Extract model, dataset, difficulty from episode metadata."""
    # Filename: {dataset}_{category}_{difficulty}_{model}_{seed}.jsonl
    fname = ep.get("file", "")
    parts = fname.replace(".jsonl", "").rsplit("_", 2)  # [..., model_chunk, seed]

    # Parse difficulty from task_id: "rock_mine_medium_csv"
    task_id = ep.get("task_id", "")
    tid_parts = task_id.rsplit("_", 2)
    difficulty = tid_parts[1] if len(tid_parts) >= 3 else "unknown"
    dataset = tid_parts[0] if len(tid_parts) >= 3 else task_id

    return {
        **ep,
        "dataset": dataset,
        "difficulty": difficulty,
    }


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

    # Add task_end summary card
    end = next((e for e in events if e.get("event") == "task_end"), None)
    if end:
        final_reward = end.get("final_reward", 0.0)
        total_steps = end.get("total_steps", 0)
        elapsed = end.get("elapsed_s", 0.0)
        reward_color = "#059669" if final_reward >= 0.5 else "#d97706" if final_reward >= 0.2 else "#dc2626"
        html_parts.append(
            f'<div style="background:#f0fdf4;border:2px solid {reward_color};border-radius:10px;padding:18px;margin-bottom:10px">'
            f'<div style="display:flex;justify-content:space-between;align-items:center">'
            f'<span style="font-weight:700;color:#1e293b;font-size:15px">Episode Complete</span>'
            f'<span style="font-size:13px;color:#64748b">'
            f'Final Reward: <b style="color:{reward_color};font-size:15px">{final_reward:.4f}</b>'
            f' &nbsp;|&nbsp; Steps: <b>{total_steps}</b>'
            f' &nbsp;|&nbsp; Time: <b>{elapsed:.1f}s</b>'
            f'</span></div></div>'
        )

    return "\n".join(html_parts)


def _build_episode_table_html(episodes: list[dict], selected_path: str = "") -> str:
    """Build HTML table of episodes with clickable rows."""
    if not episodes:
        return '<p style="color:#94a3b8">No episodes match filters.</p>'

    rows = []
    for i, ep in enumerate(episodes):
        reward = ep.get("final_reward", 0.0)
        steps = ep.get("steps", 0)
        reward_color = "#059669" if reward >= 0.5 else "#d97706" if reward >= 0.2 else "#dc2626"
        is_selected = ep.get("path", "") == selected_path
        bg = "#dbeafe" if is_selected else ("#f8fafc" if i % 2 == 0 else "#fff")
        left_border = "3px solid #2563eb" if is_selected else "3px solid transparent"

        diff_colors = {"easy": "#059669", "medium": "#d97706", "hard": "#dc2626"}
        diff_color = diff_colors.get(ep.get("difficulty", ""), "#64748b")

        rows.append(
            f'<tr style="background:{bg};border-bottom:1px solid #e2e8f0;border-left:{left_border}">'
            f'<td style="padding:10px;font-size:12px;color:#1e293b;font-weight:500">{ep.get("model", "?")}</td>'
            f'<td style="padding:10px;font-size:12px;color:#334155">{ep.get("dataset", "?")}</td>'
            f'<td style="padding:10px;font-size:12px"><span style="color:{diff_color};font-weight:600">{ep.get("difficulty", "?")}</span></td>'
            f'<td style="padding:10px;text-align:center">'
            f'<b style="color:{reward_color}">{reward:.3f}</b></td>'
            f'<td style="padding:10px;text-align:center;font-size:12px;color:#334155;font-weight:500">{steps}</td>'
            f'</tr>'
        )

    header = (
        '<div style="max-height:600px;overflow-y:auto;border:1px solid #e2e8f0;border-radius:8px">'
        '<table style="width:100%;border-collapse:collapse;font-size:13px;background:#fff">'
        '<thead><tr style="background:#f1f5f9;position:sticky;top:0;border-bottom:2px solid #cbd5e1">'
        '<th style="padding:10px;text-align:left;color:#334155;font-weight:700;font-size:12px">Model</th>'
        '<th style="padding:10px;text-align:left;color:#334155;font-weight:700;font-size:12px">Dataset</th>'
        '<th style="padding:10px;text-align:left;color:#334155;font-weight:700;font-size:12px">Difficulty</th>'
        '<th style="padding:10px;text-align:center;color:#334155;font-weight:700;font-size:12px">Reward</th>'
        '<th style="padding:10px;text-align:center;color:#334155;font-weight:700;font-size:12px">Steps</th>'
        '</tr></thead><tbody>'
    )

    return header + "\n".join(rows) + '</tbody></table></div>'


def _render_episode(path: str):
    """Load and render a single episode."""
    if not path:
        empty_fig = go.Figure()
        empty_fig.update_layout(height=250)
        return "Select an episode from the table.", gr.update(value=empty_fig, visible=False), ""

    events = load_episode_log(path)
    if not events:
        empty_fig = go.Figure()
        empty_fig.update_layout(height=250)
        return "Failed to load episode.", gr.update(value=empty_fig, visible=False), ""

    start = next((e for e in events if e.get("event") == "task_start"), {})
    end = next((e for e in events if e.get("event") == "task_end"), {})

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
    steps_html = _format_steps_html(events)

    return summary, gr.update(value=chart, visible=chart is not None), steps_html


def create_explorer_tab(log_dir: str = "outputs/benchmark/episodes") -> gr.Blocks:
    raw_episodes = list_episode_files(log_dir)
    all_episodes = [_parse_episode_meta(ep) for ep in raw_episodes]

    with gr.Blocks() as tab:
        gr.Markdown("## Episode Explorer")
        gr.Markdown("Step-by-step replay of agent cleaning episodes.")

        if not all_episodes:
            gr.Markdown(
                f"**No episodes found in `{log_dir}/`.** "
                "Run the benchmark first to generate episodes."
            )
            return tab

        # Extract unique filter values
        all_models = sorted({ep["model"] for ep in all_episodes})
        all_datasets = sorted({ep["dataset"] for ep in all_episodes})
        all_diffs = ["easy", "medium", "hard"]

        # Default: longest non-zero reward run
        nonzero = [ep for ep in all_episodes if ep.get("final_reward", 0) > 0]
        if nonzero:
            default_ep = max(nonzero, key=lambda e: e.get("steps", 0))
        else:
            default_ep = max(all_episodes, key=lambda e: e.get("steps", 0))
        default_path = default_ep["path"]

        # Store episode data as state
        episodes_state = gr.State(all_episodes)
        selected_path = gr.State(default_path)

        # ── Filters ──────────────────────────────────────
        with gr.Row():
            model_dd = gr.Dropdown(
                choices=["All"] + all_models,
                value="All",
                label="Model",
                scale=2,
            )
            dataset_dd = gr.Dropdown(
                choices=["All"] + all_datasets,
                value="All",
                label="Dataset",
                scale=2,
            )
            diff_dd = gr.Dropdown(
                choices=["All"] + all_diffs,
                value="All",
                label="Difficulty",
                scale=1,
            )

        # ── 2-column layout ──────────────────────────────
        with gr.Row():
            # Left: episode table
            with gr.Column(scale=2):
                gr.Markdown(f"*{len(all_episodes)} episodes available*")
                episode_table = gr.HTML(
                    value=_build_episode_table_html(all_episodes, default_path)
                )
                # Dropdown to select episode (since HTML table clicks can't trigger callbacks)
                episode_select = gr.Dropdown(
                    choices=[(f"{ep['model']} | {ep['dataset']}_{ep['difficulty']} | {ep['final_reward']:.3f} | {ep['steps']} steps", ep["path"])
                             for ep in all_episodes],
                    value=default_path,
                    label="Select episode to view",
                    info="Pick from filtered list above",
                )

            # Right: episode detail
            with gr.Column(scale=3):
                summary_md = gr.Markdown()
                reward_chart = gr.Plot(visible=False)
                steps_html = gr.HTML()

        # ── Filter logic ─────────────────────────────────
        def _apply_filters(model, dataset, difficulty):
            filtered = all_episodes
            if model != "All":
                filtered = [e for e in filtered if e["model"] == model]
            if dataset != "All":
                filtered = [e for e in filtered if e["dataset"] == dataset]
            if difficulty != "All":
                filtered = [e for e in filtered if e["difficulty"] == difficulty]

            # Sort: non-zero reward first, then by steps desc
            filtered.sort(key=lambda e: (-int(e.get("final_reward", 0) > 0), -e.get("steps", 0)))

            # Pick best default from filtered
            new_default = ""
            if filtered:
                nz = [e for e in filtered if e.get("final_reward", 0) > 0]
                best = max(nz, key=lambda e: e.get("steps", 0)) if nz else filtered[0]
                new_default = best["path"]

            table_html = _build_episode_table_html(filtered, new_default)
            choices = [(f"{ep['model']} | {ep['dataset']}_{ep['difficulty']} | {ep['final_reward']:.3f} | {ep['steps']} steps", ep["path"])
                       for ep in filtered]

            if new_default:
                summ, chart_update, steps = _render_episode(new_default)
                return (
                    table_html,
                    gr.update(choices=choices, value=new_default),
                    summ, chart_update, steps,
                )
            else:
                empty_fig = go.Figure()
                empty_fig.update_layout(height=250)
                return (
                    table_html,
                    gr.update(choices=choices, value=None),
                    "No episodes match filters.",
                    gr.update(value=empty_fig, visible=False),
                    "",
                )

        filter_outputs = [episode_table, episode_select, summary_md, reward_chart, steps_html]

        model_dd.change(fn=_apply_filters, inputs=[model_dd, dataset_dd, diff_dd], outputs=filter_outputs)
        dataset_dd.change(fn=_apply_filters, inputs=[model_dd, dataset_dd, diff_dd], outputs=filter_outputs)
        diff_dd.change(fn=_apply_filters, inputs=[model_dd, dataset_dd, diff_dd], outputs=filter_outputs)

        # ── Episode selection ────────────────────────────
        episode_select.change(
            fn=_render_episode,
            inputs=[episode_select],
            outputs=[summary_md, reward_chart, steps_html],
        )

        # ── Auto-load default ────────────────────────────
        default_summ, default_chart, default_steps = _render_episode(default_path)
        summary_md.value = default_summ
        reward_chart.value = default_chart.get("value") if isinstance(default_chart, dict) else default_chart
        reward_chart.visible = True
        steps_html.value = default_steps

    return tab
