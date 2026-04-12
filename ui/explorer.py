"""Episode explorer tab — step-by-step replay of agent cleaning episodes."""
from __future__ import annotations

import gradio as gr
import pandas as pd
import plotly.graph_objects as go

from ui.data_loader import list_episode_files, load_episode_log


_ACTION_COLORS = {
    "transform": "#0d9488",
    "explore":   "#2563eb",
    "validate":  "#d97706",
    "done":      "#059669",
    "undo":      "#dc2626",
}


# ── Top stats bar ──────────────────────────────────────────────────────────────

def _build_episode_stats(episodes: list[dict]) -> str:
    total = len(episodes)
    if total == 0:
        return ""
    rewards = [ep.get("final_reward", 0) for ep in episodes if ep.get("status") != "failed"]
    avg_r   = sum(rewards) / len(rewards) if rewards else 0
    passed  = sum(1 for r in rewards if r >= 0.5)
    models  = len({ep.get("model", "") for ep in episodes})

    def card(val, label, color):
        return (
            f'<div style="flex:1;min-width:90px;background:#fff;border:1px solid #e2e8f0;'
            f'border-top:3px solid {color};border-radius:10px;padding:14px 12px;text-align:center">'
            f'<div style="font-size:24px;font-weight:700;color:{color}">{val}</div>'
            f'<div style="font-size:11px;color:#64748b;font-weight:600;margin-top:4px;text-transform:uppercase;letter-spacing:.05em">{label}</div>'
            f'</div>'
        )

    cards = "".join([
        card(str(total),         "Episodes",    "#0d9488"),
        card(str(models),        "Models",      "#2563eb"),
        card(f"{avg_r:.0%}",     "Avg Reward",  "#7c3aed"),
        card(str(passed),        "Passed ≥50%", "#059669"),
    ])
    return f'<div style="display:flex;gap:10px;margin:0 0 20px 0;flex-wrap:wrap">{cards}</div>'


# ── Episode list table ─────────────────────────────────────────────────────────

def _build_episode_table_html(episodes: list[dict], selected_path: str = "") -> str:
    if not episodes:
        return '<p style="color:#94a3b8;padding:12px">No episodes match filters.</p>'

    diff_colors = {"easy": "#059669", "medium": "#d97706", "hard": "#dc2626"}
    rows = []

    for i, ep in enumerate(episodes):
        reward    = ep.get("final_reward", 0.0)
        steps     = ep.get("steps", 0)
        is_failed = ep.get("status") == "failed"
        is_sel    = ep.get("path", "") == selected_path
        diff      = ep.get("difficulty", "")
        diff_color = diff_colors.get(diff, "#64748b")

        bg           = "#f0fdf4" if is_sel else ("#fef2f2" if is_failed else ("#f8fafc" if i % 2 == 0 else "#fff"))
        left_border  = "3px solid #0d9488" if is_sel else "3px solid transparent"
        reward_color = "#94a3b8" if is_failed else ("#059669" if reward >= 0.5 else "#d97706" if reward >= 0.2 else "#dc2626")

        # Reward bar mini
        if is_failed:
            reward_cell = '<span style="background:#fecaca;color:#991b1b;font-size:10px;padding:2px 7px;border-radius:4px;font-weight:700">FAILED</span>'
        else:
            bar_w = int(reward * 56)
            reward_cell = (
                f'<div style="display:flex;align-items:center;gap:6px;justify-content:flex-end">'
                f'<div style="width:56px;height:6px;background:#f1f5f9;border-radius:3px;overflow:hidden">'
                f'<div style="width:{bar_w}px;height:100%;background:{reward_color};border-radius:3px"></div></div>'
                f'<span style="font-weight:700;color:{reward_color};font-size:13px;min-width:34px;text-align:right">'
                f'{reward:.2f}</span></div>'
            )

        rows.append(
            f'<tr style="background:{bg};border-bottom:1px solid #e2e8f0;border-left:{left_border}">'
            f'<td style="padding:10px 10px;font-size:12px;color:#1e293b;font-weight:600;max-width:120px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap">{ep.get("model","?")}</td>'
            f'<td style="padding:10px 8px;font-size:12px;color:#334155">{ep.get("dataset","?")}</td>'
            f'<td style="padding:10px 8px"><span style="color:{diff_color};font-weight:700;font-size:11px;text-transform:uppercase;letter-spacing:.04em">{diff}</span></td>'
            f'<td style="padding:10px 10px;text-align:right">{reward_cell}</td>'
            f'<td style="padding:10px 8px;text-align:center;font-size:12px;color:#94a3b8">{steps}</td>'
            f'</tr>'
        )

    header = (
        '<div style="border:1px solid #e2e8f0;border-radius:8px;overflow:hidden">'
        '<table style="width:100%;border-collapse:collapse;font-size:12px">'
        '<thead><tr style="background:#f1f5f9;border-bottom:2px solid #e2e8f0">'
        '<th style="padding:10px;text-align:left;color:#475569;font-weight:700;font-size:11px;text-transform:uppercase;letter-spacing:.06em">Model</th>'
        '<th style="padding:10px 8px;text-align:left;color:#475569;font-weight:700;font-size:11px;text-transform:uppercase;letter-spacing:.06em">Dataset</th>'
        '<th style="padding:10px 8px;text-align:left;color:#475569;font-weight:700;font-size:11px;text-transform:uppercase;letter-spacing:.06em">Diff</th>'
        '<th style="padding:10px;text-align:right;color:#475569;font-weight:700;font-size:11px;text-transform:uppercase;letter-spacing:.06em">Reward</th>'
        '<th style="padding:10px 8px;text-align:center;color:#475569;font-weight:700;font-size:11px;text-transform:uppercase;letter-spacing:.06em">Steps</th>'
        '</tr></thead>'
        '<tbody style="max-height:420px;overflow-y:auto;display:block">'
    )
    # tbody display:block breaks table layout in some browsers — use wrapper div instead
    header = (
        '<div style="max-height:420px;overflow-y:auto;border:1px solid #e2e8f0;border-radius:8px">'
        '<table style="width:100%;border-collapse:collapse;font-size:12px">'
        '<thead style="position:sticky;top:0;z-index:1"><tr style="background:#f1f5f9;border-bottom:2px solid #e2e8f0">'
        '<th style="padding:10px;text-align:left;color:#475569;font-weight:700;font-size:11px;text-transform:uppercase;letter-spacing:.06em">Model</th>'
        '<th style="padding:10px 8px;text-align:left;color:#475569;font-weight:700;font-size:11px;text-transform:uppercase;letter-spacing:.06em">Dataset</th>'
        '<th style="padding:10px 8px;text-align:left;color:#475569;font-weight:700;font-size:11px;text-transform:uppercase;letter-spacing:.06em">Diff</th>'
        '<th style="padding:10px;text-align:right;color:#475569;font-weight:700;font-size:11px;text-transform:uppercase;letter-spacing:.06em">Reward</th>'
        '<th style="padding:10px 8px;text-align:center;color:#475569;font-weight:700;font-size:11px;text-transform:uppercase;letter-spacing:.06em">Steps</th>'
        '</tr></thead><tbody>'
    )
    return header + "\n".join(rows) + '</tbody></table></div>'


# ── Reward chart ───────────────────────────────────────────────────────────────

def _build_reward_chart(events: list[dict]) -> go.Figure | None:
    steps = [e for e in events if e.get("event") == "step"]
    if not steps:
        return None

    step_nums    = [s["step"] for s in steps]
    rewards      = [s.get("reward", 0.0) for s in steps]
    action_types = [s.get("action_type", "") for s in steps]
    colors       = [_ACTION_COLORS.get(a, "#94a3b8") for a in action_types]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=step_nums, y=rewards,
        mode="lines+markers",
        line=dict(color="#0d9488", width=2),
        marker=dict(color=colors, size=9, line=dict(color="#fff", width=1.5)),
        hovertext=[f"Step {n} · {a}<br>Reward: {r:.4f}" for n, a, r in zip(step_nums, action_types, rewards)],
        hoverinfo="text",
        fill="tozeroy",
        fillcolor="rgba(13,148,136,0.07)",
    ))
    fig.update_layout(
        plot_bgcolor="#fff",
        paper_bgcolor="#fff",
        font=dict(color="#475569"),
        margin=dict(l=10, r=10, t=36, b=10),
        height=220,
        xaxis=dict(title="Step", showgrid=False, zeroline=False),
        yaxis=dict(title="Reward", range=[0, 1.05], showgrid=True, gridcolor="#e2e8f0"),
        title=dict(text="Score Progression", font=dict(size=13, color="#1e293b")),
    )
    return fig


# ── Step trace ─────────────────────────────────────────────────────────────────

def _format_steps_html(events: list[dict]) -> str:
    steps = [e for e in events if e.get("event") == "step"]
    if not steps:
        return "<p style='color:#94a3b8;padding:12px'>No step data available.</p>"

    html_parts = []
    for s in steps:
        action  = s.get("action_type", "unknown")
        color   = _ACTION_COLORS.get(action, "#94a3b8")
        step_n  = s.get("step", "?")
        reward  = s.get("reward", 0.0)
        fixed   = s.get("errors_fixed", 0)
        total   = s.get("errors_total", 0)
        content = s.get("action_content", "")
        latency = s.get("llm_latency_s", 0)

        reward_color = "#059669" if reward >= 0.5 else "#d97706" if reward >= 0.2 else "#dc2626"

        # Code block for transform / explore
        code_block = ""
        if content and action in ("transform", "explore"):
            escaped = content.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
            bg = "#0f172a" if action == "transform" else "#f8fafc"
            text_color = "#e2e8f0" if action == "transform" else "#334155"
            border = "none" if action == "transform" else "1px solid #e2e8f0"
            code_block = (
                f'<pre style="background:{bg};border:{border};border-radius:6px;'
                f'padding:12px 14px;margin:10px 0 0 0;font-size:12px;color:{text_color};'
                f'overflow-x:auto;line-height:1.6;font-family:\'SF Mono\',Monaco,monospace">'
                f'{escaped}</pre>'
            )

        # Latency tag
        lat_tag = (
            f'<span style="font-size:11px;color:#94a3b8;margin-left:8px">{latency:.1f}s</span>'
            if latency > 0 else ""
        )

        html_parts.append(
            f'<div style="background:#fff;border:1px solid #e2e8f0;border-left:3px solid {color};'
            f'border-radius:8px;padding:14px 16px;margin-bottom:8px">'
            f'<div style="display:flex;justify-content:space-between;align-items:center">'
            f'<div style="display:flex;align-items:center;gap:8px">'
            f'<span style="background:{color}18;color:{color};font-size:11px;font-weight:700;'
            f'padding:2px 9px;border-radius:4px;text-transform:uppercase;letter-spacing:.05em">{action}</span>'
            f'<span style="font-size:12px;color:#94a3b8">step {step_n}</span>'
            f'{lat_tag}'
            f'</div>'
            f'<div style="font-size:12px;color:#64748b">'
            f'Reward: <b style="color:{reward_color}">{reward:.4f}</b>'
            f'&nbsp;·&nbsp;Fixed: <b style="color:#059669">{fixed}</b>/{total}'
            f'</div></div>'
            f'{code_block}'
            f'</div>'
        )

    # Final summary card
    end = next((e for e in events if e.get("event") == "task_end"), None)
    if end:
        fr      = end.get("final_reward", 0.0)
        tsteps  = end.get("total_steps", 0)
        elapsed = end.get("elapsed_s", 0.0)
        rc      = "#059669" if fr >= 0.5 else "#d97706" if fr >= 0.2 else "#dc2626"
        outcome = "PASS" if fr >= 0.5 else "PARTIAL" if fr >= 0.2 else "FAIL"
        out_bg  = "#f0fdf4" if fr >= 0.5 else "#fffbeb" if fr >= 0.2 else "#fef2f2"
        html_parts.append(
            f'<div style="background:{out_bg};border:2px solid {rc};border-radius:8px;'
            f'padding:16px 18px;margin-top:4px;display:flex;justify-content:space-between;align-items:center">'
            f'<div style="display:flex;align-items:center;gap:10px">'
            f'<span style="background:{rc};color:#fff;font-size:11px;font-weight:700;'
            f'padding:3px 10px;border-radius:4px;letter-spacing:.06em">{outcome}</span>'
            f'<span style="font-weight:700;color:#1e293b;font-size:15px">Episode Complete</span>'
            f'</div>'
            f'<div style="font-size:13px;color:#64748b">'
            f'Final: <b style="color:{rc};font-size:16px">{fr:.4f}</b>'
            f'&nbsp;·&nbsp;{tsteps} steps'
            f'&nbsp;·&nbsp;{elapsed:.1f}s'
            f'</div></div>'
        )

    return "\n".join(html_parts)


# ── Episode render ─────────────────────────────────────────────────────────────

def _render_episode(path: str):
    if not path:
        empty = go.Figure()
        empty.update_layout(height=220)
        return "Select an episode from the list.", gr.update(value=empty, visible=False), ""

    events = load_episode_log(path)
    if not events:
        empty = go.Figure()
        empty.update_layout(height=220)
        return "Failed to load episode.", gr.update(value=empty, visible=False), ""

    start = next((e for e in events if e.get("event") == "task_start"), {})
    end   = next((e for e in events if e.get("event") == "task_end"),   {})

    model        = start.get("model") or "unknown"
    task_id      = start.get("task_id") or "unknown"
    final_reward = end.get("final_reward", end.get("reward", 0.0))
    total_steps  = end.get("total_steps", len([e for e in events if e.get("event") == "step"]))
    elapsed      = end.get("elapsed_s", 0.0)
    rc           = "#059669" if final_reward >= 0.5 else "#d97706" if final_reward >= 0.2 else "#dc2626"

    summary = (
        f"### {task_id}\n"
        f"**Model:** `{model}` &nbsp;·&nbsp; "
        f"**Reward:** <span style='color:{rc}'>{final_reward:.4f}</span> &nbsp;·&nbsp; "
        f"**Steps:** {total_steps} &nbsp;·&nbsp; **Time:** {elapsed:.1f}s"
    )

    chart       = _build_reward_chart(events)
    steps_html  = _format_steps_html(events)
    return summary, gr.update(value=chart, visible=chart is not None), steps_html


def _parse_episode_meta(ep: dict) -> dict:
    task_id   = ep.get("task_id", "")
    tid_parts = task_id.rsplit("_", 2)
    difficulty = tid_parts[1] if len(tid_parts) >= 3 else "unknown"
    dataset    = tid_parts[0] if len(tid_parts) >= 3 else task_id
    return {**ep, "dataset": dataset, "difficulty": difficulty}


# ── Tab builder ────────────────────────────────────────────────────────────────

def create_explorer_tab(log_dir: str = "outputs/benchmark/episodes") -> gr.Blocks:
    raw_episodes = list_episode_files(log_dir)
    all_episodes = [_parse_episode_meta(ep) for ep in raw_episodes]

    with gr.Blocks() as tab:
        gr.Markdown("## Episode Explorer")

        if not all_episodes:
            gr.Markdown(
                f"**No episodes found in `{log_dir}/`.**  "
                "Run the benchmark to generate episodes."
            )
            return tab

        gr.HTML(_build_episode_stats(all_episodes))

        all_models   = sorted({ep["model"]    for ep in all_episodes})
        all_datasets = sorted({ep["dataset"]  for ep in all_episodes})

        nonzero     = [ep for ep in all_episodes if ep.get("final_reward", 0) > 0]
        default_ep  = max(nonzero or all_episodes, key=lambda e: e.get("steps", 0))
        default_path = default_ep["path"]

        # ── Filters ──────────────────────────────────────
        with gr.Row():
            model_dd   = gr.Dropdown(choices=["All"] + all_models,   value="All", label="Model",      scale=2)
            dataset_dd = gr.Dropdown(choices=["All"] + all_datasets, value="All", label="Dataset",    scale=2)
            diff_dd    = gr.Dropdown(choices=["All", "easy", "medium", "hard"], value="All", label="Difficulty", scale=1)

        # ── 2-column layout ──────────────────────────────
        with gr.Row():
            with gr.Column(scale=2):
                episode_table  = gr.HTML(value=_build_episode_table_html(all_episodes, default_path))
                episode_select = gr.Dropdown(
                    choices=[
                        (f"{ep['model']} · {ep['dataset']} · {ep['difficulty']} · {ep['final_reward']:.3f}", ep["path"])
                        for ep in all_episodes
                    ],
                    value=default_path,
                    label="Select episode",
                )

            with gr.Column(scale=3):
                summary_md  = gr.Markdown()
                reward_chart = gr.Plot(visible=False)
                steps_html  = gr.HTML()

        # ── Filter callback ───────────────────────────────
        def _apply_filters(model, dataset, difficulty):
            filtered = all_episodes
            if model     != "All": filtered = [e for e in filtered if e["model"]      == model]
            if dataset   != "All": filtered = [e for e in filtered if e["dataset"]    == dataset]
            if difficulty != "All": filtered = [e for e in filtered if e["difficulty"] == difficulty]

            filtered.sort(key=lambda e: (-int(e.get("final_reward", 0) > 0), -e.get("steps", 0)))

            new_path = ""
            if filtered:
                nz = [e for e in filtered if e.get("final_reward", 0) > 0]
                best = max(nz, key=lambda e: e.get("steps", 0)) if nz else filtered[0]
                new_path = best["path"]

            table_html = _build_episode_table_html(filtered, new_path)
            choices    = [(f"{ep['model']} · {ep['dataset']} · {ep['difficulty']} · {ep['final_reward']:.3f}", ep["path"])
                          for ep in filtered]

            if new_path:
                summ, chart_u, steps = _render_episode(new_path)
                return table_html, gr.update(choices=choices, value=new_path), summ, chart_u, steps
            else:
                ef = go.Figure(); ef.update_layout(height=220)
                return table_html, gr.update(choices=choices, value=None), "No episodes match.", gr.update(value=ef, visible=False), ""

        filter_outputs = [episode_table, episode_select, summary_md, reward_chart, steps_html]
        for dd in [model_dd, dataset_dd, diff_dd]:
            dd.change(fn=_apply_filters, inputs=[model_dd, dataset_dd, diff_dd], outputs=filter_outputs)

        episode_select.change(fn=_render_episode, inputs=[episode_select], outputs=[summary_md, reward_chart, steps_html])

        # Auto-load default
        ds, dc, dsteps = _render_episode(default_path)
        summary_md.value   = ds
        reward_chart.value = dc.get("value") if isinstance(dc, dict) else dc
        reward_chart.visible = True
        steps_html.value   = dsteps

    return tab
