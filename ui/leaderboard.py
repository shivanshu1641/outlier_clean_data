"""Leaderboard tab — category filter pills + full model score table."""
from __future__ import annotations

import gradio as gr
import pandas as pd

from ui.data_loader import load_results, load_best_per_model_task
from ui.theme import CATEGORY_META, MODEL_COLORS



def _build_leaderboard_html(df: pd.DataFrame, category: str | None = None) -> str:
    """Full leaderboard table: model | overall | easy | medium | hard with bar fills."""
    if df.empty:
        return '<p style="color:#94a3b8">No benchmark data available.</p>'

    best = load_best_per_model_task(df)

    if category and category != "All":
        filtered = best[best["category"] == category]
        if filtered.empty:
            return f'<p style="color:#94a3b8">No data for category {category}.</p>'
    else:
        filtered = best

    # Aggregate: overall + per-difficulty
    diff_colors = {"easy": "#059669", "medium": "#d97706", "hard": "#dc2626"}

    models_overall = filtered.groupby("model")["reward"].mean().sort_values(ascending=False)

    # Per-difficulty breakdown: scores + task counts per model
    diff_data = {}
    diff_counts = {}
    for diff in ["easy", "medium", "hard"]:
        d = filtered[filtered["difficulty"] == diff]
        if not d.empty:
            diff_best = load_best_per_model_task(d)
            diff_data[diff] = diff_best.groupby("model")["reward"].mean()
            diff_counts[diff] = diff_best.groupby("model")["task_id"].nunique() if "task_id" in diff_best.columns else diff_best.groupby("model").size()
        else:
            diff_data[diff] = pd.Series(dtype=float)
            diff_counts[diff] = pd.Series(dtype=int)

    # Build table
    header = (
        '<table style="width:100%;border-collapse:collapse;font-size:14px">'
        '<thead><tr style="border-bottom:2px solid #e2e8f0">'
        '<th style="text-align:left;padding:10px 12px;color:#1e293b;font-weight:600">Model</th>'
        '<th style="text-align:left;padding:10px 12px;color:#1e293b;font-weight:600;width:35%">Overall</th>'
        '<th style="text-align:center;padding:10px 12px;color:#059669;font-weight:600;width:12%">Easy</th>'
        '<th style="text-align:center;padding:10px 12px;color:#d97706;font-weight:600;width:12%">Medium</th>'
        '<th style="text-align:center;padding:10px 12px;color:#dc2626;font-weight:600;width:12%">Hard</th>'
        '</tr></thead><tbody>'
    )

    rows = []
    for rank, (model, overall) in enumerate(models_overall.items(), 1):
        pct = overall * 100
        ci = rank - 1
        bar_color = MODEL_COLORS[ci % len(MODEL_COLORS)]

        easy_val = diff_data["easy"].get(model, 0) * 100
        med_val = diff_data["medium"].get(model, 0) * 100
        hard_val = diff_data["hard"].get(model, 0) * 100
        easy_n = int(diff_counts["easy"].get(model, 0))
        med_n = int(diff_counts["medium"].get(model, 0))
        hard_n = int(diff_counts["hard"].get(model, 0))

        def score_cell(val, color, n):
            if val == 0 and n == 0:
                return '<td style="text-align:center;padding:8px 12px;color:#cbd5e1">—</td>'
            return (
                f'<td style="text-align:center;padding:8px 12px">'
                f'<span style="font-weight:600;color:{color}">{val:.0f}%</span>'
                f'<br><span style="font-size:11px;color:#94a3b8">{n} tasks</span></td>'
            )

        bg = "#f8fafc" if rank % 2 == 0 else "#fff"
        rows.append(
            f'<tr style="border-bottom:1px solid #f1f5f9;background:{bg}">'
            f'<td style="padding:10px 12px;color:#1e293b;font-weight:500">'
            f'<span style="color:#94a3b8;font-size:12px;margin-right:8px">#{rank}</span>{model}</td>'
            f'<td style="padding:10px 12px">'
            f'<div style="display:flex;align-items:center;gap:10px">'
            f'<div style="flex:1;height:22px;background:#f1f5f9;border-radius:4px;overflow:hidden">'
            f'<div style="width:{pct}%;height:100%;background:{bar_color};border-radius:4px"></div></div>'
            f'<span style="font-weight:600;color:#1e293b;font-size:13px;min-width:40px;text-align:right">{pct:.0f}%</span>'
            f'<span style="font-size:11px;color:#94a3b8;min-width:45px;text-align:right">{easy_n+med_n+hard_n} tasks</span>'
            f'</div></td>'
            + score_cell(easy_val, "#059669", easy_n)
            + score_cell(med_val, "#d97706", med_n)
            + score_cell(hard_val, "#dc2626", hard_n)
            + '</tr>'
        )

    return header + "\n".join(rows) + '</tbody></table>'


def _build_subtitle(df: pd.DataFrame, category: str | None) -> str:
    if not category or category == "All":
        return "Overall scores across all categories and difficulties."
    meta = CATEGORY_META.get(category, {})
    return f"**{meta.get('name', category)}** — {meta.get('desc', '')}"


def create_leaderboard_tab(benchmark_dir: str = "outputs/benchmark") -> gr.Blocks:
    df = load_results(benchmark_dir)
    categories = list(CATEGORY_META.keys())

    with gr.Blocks() as tab:
        gr.Markdown("## Benchmark Leaderboard")

        if df.empty:
            gr.Markdown(
                "**No results found.** Run the benchmark first:\n"
                "```\n./run_benchmark.sh\n```"
            )
            return tab

        # Build radio choices with task counts
        total_tasks = df["task_id"].nunique() if "task_id" in df.columns else len(df)
        radio_choices = [(f"All ({total_tasks})", "All")]
        for c in categories:
            cat_df = df[df["category"] == c] if "category" in df.columns else pd.DataFrame()
            count = cat_df["task_id"].nunique() if not cat_df.empty and "task_id" in cat_df.columns else len(cat_df)
            radio_choices.append((f"{c}  {CATEGORY_META[c]['name']} ({count})", c))

        cat_radio = gr.Radio(
            choices=radio_choices,
            label="Filter by Category",
            value="All",
            elem_classes=["category-radio"],
        )

        # Description of selected category
        subtitle = gr.Markdown(value=_build_subtitle(df, None))

        # Show full leaderboard by default
        leaderboard = gr.HTML(value=_build_leaderboard_html(df))

        def on_filter(cat_code):
            cat = cat_code if cat_code != "All" else None
            return (
                _build_subtitle(df, cat),
                _build_leaderboard_html(df, cat),
            )

        cat_radio.change(
            fn=on_filter,
            inputs=[cat_radio],
            outputs=[subtitle, leaderboard],
        )

        # Stats
        n_runs = len(df)
        n_models = df["model"].nunique()
        n_tasks = df["task_id"].nunique() if "task_id" in df.columns else len(df)
        n_cats = df["category"].nunique() if "category" in df.columns else 0
        gr.Markdown(
            f"*{n_runs} runs | {n_models} models | {n_tasks} tasks | {n_cats} categories*"
        )

    return tab
