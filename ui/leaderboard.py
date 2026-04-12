"""Leaderboard tab — stat cards, charts, category filter, ranked table."""
from __future__ import annotations

import gradio as gr
import pandas as pd
import plotly.graph_objects as go

from ui.data_loader import load_results, load_best_per_model_task
from ui.theme import CATEGORY_META, MODEL_COLORS

# Medal accent colors for top 3
_MEDAL_BORDER = {1: "#f59e0b", 2: "#94a3b8", 3: "#b45309"}
_MEDAL_LABEL  = {1: "LEADER", 2: "2nd", 3: "3rd"}


# ── Stat cards ─────────────────────────────────────────────────────────────────

def _build_stat_cards(df: pd.DataFrame) -> str:
    n_models = df["model"].nunique() if not df.empty else 0
    n_runs   = len(df) if not df.empty else 0
    n_tasks  = df["task_id"].nunique() if not df.empty and "task_id" in df.columns else 0
    n_cats   = df["category"].nunique() if not df.empty and "category" in df.columns else 0

    best = load_best_per_model_task(df) if not df.empty else pd.DataFrame()
    if not best.empty:
        avg_by_model = best.groupby("model")["reward"].mean()
        top_score = avg_by_model.max()
        top_model = avg_by_model.idxmax()
        top_label = f"{top_score:.0%}"
        top_sub   = top_model.split("/")[-1][:18]          # trim long model names
    else:
        top_label = "—"
        top_sub   = "no data yet"

    def card(value, label, sub, color):
        return (
            f'<div style="flex:1;min-width:110px;background:#fff;border:1px solid #e2e8f0;'
            f'border-top:3px solid {color};border-radius:10px;padding:16px 14px;text-align:center">'
            f'<div style="font-size:28px;font-weight:700;color:{color};line-height:1">{value}</div>'
            f'<div style="font-size:12px;color:#1e293b;font-weight:600;margin-top:6px">{label}</div>'
            f'<div style="font-size:11px;color:#94a3b8;margin-top:2px">{sub}</div>'
            f'</div>'
        )

    cards = "".join([
        card(str(n_models), "Models",     "tested so far",     "#0d9488"),
        card(str(n_runs),   "Total Runs", "episodes logged",   "#2563eb"),
        card(str(n_tasks),  "Tasks",      "unique task IDs",   "#7c3aed"),
        card(str(n_cats),   "Categories", "damage types",      "#d97706"),
        card(top_label,     "Best Score", top_sub,             "#059669"),
    ])
    return f'<div style="display:flex;gap:12px;margin:16px 0 20px 0;flex-wrap:wrap">{cards}</div>'


# ── Charts ─────────────────────────────────────────────────────────────────────

def _build_model_bar_chart(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    if df.empty:
        return fig
    best = load_best_per_model_task(df)
    avg  = best.groupby("model")["reward"].mean().sort_values()

    fig.add_trace(go.Bar(
        x=avg.values,
        y=avg.index.tolist(),
        orientation="h",
        marker_color=[MODEL_COLORS[i % len(MODEL_COLORS)] for i in range(len(avg))],
        text=[f"{v:.0%}" for v in avg.values],
        textposition="outside",
        cliponaxis=False,
    ))
    fig.update_layout(
        title=dict(text="Average Reward by Model", font=dict(size=14, color="#1e293b")),
        xaxis=dict(range=[0, 1.18], tickformat=".0%", showgrid=True, gridcolor="#e2e8f0"),
        yaxis=dict(showgrid=False),
        plot_bgcolor="#fff",
        paper_bgcolor="#fff",
        height=max(180, len(avg) * 44 + 80),
        margin=dict(l=10, r=64, t=44, b=10),
        font=dict(color="#475569", size=13),
    )
    return fig


def _build_category_heatmap(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    if df.empty or "category" not in df.columns:
        return fig
    best  = load_best_per_model_task(df)
    pivot = best.pivot_table(index="model", columns="category", values="reward", aggfunc="mean")

    text_vals = [
        [f"{v:.0%}" if pd.notna(v) else "—" for v in row]
        for row in pivot.values
    ]
    fig.add_trace(go.Heatmap(
        z=pivot.values.tolist(),
        x=pivot.columns.tolist(),
        y=pivot.index.tolist(),
        colorscale="RdYlGn",
        zmin=0, zmax=1,
        text=text_vals,
        texttemplate="%{text}",
        hovertemplate="Model: %{y}<br>Category: %{x}<br>Reward: %{z:.2f}<extra></extra>",
        showscale=True,
    ))
    fig.update_layout(
        title=dict(text="Performance Heatmap (Model × Category)", font=dict(size=14, color="#1e293b")),
        xaxis=dict(side="top"),
        plot_bgcolor="#fff",
        paper_bgcolor="#fff",
        height=max(220, len(pivot) * 48 + 120),
        margin=dict(l=10, r=10, t=60, b=10),
        font=dict(color="#475569", size=13),
    )
    return fig


# ── Leaderboard table ──────────────────────────────────────────────────────────

def _build_leaderboard_html(df: pd.DataFrame, category: str | None = None) -> str:
    if df.empty:
        return '<p style="color:#94a3b8">No benchmark data available.</p>'

    best = load_best_per_model_task(df)

    if category and category != "All":
        filtered = best[best["category"] == category]
        if filtered.empty:
            return f'<p style="color:#94a3b8">No data for category {category}.</p>'
    else:
        filtered = best

    models_overall = filtered.groupby("model")["reward"].mean().sort_values(ascending=False)

    diff_data, diff_counts = {}, {}
    for diff in ["easy", "medium", "hard"]:
        d = filtered[filtered["difficulty"] == diff] if "difficulty" in filtered.columns else pd.DataFrame()
        if not d.empty:
            db = load_best_per_model_task(d)
            diff_data[diff]   = db.groupby("model")["reward"].mean()
            diff_counts[diff] = (
                db.groupby("model")["task_id"].nunique()
                if "task_id" in db.columns else db.groupby("model").size()
            )
        else:
            diff_data[diff]   = pd.Series(dtype=float)
            diff_counts[diff] = pd.Series(dtype=int)

    header = (
        '<table style="width:100%;border-collapse:collapse;font-size:14px">'
        '<thead><tr style="background:#f8fafc;border-bottom:2px solid #e2e8f0">'
        '<th style="text-align:left;padding:12px 14px;color:#475569;font-weight:600;font-size:12px;text-transform:uppercase;letter-spacing:.05em">Rank</th>'
        '<th style="text-align:left;padding:12px 14px;color:#475569;font-weight:600;font-size:12px;text-transform:uppercase;letter-spacing:.05em">Model</th>'
        '<th style="text-align:left;padding:12px 14px;color:#475569;font-weight:600;font-size:12px;text-transform:uppercase;letter-spacing:.05em;width:32%">Overall Score</th>'
        '<th style="text-align:center;padding:12px 8px;color:#059669;font-weight:600;font-size:12px;text-transform:uppercase;letter-spacing:.05em;width:11%">Easy</th>'
        '<th style="text-align:center;padding:12px 8px;color:#d97706;font-weight:600;font-size:12px;text-transform:uppercase;letter-spacing:.05em;width:11%">Medium</th>'
        '<th style="text-align:center;padding:12px 8px;color:#dc2626;font-weight:600;font-size:12px;text-transform:uppercase;letter-spacing:.05em;width:11%">Hard</th>'
        '</tr></thead><tbody>'
    )

    def score_cell(val, color, n):
        if val == 0 and n == 0:
            return '<td style="text-align:center;padding:10px 8px;color:#cbd5e1;font-size:13px">—</td>'
        return (
            f'<td style="text-align:center;padding:10px 8px">'
            f'<span style="font-weight:700;color:{color};font-size:14px">{val:.0f}%</span>'
            f'<br><span style="font-size:11px;color:#94a3b8">{n}t</span></td>'
        )

    rows = []
    for rank, (model, overall) in enumerate(models_overall.items(), 1):
        pct        = overall * 100
        bar_color  = MODEL_COLORS[(rank - 1) % len(MODEL_COLORS)]
        medal_border = _MEDAL_BORDER.get(rank, "transparent")

        easy_val = diff_data["easy"].get(model, 0) * 100
        med_val  = diff_data["medium"].get(model, 0) * 100
        hard_val = diff_data["hard"].get(model, 0) * 100
        easy_n   = int(diff_counts["easy"].get(model, 0))
        med_n    = int(diff_counts["medium"].get(model, 0))
        hard_n   = int(diff_counts["hard"].get(model, 0))

        # Rank cell
        if rank == 1:
            rank_html = (
                f'<td style="padding:12px 14px;border-left:4px solid {medal_border}">'
                f'<span style="background:#fef3c7;color:#92400e;font-size:11px;font-weight:700;'
                f'padding:2px 8px;border-radius:20px">LEADER</span></td>'
            )
        elif rank in _MEDAL_BORDER:
            rank_html = (
                f'<td style="padding:12px 14px;border-left:4px solid {medal_border};'
                f'color:{medal_border};font-weight:700;font-size:14px">#{rank}</td>'
            )
        else:
            rank_html = (
                f'<td style="padding:12px 14px;border-left:4px solid transparent;'
                f'color:#94a3b8;font-size:13px">#{rank}</td>'
            )

        # Model name cell
        model_html = (
            f'<td style="padding:12px 14px;color:#1e293b;font-weight:600;font-size:13px">'
            f'{model}</td>'
        )

        # Overall bar cell
        bar_html = (
            f'<td style="padding:10px 14px">'
            f'<div style="display:flex;align-items:center;gap:10px">'
            f'<div style="flex:1;height:20px;background:#f1f5f9;border-radius:4px;overflow:hidden">'
            f'<div style="width:{pct:.1f}%;height:100%;background:{bar_color};border-radius:4px;'
            f'transition:width .3s ease"></div></div>'
            f'<span style="font-weight:700;color:#1e293b;font-size:14px;min-width:38px;text-align:right">'
            f'{pct:.0f}%</span>'
            f'<span style="font-size:11px;color:#94a3b8;min-width:40px;text-align:right">'
            f'{easy_n+med_n+hard_n}t</span>'
            f'</div></td>'
        )

        bg = "#fffbeb" if rank == 1 else ("#f8fafc" if rank % 2 == 0 else "#fff")
        rows.append(
            f'<tr style="border-bottom:1px solid #f1f5f9;background:{bg}">'
            + rank_html + model_html + bar_html
            + score_cell(easy_val, "#059669", easy_n)
            + score_cell(med_val,  "#d97706", med_n)
            + score_cell(hard_val, "#dc2626", hard_n)
            + '</tr>'
        )

    return header + "\n".join(rows) + '</tbody></table>'


def _build_subtitle(df: pd.DataFrame, category: str | None) -> str:
    if not category or category == "All":
        return "Overall scores across all categories and difficulties."
    meta = CATEGORY_META.get(category, {})
    return f"**{meta.get('name', category)}**: {meta.get('desc', '')}"


# ── Tab builder ────────────────────────────────────────────────────────────────

def create_leaderboard_tab(benchmark_dir: str = "outputs/benchmark") -> gr.Blocks:
    df = load_results(benchmark_dir)
    categories = list(CATEGORY_META.keys())

    with gr.Blocks() as tab:

        if df.empty:
            gr.Markdown(
                "**No results yet.** Run the benchmark first:\n"
                "```\npython -m tools.benchmark_runner\n```"
            )
            return tab

        # ── Stat cards ──────────────────────────────────────
        gr.HTML(_build_stat_cards(df))

        # ── Charts ─────────────────────────────────────────
        with gr.Row():
            with gr.Column(scale=1):
                gr.Plot(value=_build_model_bar_chart(df), label="Avg Reward by Model")
            with gr.Column(scale=1):
                gr.Plot(value=_build_category_heatmap(df), label="Category Heatmap")

        # ── Category filter ─────────────────────────────────
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
        subtitle   = gr.Markdown(value=_build_subtitle(df, None))
        leaderboard = gr.HTML(value=_build_leaderboard_html(df))

        def on_filter(cat_code):
            cat = cat_code if cat_code != "All" else None
            return _build_subtitle(df, cat), _build_leaderboard_html(df, cat)

        cat_radio.change(fn=on_filter, inputs=[cat_radio], outputs=[subtitle, leaderboard])

    return tab
