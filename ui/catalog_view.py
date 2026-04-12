"""Dataset catalog browser tab — two-pane: table left, rules right."""
from __future__ import annotations

import gradio as gr
import pandas as pd

from ui.data_loader import load_catalog


def _catalog_to_df(catalog: dict) -> pd.DataFrame:
    rows = []
    for dataset_id, entry in catalog.items():
        rules = entry.get("rules", [])
        rule_types = list({r.get("type", "") for r in rules})
        rows.append({
            "Dataset": dataset_id,
            "Domain": entry.get("domain", ""),
            "Rows": entry.get("rows", ""),
            "Cols": entry.get("cols", ""),
            "Rules": len(rules),
            "Rule Types": ", ".join(sorted(rule_types)),
        })
    return pd.DataFrame(rows).sort_values("Dataset").reset_index(drop=True)


def _format_rules_html(rules: list[dict]) -> str:
    if not rules:
        return "<p style='color:#94a3b8'>No semantic rules inferred.</p>"

    type_colors = {
        "range": "#0d9488", "enum": "#2563eb", "regex": "#d97706",
        "not_null": "#dc2626", "unique": "#7c3aed", "dtype": "#059669",
        "cross_column": "#db2777",
    }

    lines = []
    for r in rules:
        rtype = r.get("type", "unknown")
        col = r.get("column", r.get("columns", ""))
        color = type_colors.get(rtype, "#94a3b8")

        detail = ""
        if rtype == "range":
            detail = f"[{r.get('min', '?')}, {r.get('max', '?')}]"
        elif rtype == "enum":
            vals = r.get("values", r.get("allowed_values", []))
            preview = ", ".join(str(v) for v in vals[:6])
            if len(vals) > 6:
                preview += f" +{len(vals) - 6} more"
            detail = f"{{{preview}}}"
        elif rtype == "regex":
            detail = f"<code style='background:#f1f5f9;padding:1px 5px;border-radius:3px'>{r.get('pattern', '')}</code>"
        elif rtype == "dtype":
            detail = r.get("expected_dtype", "?")
        elif rtype == "cross_column":
            detail = r.get("relationship", "?")

        lines.append(
            f'<div style="padding:8px 12px;margin:4px 0;background:#f8fafc;border-radius:6px;'
            f'border-left:3px solid {color};font-size:13px">'
            f'<span style="color:{color};font-weight:600">{rtype.title()}</span> '
            f'<code style="color:#1e293b;font-weight:500">{col}</code>'
            f'{f"<br><span style=\'color:#64748b;font-size:12px\'>{detail}</span>" if detail else ""}'
            f'</div>'
        )
    return "\n".join(lines)


def _build_rules_panel(catalog: dict, dataset_id: str | None) -> str:
    if not dataset_id or dataset_id not in catalog:
        return '<p style="color:#94a3b8;padding:20px">Select a dataset from the table.</p>'

    entry = catalog[dataset_id]
    rules = entry.get("rules", [])

    header = (
        f'<div style="padding:4px 0">'
        f'<h3 style="color:#1e293b;margin:0 0 4px 0;font-size:18px">{dataset_id}</h3>'
        f'<p style="color:#64748b;font-size:13px;margin:0 0 16px 0">'
        f'<span style="background:#f1f5f9;padding:2px 8px;border-radius:4px;margin-right:8px">{entry.get("domain", "N/A")}</span>'
        f'{entry.get("rows", "?")} rows &times; {entry.get("cols", "?")} cols'
        f'</p>'
        f'<h4 style="color:#475569;font-size:14px;margin:0 0 10px 0">Semantic Rules ({len(rules)})</h4>'
    )

    return header + _format_rules_html(rules) + '</div>'


def create_catalog_tab() -> gr.Blocks:
    catalog = load_catalog()

    with gr.Blocks() as tab:
        gr.Markdown("## Dataset Catalog")
        gr.Markdown(f"*{len(catalog)} datasets registered in the benchmark suite.*")

        if not catalog:
            gr.Markdown("**No catalog found at `datasets/catalog.json`.**")
            return tab

        catalog_df = _catalog_to_df(catalog)
        default_ds = sorted(catalog.keys())[0]

        with gr.Row():
            # Left pane — table
            with gr.Column(scale=3):
                dataset_dd = gr.Dropdown(
                    choices=sorted(catalog.keys()),
                    label="▼ Select Dataset",
                    info="Pick a dataset to view its semantic rules on the right",
                    value=default_ds,
                    elem_classes=["interactive-dropdown"],
                )
                table = gr.DataFrame(
                    value=catalog_df,
                    label="Datasets",
                    interactive=False,
                )

            # Right pane — rules
            with gr.Column(scale=2):
                rules_panel = gr.HTML(
                    value=_build_rules_panel(catalog, default_ds),
                )

        dataset_dd.change(
            fn=lambda ds: _build_rules_panel(catalog, ds),
            inputs=[dataset_dd],
            outputs=[rules_panel],
        )

    return tab
