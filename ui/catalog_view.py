# ui/catalog_view.py
"""Dataset catalog browser tab."""
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
            "dataset_id": dataset_id,
            "domain": entry.get("domain", ""),
            "rows": entry.get("rows", ""),
            "cols": entry.get("cols", ""),
            "size_class": entry.get("size_class", ""),
            "num_rules": len(rules),
            "rule_types": ", ".join(sorted(rule_types)),
            "filename": entry.get("filename", ""),
        })
    return pd.DataFrame(rows).sort_values("dataset_id").reset_index(drop=True)


def _format_rules(rules: list[dict]) -> str:
    if not rules:
        return "*No semantic rules inferred for this dataset.*"
    lines = []
    for r in rules:
        rtype = r.get("type", "unknown")
        col = r.get("column", r.get("columns", ""))
        source = r.get("source", "")
        if rtype == "range":
            lines.append(f"- **Range** `{col}`: [{r.get('min', '?')}, {r.get('max', '?')}] *(source: {source})*")
        elif rtype == "enum":
            vals = r.get("values", r.get("allowed_values", []))
            preview = ", ".join(str(v) for v in vals[:10])
            if len(vals) > 10:
                preview += f" ... ({len(vals)} total)"
            lines.append(f"- **Enum** `{col}`: {{{preview}}} *(source: {source})*")
        elif rtype == "regex":
            lines.append(f"- **Regex** `{col}`: `{r.get('pattern', '')}` *(source: {source})*")
        elif rtype == "not_null":
            lines.append(f"- **NotNull** `{col}` *(source: {source})*")
        elif rtype == "unique":
            lines.append(f"- **Unique** `{col}` *(source: {source})*")
        elif rtype == "dtype":
            lines.append(f"- **Dtype** `{col}`: {r.get('expected_dtype', '?')} *(source: {source})*")
        elif rtype == "cross_column":
            lines.append(f"- **CrossColumn** `{col}`: {r.get('relationship', '?')} *(source: {source})*")
        else:
            lines.append(f"- **{rtype}** `{col}` *(source: {source})*")
    return "\n".join(lines)


def create_catalog_tab() -> gr.Blocks:
    catalog = load_catalog()

    with gr.Blocks() as tab:
        gr.Markdown("## Dataset Catalog")
        gr.Markdown(f"*{len(catalog)} datasets registered.*")

        if not catalog:
            gr.Markdown("**No catalog found at `datasets/catalog.json`.**")
            return tab

        catalog_df = _catalog_to_df(catalog)
        table = gr.DataFrame(value=catalog_df, label="Datasets", interactive=False)

        dataset_dd = gr.Dropdown(choices=sorted(catalog.keys()), label="Select dataset to view rules")
        rules_display = gr.Markdown("Select a dataset above to see its semantic rules.")

        def show_rules(dataset_id):
            if not dataset_id or dataset_id not in catalog:
                return "Select a dataset."
            entry = catalog[dataset_id]
            rules = entry.get("rules", [])
            header = (
                f"### {dataset_id}\n\n"
                f"**Domain:** {entry.get('domain', 'N/A')} | "
                f"**Rows:** {entry.get('rows', '?')} | "
                f"**Cols:** {entry.get('cols', '?')}\n\n"
                f"#### Semantic Rules ({len(rules)})\n\n"
            )
            return header + _format_rules(rules)

        dataset_dd.change(fn=show_rules, inputs=[dataset_dd], outputs=[rules_display])

    return tab
