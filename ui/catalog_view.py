"""Dataset catalog tab — dataset list + rules detail panel."""
from __future__ import annotations

import gradio as gr

from ui.data_loader import load_catalog

_DOMAIN_COLORS = {
    "health":       "#0d9488",
    "finance":      "#2563eb",
    "environment":  "#059669",
    "biology":      "#7c3aed",
    "physics":      "#d97706",
    "demographics": "#db2777",
    "agriculture":  "#65a30d",
    "science":      "#0891b2",
    "materials":    "#ea580c",
    "transportation": "#0891b2",
}

_RULE_COLORS = {
    "range":        "#0d9488",
    "enum":         "#2563eb",
    "regex":        "#d97706",
    "not_null":     "#dc2626",
    "unique":       "#7c3aed",
    "dtype":        "#059669",
    "cross_column": "#db2777",
}


def _domain_color(domain: str) -> str:
    for key, color in _DOMAIN_COLORS.items():
        if key in domain.lower():
            return color
    return "#64748b"


# ── Dataset list ───────────────────────────────────────────────────────────────

def _build_dataset_list(catalog: dict, selected: str = "") -> str:
    if not catalog:
        return '<p style="color:#94a3b8">No datasets found.</p>'

    rows = []
    for ds_id, entry in sorted(catalog.items()):
        domain  = entry.get("domain", "unknown")
        r       = entry.get("rows", "?")
        c       = entry.get("cols", "?")
        n_rules = len(entry.get("rules", []))
        color   = _domain_color(domain)
        is_sel  = ds_id == selected

        bg           = "#f0fdfa" if is_sel else "#fff"
        left_border  = f"3px solid {color}" if is_sel else "3px solid transparent"
        name_weight  = "700" if is_sel else "500"
        rows_text    = f"{r:,} × {c}" if isinstance(r, int) else f"{r} × {c}"

        rows.append(
            f'<tr style="background:{bg};border-bottom:1px solid #f1f5f9;border-left:{left_border}">'
            f'<td style="padding:11px 14px;font-size:13px;color:#1e293b;font-weight:{name_weight}">{ds_id}</td>'
            f'<td style="padding:11px 8px">'
            f'<span style="background:{color}18;color:{color};font-size:10px;font-weight:700;'
            f'padding:2px 8px;border-radius:4px;text-transform:uppercase;letter-spacing:.05em;white-space:nowrap">'
            f'{domain}</span></td>'
            f'<td style="padding:11px 8px;font-size:12px;color:#64748b;white-space:nowrap">{rows_text}</td>'
            f'<td style="padding:11px 14px;text-align:right">'
            f'<span style="font-size:12px;font-weight:600;color:#475569">{n_rules}</span>'
            f'<span style="font-size:11px;color:#94a3b8"> rules</span></td>'
            f'</tr>'
        )

    header = (
        '<div style="border:1px solid #e2e8f0;border-radius:10px;overflow:hidden">'
        '<div style="max-height:500px;overflow-y:auto">'
        '<table style="width:100%;border-collapse:collapse">'
        '<thead><tr style="background:#f8fafc;border-bottom:2px solid #e2e8f0;position:sticky;top:0">'
        '<th style="padding:10px 14px;text-align:left;font-size:11px;font-weight:700;color:#475569;text-transform:uppercase;letter-spacing:.06em">Dataset</th>'
        '<th style="padding:10px 8px;text-align:left;font-size:11px;font-weight:700;color:#475569;text-transform:uppercase;letter-spacing:.06em">Domain</th>'
        '<th style="padding:10px 8px;text-align:left;font-size:11px;font-weight:700;color:#475569;text-transform:uppercase;letter-spacing:.06em">Size</th>'
        '<th style="padding:10px 14px;text-align:right;font-size:11px;font-weight:700;color:#475569;text-transform:uppercase;letter-spacing:.06em">Rules</th>'
        '</tr></thead>'
        '<tbody>' + "".join(rows) + '</tbody>'
        '</table></div></div>'
    )
    return header


# ── Rules panel ────────────────────────────────────────────────────────────────

def _build_rules_panel(catalog: dict, dataset_id: str | None) -> str:
    if not dataset_id or dataset_id not in catalog:
        return (
            '<div style="display:flex;align-items:center;justify-content:center;height:200px;'
            'color:#94a3b8;font-size:14px;border:2px dashed #e2e8f0;border-radius:10px">'
            'Select a dataset to view its semantic rules.'
            '</div>'
        )

    entry  = catalog[dataset_id]
    rules  = entry.get("rules", [])
    domain = entry.get("domain", "N/A")
    rows   = entry.get("rows", "?")
    cols   = entry.get("cols", "?")
    color  = _domain_color(domain)
    rows_text = f"{rows:,}" if isinstance(rows, int) else str(rows)

    header = (
        f'<div style="margin-bottom:16px;padding-bottom:16px;border-bottom:1px solid #e2e8f0">'
        f'<div style="display:flex;align-items:center;gap:10px;margin-bottom:6px">'
        f'<h3 style="margin:0;color:#1e293b;font-size:18px;font-weight:700">{dataset_id}</h3>'
        f'<span style="background:{color}18;color:{color};font-size:11px;font-weight:700;'
        f'padding:2px 10px;border-radius:4px;text-transform:uppercase;letter-spacing:.05em">{domain}</span>'
        f'</div>'
        f'<span style="font-size:13px;color:#64748b">'
        f'{rows_text} rows &times; {cols} cols'
        f'&nbsp;&nbsp;·&nbsp;&nbsp;{len(rules)} semantic rule{"s" if len(rules) != 1 else ""}'
        f'</span>'
        f'</div>'
    )

    if not rules:
        rules_html = '<p style="color:#94a3b8;font-size:13px">No semantic rules defined.</p>'
    else:
        rule_lines = []
        for r in rules:
            rtype  = r.get("type", "unknown")
            col    = r.get("column", r.get("columns", ""))
            rcolor = _RULE_COLORS.get(rtype, "#94a3b8")

            detail = ""
            if rtype == "range":
                detail = f"{r.get('min','?')} → {r.get('max','?')}"
            elif rtype == "enum":
                vals    = r.get("values", r.get("allowed_values", []))
                preview = ", ".join(str(v) for v in vals[:5])
                if len(vals) > 5:
                    preview += f"  +{len(vals)-5} more"
                detail = preview
            elif rtype == "regex":
                detail = r.get("pattern", "")
            elif rtype == "dtype":
                detail = r.get("expected_dtype", "?")
            elif rtype == "cross_column":
                detail = r.get("relationship", "?")

            detail_span = (
                f'<span style="font-size:11px;color:#94a3b8;margin-left:10px;font-family:monospace">{detail}</span>'
                if detail else ""
            )
            rule_lines.append(
                f'<div style="display:flex;align-items:center;padding:9px 12px;margin-bottom:4px;'
                f'background:#f8fafc;border-radius:6px;border-left:3px solid {rcolor}">'
                f'<span style="background:{rcolor}18;color:{rcolor};font-size:10px;font-weight:700;'
                f'padding:2px 8px;border-radius:3px;text-transform:uppercase;letter-spacing:.05em;'
                f'min-width:72px;text-align:center;flex-shrink:0">{rtype}</span>'
                f'<code style="color:#1e293b;font-size:12px;font-weight:600;margin-left:10px">{col}</code>'
                f'{detail_span}'
                f'</div>'
            )
        rules_html = "\n".join(rule_lines)

    return header + rules_html


# ── Tab builder ────────────────────────────────────────────────────────────────

def create_catalog_tab() -> gr.Blocks:
    catalog = load_catalog()

    with gr.Blocks() as tab:
        gr.Markdown("## Dataset Catalog")

        if not catalog:
            gr.Markdown("**No catalog found at `datasets/catalog.json`.**")
            return tab

        default_ds = sorted(catalog.keys())[0]
        domains    = {entry.get("domain", "") for entry in catalog.values()}

        gr.HTML(
            f'<p style="color:#64748b;font-size:13px;margin:0 0 16px 0">'
            f'{len(catalog)} datasets &nbsp;·&nbsp; {len(domains)} domains &nbsp;·&nbsp;'
            f' Select a dataset to inspect its semantic validation rules.</p>'
        )

        with gr.Row():
            with gr.Column(scale=3):
                dataset_list = gr.HTML(value=_build_dataset_list(catalog, default_ds))
                dataset_dd   = gr.Dropdown(
                    choices=sorted(catalog.keys()),
                    value=default_ds,
                    label="Selected dataset",
                )

            with gr.Column(scale=2):
                rules_panel = gr.HTML(value=_build_rules_panel(catalog, default_ds))

        def on_select(ds):
            return _build_dataset_list(catalog, ds), _build_rules_panel(catalog, ds)

        dataset_dd.change(fn=on_select, inputs=[dataset_dd], outputs=[dataset_list, rules_panel])

    return tab
