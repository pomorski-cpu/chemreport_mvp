# core/report.py
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional
import io
import os
import textwrap
import math

from core.utils import resource_path

# ----------------------------
# Payload (единый формат данных)
# ----------------------------

def build_report_payload(
    *,
    meta: Dict[str, Any],
    descriptors: Dict[str, Any],
    predictions: List[Dict[str, Any]],
    warnings: List[str],
    decision: Optional[Dict[str, Any]] = None,
    profile: Optional[Dict[str, Any]] = None,
    analogues: Optional[List[Dict[str, Any]]] = None,
    category: Optional[Dict[str, Any]] = None,
    read_across: Optional[Dict[str, Any]] = None,
    reliability: Optional[Dict[str, Any]] = None,
    svg: Optional[str] = None,
) -> Dict[str, Any]:
    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "meta": meta or {},
        "descriptors": descriptors or {},
        "predictions": predictions or [],
        "warnings": warnings or [],
        "decision": decision or {},
        "profile": profile or {},
        "analogues": analogues or [],
        "category": category or {},
        "read_across": read_across or {},
        "reliability": reliability or {},
        "svg": svg or "",
    }


# ----------------------------
# HTML (если хочешь оставить)
# ----------------------------


def _localize_task_name(task: str) -> str:
    mapping = {
        "LogP": "LogP",
        "Toxicity": "Токсичность",
        "Pesticide Class": "Класс пестицида",
        "Model": "Модель",
    }
    return mapping.get(str(task), str(task))


def _localize_decision_status(status: str) -> str:
    mapping = {
        "approve": "Одобрить",
        "review": "Проверить вручную",
        "reject": "Отклонить",
        "insufficient_data": "Недостаточно данных",
    }
    return mapping.get(str(status), str(status) if status is not None else "-")


def _localize_risk_level(level: str) -> str:
    mapping = {
        "low": "Низкий",
        "medium": "Средний",
        "high": "Высокий",
        "critical": "Критический",
    }
    return mapping.get(str(level), str(level) if level is not None else "-")

def render_report_html(payload: Dict[str, Any]) -> str:
    meta = payload.get("meta", {})
    preds = payload.get("predictions", [])
    warns = payload.get("warnings", [])
    decision = payload.get("decision", {}) or {}
    profile = payload.get("profile", {}) or {}
    read_across = payload.get("read_across", {}) or {}
    analogues = payload.get("analogues", []) or []
    category = payload.get("category", {}) or {}
    reliability = payload.get("reliability", {}) or {}
    tox_meta = (decision.get("meta", {}) or {}).get("toxicity", {}) or {}
    svg = payload.get("svg", "")

    def esc(x: Any) -> str:
        s = "" if x is None else str(x)
        return (
            s.replace("&", "&amp;")
             .replace("<", "&lt;")
             .replace(">", "&gt;")
        )

    def fmt_val(v: Any) -> str:
        if isinstance(v, (int, float)):
            return f"{float(v):.3f}"
        return esc(v)

    rows = []
    for p in preds:
        rows.append(
            f"<tr>"
            f"<td>{esc(_localize_task_name(p.get('task','')))}</td>"
            f"<td style='text-align:right'>{fmt_val(p.get('value',''))}</td>"
            f"<td>{esc(p.get('confidence',''))}</td>"
            f"<td>{esc(p.get('notes',''))}</td>"
            f"</tr>"
        )

    warn_html = "<br/>".join(esc(w) for w in warns) if warns else "Предупреждения отсутствуют."
    profile_html = "<br/>".join(esc(item) for item in profile.get("summary_ru", [])) if profile else "Профиль недоступен."
    analogue_summary = esc(category.get("summary_ru", "")) if category else "Сводка по аналогам недоступна."
    target_blocks = []
    target_tables = []
    for target_key, target_data in (read_across.get("targets", {}) or {}).items():
        prediction = target_data.get("prediction") or {}
        target_analogues = target_data.get("analogues", []) or []
        target_blocks.append(
            "<div style='margin:6px 0;'>"
            f"<b>{esc(target_data.get('label_ru', target_key))}</b>: "
            f"{esc(prediction.get('value', '-'))} "
            f"({esc(prediction.get('confidence', '-'))})"
            "</div>"
        )
        rows_html = []
        for analogue in target_analogues:
            rows_html.append(
                f"<tr>"
                f"<td>{esc(analogue.get('rank', ''))}</td>"
                f"<td>{esc(fmt_val(analogue.get('similarity', '')))}</td>"
                f"<td>{esc(fmt_val(analogue.get('value', analogue.get('logp', ''))))}</td>"
                f"<td>{esc(analogue.get('class_name', ''))}</td>"
                f"<td>{esc(analogue.get('smiles', ''))}</td>"
                f"</tr>"
            )
        target_tables.append(
            "<div style='margin-top:12px;'>"
            f"<h4 style='margin:0 0 8px 0;'>{esc(target_data.get('label_ru', target_key))}</h4>"
            "<table>"
            "<tr><th>#</th><th>Похожесть</th><th>Значение</th><th>Класс</th><th>SMILES</th></tr>"
            f"{''.join(rows_html) if rows_html else '<tr><td colspan=\"5\" class=\"muted\">Аналоги не найдены.</td></tr>'}"
            "</table>"
            "</div>"
        )
    reliability_html = esc(reliability.get("summary_ru", "")) if reliability else "Сводка по надёжности недоступна."
    tox_html = ""
    tox_prob = tox_meta.get("prob_toxic")
    tox_th = tox_meta.get("threshold")
    tox_decision = tox_meta.get("decision")
    if tox_prob is not None:
        tox_html = f"<br/>P(токсичности): {esc(fmt_val(tox_prob))}"
        if tox_th is not None:
            tox_html += f" (порог: {esc(fmt_val(tox_th))})"
        if tox_decision is not None:
            tox_html += f"; решение: {esc('токсично' if tox_decision else 'нетоксично')}"
    decision_html = (
        f"Статус: {esc(_localize_decision_status(decision.get('decision_status', '')))}<br/>"
        f"Уровень риска: {esc(_localize_risk_level(decision.get('risk_level', '')))}<br/>"
        f"Сводный балл: {esc(decision.get('score', ''))}<br/>"
        f"{tox_html}"
        f"Рекомендация: {esc(decision.get('recommendation', ''))}<br/>"
        f"Обоснование: {esc('; '.join(decision.get('rationale', [])) if decision else '-')}<br/>"
        f"Следующие действия: {esc('; '.join(decision.get('next_actions', [])) if decision else '-')}"
        if decision
        else "Сводка по DSS недоступна."
    )

    return f"""<!doctype html>
<html>
<head>
<meta charset="utf-8"/>
<title>Химический отчёт</title>
<style>
  body {{ font-family: system-ui, -apple-system, Segoe UI, Roboto, Arial, sans-serif; margin: 24px; color:#111; }}
  h1 {{ margin:0 0 6px 0; }}
  .muted {{ color:#555; font-size:12px; }}
  .grid {{ display:grid; grid-template-columns: 1.2fr 1fr; gap:16px; margin-top:16px; }}
  .card {{ border:1px solid #ddd; border-radius:10px; padding:14px; }}
  table {{ width:100%; border-collapse:collapse; font-size:13px; }}
  th, td {{ border-bottom:1px solid #eee; padding:8px; vertical-align:top; }}
  th {{ text-align:left; background:#fafafa; }}
  .right {{ text-align:right; }}
</style>
</head>
<body>
  <h1>Химический отчёт</h1>
  <div class="muted">Сформирован: {esc(payload.get("generated_at",""))}</div>

  <div class="grid">
    <div class="card">
      <h3 style="margin:0 0 10px 0;">Структура (2D)</h3>
      <div>{svg if svg else "<div class='muted'>SVG-структура не предоставлена.</div>"}</div>
    </div>

    <div class="card">
      <h3 style="margin:0 0 10px 0;">Идентификация</h3>
      <table>
        <tr><th>Ввод</th><td>{esc(meta.get("input",""))}</td></tr>
        <tr><th>SMILES</th><td>{esc(meta.get("smiles",""))}</td></tr>
        <tr><th>InChIKey</th><td>{esc(meta.get("inchikey",""))}</td></tr>
        <tr><th>Источник</th><td>{esc(meta.get("source",""))}</td></tr>
      </table>
    </div>
  </div>

  <div class="card" style="margin-top:16px;">
    <h3 style="margin:0 0 10px 0;">Прогнозы</h3>
    <table>
      <tr><th>Задача</th><th class="right">Значение</th><th>Уверенность</th><th>Примечания</th></tr>
      {''.join(rows) if rows else "<tr><td colspan='4' class='muted'>Прогнозы отсутствуют.</td></tr>"}
    </table>
  </div>

  <div class="card" style="margin-top:16px;">
    <h3 style="margin:0 0 10px 0;">Предупреждения</h3>
    <div style="font-size:13px;">{warn_html}</div>
  </div>

  <div class="card" style="margin-top:16px;">
    <h3 style="margin:0 0 10px 0;">Сводка DSS</h3>
    <div style="font-size:13px;">{decision_html}</div>
  </div>

  <div class="card" style="margin-top:16px;">
    <h3 style="margin:0 0 10px 0;">Структурный профиль</h3>
    <div style="font-size:13px;">{profile_html}</div>
  </div>

  <div class="card" style="margin-top:16px;">
    <h3 style="margin:0 0 10px 0;">Надёжность</h3>
    <div style="font-size:13px;">{reliability_html}</div>
  </div>

  <div class="card" style="margin-top:16px;">
    <h3 style="margin:0 0 10px 0;">Прогноз по аналогам</h3>
    <div style="font-size:13px; margin-bottom:10px;">{analogue_summary}</div>
    <div style="font-size:13px; margin-bottom:10px;">{''.join(target_blocks) if target_blocks else "Целевые результаты по аналогам недоступны."}</div>
    {''.join(target_tables) if target_tables else "<div class='muted'>Детальные таблицы аналогов недоступны.</div>"}
  </div>
</body>
</html>"""


# ----------------------------
# PDF (бизнес-отчёт)
# ----------------------------


def _resolve_pdf_fonts(pdfmetrics, TTFont) -> tuple[str, str]:
    regular_name = "Helvetica"
    bold_name = "Helvetica-Bold"
    registered = set(pdfmetrics.getRegisteredFontNames())
    if "ChemReportUnicode" in registered and "ChemReportUnicode-Bold" in registered:
        return "ChemReportUnicode", "ChemReportUnicode-Bold"

    candidate_pairs = [
        (
            resource_path("fonts/DejaVuSans.ttf"),
            resource_path("fonts/DejaVuSans-Bold.ttf"),
        ),
        (
            resource_path("assets/fonts/DejaVuSans.ttf"),
            resource_path("assets/fonts/DejaVuSans-Bold.ttf"),
        ),
        (
            r"C:\Windows\Fonts\arial.ttf",
            r"C:\Windows\Fonts\arialbd.ttf",
        ),
        (
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        ),
        (
            "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
        ),
        (
            "/usr/share/fonts/truetype/noto/NotoSans-Regular.ttf",
            "/usr/share/fonts/truetype/noto/NotoSans-Bold.ttf",
        ),
        (
            "/Library/Fonts/Arial.ttf",
            "/Library/Fonts/Arial Bold.ttf",
        ),
        (
            "/System/Library/Fonts/Supplemental/Arial.ttf",
            "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
        ),
        (
            "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
            "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
        ),
    ]

    for regular_path, bold_path in candidate_pairs:
        if not (os.path.exists(regular_path) and os.path.exists(bold_path)):
            continue
        try:
            if "ChemReportUnicode" not in registered:
                pdfmetrics.registerFont(TTFont("ChemReportUnicode", str(regular_path)))
            if "ChemReportUnicode-Bold" not in registered:
                pdfmetrics.registerFont(TTFont("ChemReportUnicode-Bold", str(bold_path)))
            return "ChemReportUnicode", "ChemReportUnicode-Bold"
        except Exception:
            continue

    return regular_name, bold_name


def render_report_pdf(payload: Dict[str, Any], out_path: str) -> None:
    """
    Minimal PDF format: only SMILES, Properties, Predictions.
    """
    from reportlab.lib.pagesizes import A4
    from reportlab.lib import colors
    from reportlab.lib.units import mm
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.utils import ImageReader
    import cairosvg

    meta = payload.get("meta", {}) or {}
    desc = payload.get("descriptors", {}) or {}
    preds = payload.get("predictions", []) or []
    warns = payload.get("warnings", []) or []
    decision = payload.get("decision", {}) or {}
    profile = payload.get("profile", {}) or {}
    read_across = payload.get("read_across", {}) or {}
    analogues = payload.get("analogues", []) or []
    category = payload.get("category", {}) or {}
    reliability = payload.get("reliability", {}) or {}
    tox_meta = (decision.get("meta", {}) or {}).get("toxicity", {}) or {}
    gen_at = payload.get("generated_at", "")
    svg = payload.get("svg", "")

    font_regular, font_bold = _resolve_pdf_fonts(pdfmetrics, TTFont)

    styles = getSampleStyleSheet()
    title = ParagraphStyle(
        "Title",
        parent=styles["Title"],
        fontName=font_bold,
        fontSize=20,
        leading=24,
        textColor=colors.HexColor("#111111"),
        spaceAfter=6,
    )
    muted = ParagraphStyle(
        "Muted",
        parent=styles["Normal"],
        fontName=font_regular,
        fontSize=9.5,
        leading=12,
        textColor=colors.HexColor("#606770"),
    )
    h = ParagraphStyle(
        "H",
        parent=styles["Heading3"],
        fontName=font_bold,
        fontSize=12.5,
        leading=15,
        textColor=colors.HexColor("#111111"),
        spaceAfter=8,
    )
    p_small = ParagraphStyle(
        "PSmall",
        parent=styles["Normal"],
        fontName=font_regular,
        fontSize=9.8,
        leading=13,
        wordWrap="CJK",
    )

    def P(text: Any):
        return Paragraph("" if text is None else str(text), p_small)

    def fmt_value(v: Any) -> str:
        if isinstance(v, (int, float)):
            return f"{float(v):.3f}"
        return "" if v is None else str(v)

    def wrap_text(s: Any, width: int = 80) -> str:
        text = "" if s is None else str(s)
        if not text:
            return ""
        return "<br/>".join(textwrap.wrap(text, width=width))

    def build_svg_flowable(svg_text: str, max_width: float):
        if not svg_text:
            return P("Структура недоступна.")
        try:
            png_bytes = cairosvg.svg2png(bytestring=svg_text.encode("utf-8"))
            image_buffer = io.BytesIO(png_bytes)
            img_reader = ImageReader(image_buffer)
            width_px, height_px = img_reader.getSize()
            if not width_px or not height_px:
                return P("Структура недоступна.")
            display_width = min(max_width, float(width_px))
            display_height = display_width * float(height_px) / float(width_px)
            image_buffer.seek(0)
            return Image(image_buffer, width=display_width, height=display_height)
        except Exception:
            return P("Не удалось встроить изображение структуры в PDF.")

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    doc = SimpleDocTemplate(
        out_path,
        pagesize=A4,
        leftMargin=18 * mm,
        rightMargin=18 * mm,
        topMargin=16 * mm,
        bottomMargin=16 * mm,
        title="Химический отчёт",
        author="ChemReport MVP",
    )
    W, _ = A4
    content_w = W - doc.leftMargin - doc.rightMargin
    card_bg = colors.HexColor("#F7F8FA")
    card_border = colors.HexColor("#D7DBE1")

    story: List[Any] = []
    story.append(Paragraph("Химический отчёт", title))
    story.append(Paragraph(f"Сформирован: {gen_at}", muted))
    story.append(Spacer(1, 10))

    story.append(Paragraph("Структура (2D)", h))
    structure_card = Table(
        [[build_svg_flowable(svg, content_w - 24)]],
        colWidths=[content_w],
        style=TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), card_bg),
            ("BOX", (0, 0), (-1, -1), 1, card_border),
            ("ALIGN", (0, 0), (-1, -1), "CENTER"),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("LEFTPADDING", (0, 0), (-1, -1), 12),
            ("RIGHTPADDING", (0, 0), (-1, -1), 12),
            ("TOPPADDING", (0, 0), (-1, -1), 12),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 12),
        ]),
    )
    story.append(structure_card)
    story.append(Spacer(1, 12))

    story.append(Paragraph("SMILES", h))
    smiles_card = Table(
        [[Paragraph(wrap_text(meta.get("smiles", ""), 110) or "-", p_small)]],
        colWidths=[content_w],
        style=TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), card_bg),
            ("BOX", (0, 0), (-1, -1), 1, card_border),
            ("LEFTPADDING", (0, 0), (-1, -1), 12),
            ("RIGHTPADDING", (0, 0), (-1, -1), 12),
            ("TOPPADDING", (0, 0), (-1, -1), 10),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 10),
        ]),
    )
    story.append(smiles_card)
    story.append(Spacer(1, 12))

    story.append(Paragraph("Свойства", h))
    prop_data = [[P("Свойство"), P("Значение")]]
    if desc:
        for k, v in desc.items():
            prop_data.append([P(str(k)), P(fmt_value(v))])
    else:
        prop_data.append([P(""), P("Свойства отсутствуют.")])

    prop_tbl = Table(prop_data, colWidths=[content_w * 0.45, content_w * 0.55], hAlign="LEFT")
    prop_tbl.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), font_bold),
        ("FONTSIZE", (0, 0), (-1, 0), 10),
        ("LINEBELOW", (0, 0), (-1, 0), 1, colors.HexColor("#E3E6EA")),
        ("FONTNAME", (0, 1), (-1, -1), font_regular),
        ("FONTSIZE", (0, 1), (-1, -1), 9.8),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 8),
        ("RIGHTPADDING", (0, 0), (-1, -1), 8),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.HexColor("#FFFFFF"), colors.HexColor("#FBFCFE")]),
        ("BOX", (0, 0), (-1, -1), 1, card_border),
    ]))
    prop_card = Table(
        [[prop_tbl]],
        colWidths=[content_w],
        style=TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), card_bg),
            ("BOX", (0, 0), (-1, -1), 1, card_border),
            ("LEFTPADDING", (0, 0), (-1, -1), 10),
            ("RIGHTPADDING", (0, 0), (-1, -1), 10),
            ("TOPPADDING", (0, 0), (-1, -1), 10),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 10),
        ]),
    )
    story.append(prop_card)
    story.append(Spacer(1, 12))

    story.append(Paragraph("Прогнозы", h))
    pred_data = [[P("Задача"), P("Значение"), P("Уверенность"), P("Примечания")]]
    if preds:
        for pr in preds:
            pred_data.append([
                P(_localize_task_name(pr.get("task", ""))),
                P(fmt_value(pr.get("value", ""))),
                P(wrap_text(pr.get("confidence", ""), 30)),
                P(wrap_text(pr.get("notes", ""), 45)),
            ])
    else:
        pred_data.append([P(""), P(""), P("Прогнозы отсутствуют."), P("")])

    pred_inner_w = content_w - 20
    pred_tbl = Table(
        pred_data,
        colWidths=[pred_inner_w * 0.18, pred_inner_w * 0.12, pred_inner_w * 0.22, pred_inner_w * 0.48],
        hAlign="LEFT",
    )
    pred_tbl.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), font_bold),
        ("FONTSIZE", (0, 0), (-1, 0), 10),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#111111")),
        ("LINEBELOW", (0, 0), (-1, 0), 1, colors.HexColor("#E3E6EA")),
        ("FONTNAME", (0, 1), (-1, -1), font_regular),
        ("FONTSIZE", (0, 1), (-1, -1), 9.8),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("TEXTCOLOR", (0, 1), (-1, -1), colors.HexColor("#111111")),
        ("ALIGN", (1, 1), (1, -1), "RIGHT"),
        ("LEFTPADDING", (0, 0), (-1, -1), 8),
        ("RIGHTPADDING", (0, 0), (-1, -1), 8),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.HexColor("#FFFFFF"), colors.HexColor("#FBFCFE")]),
        ("BOX", (0, 0), (-1, -1), 1, card_border),
    ]))
    pred_card = Table(
        [[pred_tbl]],
        colWidths=[content_w],
        style=TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), card_bg),
            ("BOX", (0, 0), (-1, -1), 1, card_border),
            ("LEFTPADDING", (0, 0), (-1, -1), 10),
            ("RIGHTPADDING", (0, 0), (-1, -1), 10),
            ("TOPPADDING", (0, 0), (-1, -1), 10),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 10),
        ]),
    )
    story.append(pred_card)
    story.append(Spacer(1, 12))

    story.append(Paragraph("Предупреждения", h))
    warnings_text = "<br/>".join(wrap_text(item, 90) for item in warns) if warns else "-"
    warnings_tbl = Table(
        [[P("Сводка"), P(warnings_text)]],
        colWidths=[content_w * 0.24, content_w * 0.76],
        hAlign="LEFT",
    )
    warnings_tbl.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), card_bg),
                ("BOX", (0, 0), (-1, -1), 1, card_border),
                ("FONTNAME", (0, 0), (0, -1), font_bold),
                ("FONTNAME", (1, 0), (1, -1), font_regular),
                ("FONTSIZE", (0, 0), (-1, -1), 9.8),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 8),
                ("RIGHTPADDING", (0, 0), (-1, -1), 8),
                ("TOPPADDING", (0, 0), (-1, -1), 6),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
                ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#DCE2EA")),
            ]
        )
    )
    story.append(warnings_tbl)
    story.append(Spacer(1, 12))

    story.append(Paragraph("Сводка DSS", h))
    tox_line = "-"
    tox_prob = tox_meta.get("prob_toxic")
    tox_th = tox_meta.get("threshold")
    tox_decision = tox_meta.get("decision")
    if tox_prob is not None:
        tox_line = f"P(токсичности)={fmt_value(tox_prob)}"
        if tox_th is not None:
            tox_line += f"; порог={fmt_value(tox_th)}"
        if tox_decision is not None:
            tox_line += f"; решение={'токсично' if tox_decision else 'нетоксично'}"
    decision_rows = [
        [P("Статус"), P(_localize_decision_status(decision.get("decision_status", "-")))],
        [P("Уровень риска"), P(_localize_risk_level(decision.get("risk_level", "-")))],
        [P("Сводный балл"), P(fmt_value(decision.get("score", "-")))],
        [P("Основание по токсичности"), P(tox_line)],
        [P("Рекомендация"), P(wrap_text(decision.get("recommendation", "-"), 90))],
        [
            P("Обоснование"),
            P(wrap_text("; ".join(decision.get("rationale", [])) if decision else "-", 90)),
        ],
        [
            P("Следующие действия"),
            P(wrap_text("; ".join(decision.get("next_actions", [])) if decision else "-", 90)),
        ],
    ]
    d_tbl = Table(decision_rows, colWidths=[content_w * 0.24, content_w * 0.76], hAlign="LEFT")
    d_tbl.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), card_bg),
                ("BOX", (0, 0), (-1, -1), 1, card_border),
                ("FONTNAME", (0, 0), (0, -1), font_bold),
                ("FONTNAME", (1, 0), (1, -1), font_regular),
                ("FONTSIZE", (0, 0), (-1, -1), 9.8),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 8),
                ("RIGHTPADDING", (0, 0), (-1, -1), 8),
                ("TOPPADDING", (0, 0), (-1, -1), 6),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
                ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#DCE2EA")),
            ]
        )
    )
    story.append(d_tbl)
    story.append(Spacer(1, 12))

    story.append(Paragraph("Структурный профиль", h))
    profile_lines = profile.get("summary_ru", []) if profile else []
    profile_text = "<br/>".join(wrap_text(line, 90) for line in profile_lines) if profile_lines else "-"
    profile_tbl = Table(
        [[P("Сводка"), P(profile_text)]],
        colWidths=[content_w * 0.24, content_w * 0.76],
        hAlign="LEFT",
    )
    profile_tbl.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), card_bg),
                ("BOX", (0, 0), (-1, -1), 1, card_border),
                ("FONTNAME", (0, 0), (0, -1), font_bold),
                ("FONTNAME", (1, 0), (1, -1), font_regular),
                ("FONTSIZE", (0, 0), (-1, -1), 9.8),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 8),
                ("RIGHTPADDING", (0, 0), (-1, -1), 8),
                ("TOPPADDING", (0, 0), (-1, -1), 6),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
                ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#DCE2EA")),
            ]
        )
    )
    story.append(profile_tbl)
    story.append(Spacer(1, 12))

    story.append(Paragraph("Надёжность", h))
    reliability_rows = [
        [P("Метка"), P(reliability.get("final_label", "-"))],
        [P("Сводный балл"), P(fmt_value(reliability.get("final_score", "-")))],
        [P("Сводка"), P(wrap_text(reliability.get("summary_ru", "-"), 90))],
    ]
    r_tbl = Table(reliability_rows, colWidths=[content_w * 0.24, content_w * 0.76], hAlign="LEFT")
    r_tbl.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), card_bg),
                ("BOX", (0, 0), (-1, -1), 1, card_border),
                ("FONTNAME", (0, 0), (0, -1), font_bold),
                ("FONTNAME", (1, 0), (1, -1), font_regular),
                ("FONTSIZE", (0, 0), (-1, -1), 9.8),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 8),
                ("RIGHTPADDING", (0, 0), (-1, -1), 8),
                ("TOPPADDING", (0, 0), (-1, -1), 6),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
                ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#DCE2EA")),
            ]
        )
    )
    story.append(r_tbl)
    story.append(Spacer(1, 12))

    story.append(Paragraph("Прогноз по аналогам", h))
    analogue_rows = [
        [P("Сводка"), P(wrap_text(category.get("summary_ru", "Аналоги не найдены."), 90))],
    ]
    for target_key, target_data in (read_across.get("targets", {}) or {}).items():
        prediction = target_data.get("prediction") or {}
        analogue_rows.append(
            [
                P(target_data.get("label_ru", target_key)),
                P(
                    wrap_text(
                        f"Значение: {prediction.get('value', '-')}; "
                        f"уверенность: {prediction.get('confidence', '-')}",
                        90,
                    )
                ),
            ]
        )
    ra_summary_tbl = Table(analogue_rows, colWidths=[content_w * 0.24, content_w * 0.76], hAlign="LEFT")
    ra_summary_tbl.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), card_bg),
                ("BOX", (0, 0), (-1, -1), 1, card_border),
                ("FONTNAME", (0, 0), (0, -1), font_bold),
                ("FONTNAME", (1, 0), (1, -1), font_regular),
                ("FONTSIZE", (0, 0), (-1, -1), 9.8),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 8),
                ("RIGHTPADDING", (0, 0), (-1, -1), 8),
                ("TOPPADDING", (0, 0), (-1, -1), 6),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
                ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#DCE2EA")),
            ]
        )
    )
    story.append(ra_summary_tbl)
    target_results = (read_across.get("targets", {}) or {})
    if target_results:
        for target_key, target_data in target_results.items():
            story.append(Spacer(1, 8))
            story.append(Paragraph(str(target_data.get("label_ru", target_key)), h))

            analogue_table_data = [[P("#"), P("Похожесть"), P("Значение"), P("Класс"), P("SMILES")]]
            target_analogues = target_data.get("analogues", []) or []
            if target_analogues:
                for analogue in target_analogues:
                    analogue_table_data.append(
                        [
                            P(analogue.get("rank", "")),
                            P(fmt_value(analogue.get("similarity", ""))),
                            P(fmt_value(analogue.get("value", analogue.get("logp", "")))),
                            P(analogue.get("class_name", "")),
                            P(wrap_text(analogue.get("smiles", ""), 55)),
                        ]
                    )
            else:
                analogue_table_data.append([P(""), P(""), P(""), P(""), P("Аналоги не найдены.")])

            analogue_tbl = Table(
                analogue_table_data,
                colWidths=[
                    content_w * 0.06,
                    content_w * 0.14,
                    content_w * 0.12,
                    content_w * 0.18,
                    content_w * 0.50,
                ],
                hAlign="LEFT",
            )
            analogue_tbl.setStyle(
                TableStyle(
                    [
                        ("BACKGROUND", (0, 0), (-1, 0), colors.white),
                        ("FONTNAME", (0, 0), (-1, 0), font_bold),
                        ("FONTSIZE", (0, 0), (-1, 0), 10),
                        ("LINEBELOW", (0, 0), (-1, 0), 1, colors.HexColor("#E3E6EA")),
                        ("FONTNAME", (0, 1), (-1, -1), font_regular),
                        ("FONTSIZE", (0, 1), (-1, -1), 9.3),
                        ("VALIGN", (0, 0), (-1, -1), "TOP"),
                        ("LEFTPADDING", (0, 0), (-1, -1), 8),
                        ("RIGHTPADDING", (0, 0), (-1, -1), 8),
                        ("TOPPADDING", (0, 0), (-1, -1), 6),
                        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
                        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.HexColor("#FFFFFF"), colors.HexColor("#FBFCFE")]),
                        ("BOX", (0, 0), (-1, -1), 1, card_border),
                    ]
                )
            )
            story.append(analogue_tbl)
    else:
        analogue_table_data = [[P("#"), P("Похожесть"), P("Значение"), P("Класс"), P("SMILES")]]
        if analogues:
            for analogue in analogues:
                analogue_table_data.append(
                    [
                        P(analogue.get("rank", "")),
                        P(fmt_value(analogue.get("similarity", ""))),
                        P(fmt_value(analogue.get("value", analogue.get("logp", "")))),
                        P(analogue.get("class_name", "")),
                        P(wrap_text(analogue.get("smiles", ""), 55)),
                    ]
                )
        else:
            analogue_table_data.append([P(""), P(""), P(""), P(""), P("Аналоги не найдены.")])

        analogue_tbl = Table(
            analogue_table_data,
            colWidths=[
                content_w * 0.06,
                content_w * 0.14,
                content_w * 0.12,
                content_w * 0.18,
                content_w * 0.50,
            ],
            hAlign="LEFT",
        )
        analogue_tbl.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.white),
                    ("FONTNAME", (0, 0), (-1, 0), font_bold),
                    ("FONTSIZE", (0, 0), (-1, 0), 10),
                    ("LINEBELOW", (0, 0), (-1, 0), 1, colors.HexColor("#E3E6EA")),
                    ("FONTNAME", (0, 1), (-1, -1), font_regular),
                    ("FONTSIZE", (0, 1), (-1, -1), 9.3),
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                    ("LEFTPADDING", (0, 0), (-1, -1), 8),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 8),
                    ("TOPPADDING", (0, 0), (-1, -1), 6),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
                    ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.HexColor("#FFFFFF"), colors.HexColor("#FBFCFE")]),
                    ("BOX", (0, 0), (-1, -1), 1, card_border),
                ]
            )
        )
        story.append(Spacer(1, 8))
        story.append(analogue_tbl)

    doc.build(story)


def render_batch_table_pdf(df, out_path: str, title: str = "Пакетный химический отчёт") -> None:
    """
    Export batch results DataFrame to a single multi-page PDF table.
    """
    from reportlab.lib.pagesizes import A4, landscape
    from reportlab.lib import colors
    from reportlab.lib.units import mm
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    doc = SimpleDocTemplate(
        out_path,
        pagesize=landscape(A4),
        leftMargin=12 * mm,
        rightMargin=12 * mm,
        topMargin=12 * mm,
        bottomMargin=12 * mm,
        title=title,
        author="ChemReport MVP",
    )

    font_regular, font_bold = _resolve_pdf_fonts(pdfmetrics, TTFont)

    styles = getSampleStyleSheet()
    h = ParagraphStyle(
        "BatchTitle",
        parent=styles["Title"],
        fontName=font_bold,
        fontSize=16,
        leading=19,
        textColor=colors.HexColor("#111111"),
    )
    sub = ParagraphStyle(
        "BatchSub",
        parent=styles["Normal"],
        fontName=font_regular,
        fontSize=9,
        leading=11,
        textColor=colors.HexColor("#5B6470"),
    )
    cell = ParagraphStyle(
        "BatchCell",
        parent=styles["Normal"],
        fontName=font_regular,
        fontSize=8.5,
        leading=10.5,
    )

    def esc(x: Any) -> str:
        s = "" if x is None else str(x)
        return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

    cols = [str(c) for c in df.columns]
    rows = []
    rows.append([Paragraph(f"<b>{esc(c)}</b>", cell) for c in cols])
    for _, r in df.iterrows():
        row = []
        for c in cols:
            v = r[c]
            if isinstance(v, float) and math.isfinite(v):
                txt = f"{v:.6g}"
            else:
                txt = "" if v is None else str(v)
            row.append(Paragraph(esc(txt), cell))
        rows.append(row)

    page_w, _ = landscape(A4)
    usable_w = page_w - doc.leftMargin - doc.rightMargin
    col_w = usable_w / max(1, len(cols))
    col_widths = [col_w] * len(cols)

    table = Table(rows, colWidths=col_widths, repeatRows=1)
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#EFF3F8")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#111111")),
        ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#D0D7DE")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]))

    story = [
        Paragraph(title, h),
        Paragraph(f"Сформирован: {datetime.now().isoformat(timespec='seconds')} | Строк: {len(df)}", sub),
        Spacer(1, 8),
        table,
    ]
    doc.build(story)


# Удобный алиас, если в app ты хочешь заменить экспорт HTML на PDF:
def export_report_pdf(payload: Dict[str, Any], out_path: str) -> None:
    render_report_pdf(payload, out_path)
