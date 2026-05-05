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
        "Bioactivity EC50 Invertebrates": "Биоактивность: EC50, водные беспозвоночные",
        "Bioactivity LC50 Fish": "Биоактивность: LC50, рыбы",
        "Bioactivity LD50 Mammals Oral": "Биоактивность: LD50, млекопитающие перорально",
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


def _decision_dss_html(decision: Dict[str, Any], esc, fmt_val) -> str:
    if not decision:
        return "Сводка по DSS недоступна."

    def evidence_lines(kind: str) -> str:
        items = [
            item for item in (decision.get("evidence", []) or [])
            if item.get("category") == kind and float(item.get("score_delta") or 0) > 0
        ]
        if not items:
            return "-"
        items = sorted(items, key=lambda item: float(item.get("score_delta") or 0), reverse=True)
        return "<br/>".join(
            f"- {esc(item.get('label', item.get('source', 'Фактор')))} "
            f"({esc(item.get('source', '-'))}; вклад={esc(fmt_val(item.get('score_delta', '')))}). "
            f"{esc(item.get('rationale', ''))}"
            for item in items[:4]
        )

    conflicts = decision.get("conflicts", []) or []
    flags = decision.get("data_quality_flags", []) or []
    conflicts_html = "<br/>".join(f"- {esc(item.get('message', item.get('code', 'конфликт')))}" for item in conflicts) or "-"
    flags_html = "<br/>".join(f"- {esc(item.get('message', item.get('code', 'флаг качества')))}" for item in flags[:4]) or "-"
    return (
        f"Статус: {esc(_localize_decision_status(decision.get('decision_status', '')))}<br/>"
        f"Уровень риска: {esc(_localize_risk_level(decision.get('risk_level', '')))}<br/>"
        f"Сводный балл: {esc(fmt_val(decision.get('score', '')))}<br/>"
        f"Опасность: {esc(fmt_val(decision.get('hazard_score', '')))}<br/>"
        f"Неопределённость: {esc(fmt_val(decision.get('uncertainty_score', '')))}<br/>"
        f"<b>Ключевые факторы риска:</b><br/>{evidence_lines('hazard')}<br/>"
        f"<b>Факторы неопределённости:</b><br/>{evidence_lines('uncertainty')}<br/>"
        f"<b>Конфликты:</b><br/>{conflicts_html}<br/>"
        f"<b>Флаги качества:</b><br/>{flags_html}<br/>"
        f"Рекомендация: {esc(decision.get('recommendation', ''))}<br/>"
        f"Обоснование: {esc('; '.join(decision.get('rationale', [])) if decision else '-')}<br/>"
        f"Следующие действия: {esc('; '.join(decision.get('next_actions', [])) if decision else '-')}"
    )


def _decision_dss_plain(decision: Dict[str, Any], fmt_value) -> Dict[str, str]:
    evidence = decision.get("evidence", []) or []
    hazard = [
        item for item in evidence
        if item.get("category") == "hazard" and float(item.get("score_delta") or 0) > 0
    ]
    uncertainty = [
        item for item in evidence
        if item.get("category") == "uncertainty" and float(item.get("score_delta") or 0) > 0
    ]
    hazard = sorted(hazard, key=lambda item: float(item.get("score_delta") or 0), reverse=True)
    uncertainty = sorted(uncertainty, key=lambda item: float(item.get("score_delta") or 0), reverse=True)

    def join_items(items: List[Dict[str, Any]]) -> str:
        return "; ".join(
            f"{item.get('label', item.get('source', 'Фактор'))} ({item.get('source', '-')}; вклад={fmt_value(item.get('score_delta', ''))})"
            for item in items[:4]
        ) or "-"

    return {
        "hazard": join_items(hazard),
        "uncertainty": join_items(uncertainty),
        "conflicts": "; ".join(item.get("message", item.get("code", "конфликт")) for item in (decision.get("conflicts", []) or [])) or "-",
        "flags": "; ".join(item.get("message", item.get("code", "флаг качества")) for item in (decision.get("data_quality_flags", []) or [])[:4]) or "-",
    }

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
    decision_html = _decision_dss_html(decision, esc, fmt_val)

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

  <div class="grid top-grid">
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
            import cairosvg

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
    dss_plain = _decision_dss_plain(decision, fmt_value)
    decision_rows = [
        [P("Статус"), P(_localize_decision_status(decision.get("decision_status", "-")))],
        [P("Уровень риска"), P(_localize_risk_level(decision.get("risk_level", "-")))],
        [P("Сводный балл"), P(fmt_value(decision.get("score", "-")))],
        [P("Балл опасности"), P(fmt_value(decision.get("hazard_score", "-")))],
        [P("Балл неопределённости"), P(fmt_value(decision.get("uncertainty_score", "-")))],
        [P("Ключевые факторы риска"), P(wrap_text(dss_plain["hazard"], 90))],
        [P("Факторы неопределённости"), P(wrap_text(dss_plain["uncertainty"], 90))],
        [P("Конфликты"), P(wrap_text(dss_plain["conflicts"], 90))],
        [P("Флаги качества"), P(wrap_text(dss_plain["flags"], 90))],
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
        Paragraph(escape(_maybe_fix_mojibake(title)), h),
        Paragraph(f"Сформирован: {datetime.now().isoformat(timespec='seconds')} | Строк: {len(df)}", sub),
        Spacer(1, 8),
        table,
    ]
    doc.build(story)


# Удобный алиас, если в app ты хочешь заменить экспорт HTML на PDF:
def export_report_pdf(payload: Dict[str, Any], out_path: str) -> None:
    render_report_pdf(payload, out_path)


# ---------------------------------------------------------------------------
# R&D report redesign v2
# ---------------------------------------------------------------------------
# The functions below intentionally override the earlier report renderers.
# They keep the public API unchanged while producing a compact R&D-oriented
# report with Russian labels, pastel risk highlighting, and an appendix for
# verbose model details.

def _report_text(value: Any, default: str = "-") -> str:
    if value is None:
        return default
    text = str(value)
    return text if text else default


def _maybe_fix_mojibake(text: Any) -> str:
    value = _report_text(text, "")
    if not value:
        return value
    try:
        if any(ch in value for ch in ("Р", "С", "Ѓ", "љ", "њ")):
            fixed = value.encode("latin1", errors="ignore").decode("utf-8", errors="ignore")
            if fixed and sum("а" <= ch.lower() <= "я" or ch == "ё" for ch in fixed) > sum(
                "а" <= ch.lower() <= "я" or ch == "ё" for ch in value
            ):
                return fixed
    except Exception:
        pass
    return value


def _localize_task_name(task: str) -> str:
    mapping = {
        "LogP": "LogP",
        "Toxicity": "Токсичность",
        "Pesticide Class": "Класс пестицида",
        "Bioactivity EC50 Invertebrates": "EC50, водные беспозвоночные",
        "Bioactivity LC50 Fish": "LC50, рыбы",
        "Bioactivity LD50 Mammals Oral": "LD50, млекопитающие, перорально",
        "Биоактивность: EC50, водные беспозвоночные": "EC50, водные беспозвоночные",
        "Биоактивность: LC50, рыбы": "LC50, рыбы",
        "Биоактивность: LD50, млекопитающие перорально": "LD50, млекопитающие, перорально",
        "Биоактивность: EC50 регрессия": "EC50, регрессия",
        "Биоактивность: LD50 регрессия": "LD50, регрессия",
        "Model": "Модель",
    }
    return mapping.get(str(task), _maybe_fix_mojibake(task))


def _localize_decision_status(status: str) -> str:
    value = str(status or "").strip().lower()
    mapping = {
        "approve": "Одобрить",
        "review": "Проверить вручную",
        "reject": "Отклонить",
        "insufficient_data": "Недостаточно данных",
    }
    if value in mapping:
        return mapping[value]
    return _maybe_fix_mojibake(status) or "-"


def _localize_risk_level(level: str) -> str:
    value = str(level or "").strip().lower()
    mapping = {
        "low": "Низкий",
        "medium": "Средний",
        "high": "Высокий",
        "critical": "Критический",
    }
    if value in mapping:
        return mapping[value]
    return _maybe_fix_mojibake(level) or "-"


def _fmt_report_value(value: Any, ndigits: int = 3) -> str:
    if isinstance(value, bool):
        return "да" if value else "нет"
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return f"{float(value):.{ndigits}f}"
    return _maybe_fix_mojibake(value) or "-"


def _score_to_percent(value: Any) -> str:
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return f"{float(value) * 100:.0f}%"
    return _fmt_report_value(value)


def _status_palette(status: Any) -> tuple[str, str]:
    raw = str(status or "").strip().lower()
    label = _localize_decision_status(raw)
    if raw == "approve" or "одобр" in label.lower():
        return "#DDEFD8", "#2F5C3B"
    if raw == "review" or "провер" in label.lower():
        return "#F7E7BE", "#755B16"
    if raw == "reject" or "отклон" in label.lower():
        return "#F3D2D2", "#7A3030"
    if raw == "insufficient_data" or "недостат" in label.lower():
        return "#DCE7F3", "#2C4D6B"
    return "#ECEFF3", "#3E4852"


def _risk_palette(risk: Any) -> tuple[str, str]:
    raw = str(risk or "").strip().lower()
    label = _localize_risk_level(raw)
    if raw == "low" or "низ" in label.lower():
        return "#DDEFD8", "#2F5C3B"
    if raw == "medium" or "сред" in label.lower():
        return "#F7E7BE", "#755B16"
    if raw in {"high", "critical"} or "выс" in label.lower() or "крит" in label.lower():
        return "#F3D2D2", "#7A3030"
    return "#ECEFF3", "#3E4852"


def _score_palette(value: Any) -> tuple[str, str]:
    try:
        score = float(value)
    except Exception:
        return "#ECEFF3", "#3E4852"
    if score >= 0.70:
        return "#F3D2D2", "#7A3030"
    if score >= 0.45:
        return "#F7E7BE", "#755B16"
    return "#DDEFD8", "#2F5C3B"


def _decision_evidence_groups(decision: Dict[str, Any]) -> Dict[str, List[Dict[str, Any]]]:
    evidence = decision.get("evidence", []) or []
    groups = {"hazard": [], "uncertainty": []}
    for item in evidence:
        category = item.get("category")
        try:
            delta = float(item.get("score_delta") or 0)
        except Exception:
            delta = 0.0
        if category in groups and delta > 0:
            groups[category].append(item)
    for key in groups:
        groups[key] = sorted(groups[key], key=lambda item: float(item.get("score_delta") or 0), reverse=True)
    return groups


def _evidence_line(item: Dict[str, Any], *, include_delta: bool = True) -> str:
    label = _maybe_fix_mojibake(item.get("label") or item.get("source") or "Фактор")
    source = _maybe_fix_mojibake(item.get("source") or "")
    rationale = _maybe_fix_mojibake(item.get("rationale") or item.get("message") or "")
    delta = _fmt_report_value(item.get("score_delta"))
    parts = [label]
    if source and source != label:
        parts.append(f"источник: {source}")
    if include_delta:
        parts.append(f"вклад: {delta}")
    if rationale:
        parts.append(rationale)
    return "; ".join(parts)


def _short_prediction_note(prediction: Dict[str, Any]) -> str:
    parts = []
    if prediction.get("prob_toxic") is not None:
        parts.append(f"P(toxic)={_fmt_report_value(prediction.get('prob_toxic'))}")
    elif prediction.get("probability") is not None:
        parts.append(f"P={_fmt_report_value(prediction.get('probability'))}")
    if prediction.get("confidence_score") is not None:
        parts.append(f"уверенность={_fmt_report_value(prediction.get('confidence_score'))}")
    if prediction.get("in_domain") is not None:
        parts.append(f"AD={'да' if prediction.get('in_domain') else 'нет'}")
    if prediction.get("ad_score") is not None:
        parts.append(f"AD score={_fmt_report_value(prediction.get('ad_score'))}")
    return "; ".join(parts) or _maybe_fix_mojibake(prediction.get("confidence") or "-")


def _compact_report_text(value: Any, max_chars: int = 180) -> str:
    text = " ".join(_maybe_fix_mojibake(value).split())
    if len(text) <= max_chars:
        return text
    return text[: max(0, max_chars - 1)].rstrip() + "…"


def _prediction_ad_label(prediction: Dict[str, Any]) -> str:
    status = prediction.get("ad_status_ru") or prediction.get("ad_status")
    if status:
        return _compact_report_text(status, 70)
    if prediction.get("in_domain") is True:
        return "в AD"
    if prediction.get("in_domain") is False:
        return "вне AD"
    return "-"


def _prediction_comment(prediction: Dict[str, Any]) -> str:
    note = _maybe_fix_mojibake(prediction.get("notes") or "")
    if note:
        return _compact_report_text(note, 170)
    return _compact_report_text(_short_prediction_note(prediction), 170)


def _ad_reason_short(item: Dict[str, Any]) -> str:
    reason = item.get("reason") or item.get("method") or "-"
    return _compact_report_text(reason, 150)


def _status_sort_key(value: Any) -> int:
    text = _maybe_fix_mojibake(value).strip().lower()
    if "approve" in text or "одобр" in text:
        return 0
    if "review" in text or "провер" in text:
        return 1
    if "reject" in text or "отклон" in text:
        return 2
    if "insufficient" in text or "недостат" in text:
        return 3
    return 4


def _find_df_column(df, candidates: List[str]) -> Optional[str]:
    lowered = {_maybe_fix_mojibake(col).strip().lower(): col for col in df.columns}
    raw_lowered = {str(col).strip().lower(): col for col in df.columns}
    for candidate in candidates:
        cand = candidate.strip().lower()
        if cand in lowered:
            return lowered[cand]
        if cand in raw_lowered:
            return raw_lowered[cand]
    for col in df.columns:
        normalized = _maybe_fix_mojibake(col).strip().lower()
        raw = str(col).strip().lower()
        for candidate in candidates:
            cand = candidate.strip().lower()
            if cand in normalized or cand in raw:
                return col
    return None



def render_report_html(payload: Dict[str, Any]) -> str:
    from html import escape

    meta = payload.get("meta", {}) or {}
    desc = payload.get("descriptors", {}) or {}
    preds = payload.get("predictions", []) or []
    warns = payload.get("warnings", []) or []
    decision = payload.get("decision", {}) or {}
    profile = payload.get("profile", {}) or {}
    read_across = payload.get("read_across", {}) or {}
    category = payload.get("category", {}) or {}
    reliability = payload.get("reliability", {}) or {}
    applicability_domain = payload.get("applicability_domain", {}) or {}
    svg = payload.get("svg", "") or ""

    def esc(value: Any) -> str:
        return escape(_maybe_fix_mojibake(value))

    status = decision.get("decision_status", "-")
    risk = decision.get("risk_level", "-")
    status_bg, status_fg = _status_palette(status)
    risk_bg, risk_fg = _risk_palette(risk)
    hazard_bg, hazard_fg = _score_palette(decision.get("hazard_score"))
    uncertainty_bg, uncertainty_fg = _score_palette(decision.get("uncertainty_score"))
    groups = _decision_evidence_groups(decision)

    def evidence_html(items: List[Dict[str, Any]], empty: str) -> str:
        if not items:
            return f"<div class='muted'>{esc(empty)}</div>"
        return "<ul>" + "".join(f"<li>{esc(_compact_report_text(_evidence_line(item), 220))}</li>" for item in items[:5]) + "</ul>"

    pred_rows = []
    appendix_rows = []
    for pred in preds:
        pred_rows.append(
            "<tr>"
            f"<td>{esc(_localize_task_name(pred.get('task', 'Model')))}</td>"
            f"<td><b>{esc(_fmt_report_value(pred.get('value')))}</b></td>"
            f"<td>{esc(_maybe_fix_mojibake(pred.get('confidence') or '-'))}</td>"
            f"<td>{esc(_prediction_ad_label(pred))}</td>"
            f"<td>{esc(_prediction_comment(pred))}</td>"
            "</tr>"
        )
        appendix_rows.append(
            "<tr>"
            f"<td>{esc(_localize_task_name(pred.get('task', 'Model')))}</td>"
            f"<td>{esc(_fmt_report_value(pred.get('value')))}</td>"
            f"<td>{esc(_maybe_fix_mojibake(pred.get('confidence') or '-'))}</td>"
            f"<td>{esc(_maybe_fix_mojibake(pred.get('notes') or '-'))}</td>"
            "</tr>"
        )

    warning_html = (
        "<ul>" + "".join(f"<li>{esc(w)}</li>" for w in warns[:8]) + "</ul>"
        if warns else "<div class='muted'>Предупреждения отсутствуют.</div>"
    )
    profile_lines = profile.get("summary_ru", []) if profile else []
    profile_html = (
        "<ul>" + "".join(f"<li>{esc(line)}</li>" for line in profile_lines[:6]) + "</ul>"
        if profile_lines else "<div class='muted'>Структурный профиль недоступен.</div>"
    )

    ad_rows = []
    for item in applicability_domain.get("items", []) or []:
        ad_rows.append(
            "<tr>"
            f"<td>{esc(item.get('task', '-'))}</td>"
            f"<td><b>{esc(item.get('status_ru', item.get('status', '-')))}</b></td>"
            f"<td>{esc(_fmt_report_value(item.get('ad_score')))}</td>"
            f"<td>{esc(_ad_reason_short(item))}</td>"
            "</tr>"
        )
    ad_html = (
        f"<p><b>Сводно:</b> {esc(applicability_domain.get('summary_ru', 'Оценка AD недоступна.'))}</p>"
        "<table class='ad-table'><thead><tr><th>Модель</th><th>AD-статус</th><th>AD score</th><th>Причина</th></tr></thead>"
        f"<tbody>{''.join(ad_rows) or '<tr><td colspan=\"4\" class=\"muted\">AD-детализация недоступна.</td></tr>'}</tbody></table>"
    )

    conflicts = decision.get("conflicts", []) or []
    flags = decision.get("data_quality_flags", []) or []
    conflicts_html = (
        "<ul>" + "".join(f"<li>{esc(item.get('message') or item.get('code'))}</li>" for item in conflicts[:5]) + "</ul>"
        if conflicts else "<div class='muted'>Конфликты между источниками не выявлены.</div>"
    )
    flags_html = (
        "<ul>" + "".join(f"<li>{esc(item.get('message') or item.get('code'))}</li>" for item in flags[:5]) + "</ul>"
        if flags else "<div class='muted'>Критичных флагов качества нет.</div>"
    )

    ra_blocks = []
    for target_key, target_data in (read_across.get("targets", {}) or {}).items():
        prediction = target_data.get("prediction") or {}
        analogues = target_data.get("analogues", []) or []
        analogue_rows = "".join(
            "<tr>"
            f"<td>{esc(a.get('rank', ''))}</td>"
            f"<td>{esc(_fmt_report_value(a.get('similarity')))}</td>"
            f"<td>{esc(_fmt_report_value(a.get('value', a.get('logp'))))}</td>"
            f"<td>{esc(a.get('class_name', ''))}</td>"
            f"<td class='mono'>{esc(a.get('smiles', ''))}</td>"
            "</tr>"
            for a in analogues[:8]
        )
        ra_blocks.append(
            f"<h3>{esc(target_data.get('label_ru', target_key))}</h3>"
            f"<p><b>Прогноз:</b> {esc(prediction.get('value', '-'))}; "
            f"<b>уверенность:</b> {esc(prediction.get('confidence', '-'))}</p>"
            "<table><thead><tr><th>#</th><th>Похожесть</th><th>Значение</th><th>Класс</th><th>SMILES</th></tr></thead>"
            f"<tbody>{analogue_rows or '<tr><td colspan=\"5\" class=\"muted\">Аналоги не найдены.</td></tr>'}</tbody></table>"
        )

    return f"""<!doctype html>
<html>
<head>
<meta charset="utf-8"/>
<title>R&amp;D QSAR-отчёт</title>
<style>
  body {{ font-family: 'Segoe UI', Arial, sans-serif; margin: 28px; color: #20242A; }}
  h1 {{ margin: 0 0 4px 0; font-size: 28px; }}
  h2 {{ margin: 24px 0 10px 0; font-size: 18px; }}
  h3 {{ margin: 16px 0 8px 0; font-size: 14px; }}
  .muted {{ color: #68717C; font-size: 12px; }}
  .grid {{ display: grid; grid-template-columns: 1.05fr 0.95fr; gap: 14px; }}
  .top-grid {{ grid-template-columns: minmax(0, 0.7fr) minmax(0, 1.3fr); align-items: start; }}
  .card {{ border: 1px solid #DDE3EA; border-radius: 8px; padding: 14px; background: #FFFFFF; }}
  .structure-card svg {{ max-height: 145px; width: 100%; }}
  .structure-card h2 {{ margin-bottom: 6px; }}
  .soft {{ background: #F7F9FB; }}
  .card {{ break-inside: avoid; page-break-inside: avoid; }}
  .badges {{ display: flex; gap: 8px; flex-wrap: wrap; margin: 12px 0; }}
  .badge {{ border-radius: 6px; padding: 7px 10px; font-weight: 600; font-size: 13px; }}
  .badge small {{ display: block; font-weight: 400; opacity: 0.82; }}
  table {{ width: 100%; border-collapse: collapse; font-size: 12.5px; }}
  th, td {{ border-bottom: 1px solid #E6EBF0; padding: 7px 8px; vertical-align: top; overflow-wrap: anywhere; }}
  th {{ text-align: left; background: #F2F5F8; }}
  tr {{ break-inside: avoid; page-break-inside: avoid; }}
  .pred-table {{ table-layout: fixed; }}
  .pred-table th:nth-child(1) {{ width: 18%; }}
  .pred-table th:nth-child(2) {{ width: 25%; }}
  .pred-table th:nth-child(3) {{ width: 17%; }}
  .pred-table th:nth-child(4) {{ width: 13%; }}
  .pred-table th:nth-child(5) {{ width: 27%; }}
  .ad-table {{ table-layout: fixed; }}
  .ad-table th:nth-child(1) {{ width: 24%; }}
  .ad-table th:nth-child(2) {{ width: 18%; }}
  .ad-table th:nth-child(3) {{ width: 12%; }}
  .ad-table th:nth-child(4) {{ width: 46%; }}
  ul {{ margin: 6px 0 0 18px; padding: 0; }}
  li {{ margin-bottom: 4px; }}
  .mono {{ font-family: Consolas, monospace; font-size: 11px; overflow-wrap: anywhere; }}
  .section-gap {{ height: 12px; }}
  @media print {{
    body {{ margin: 18mm; }}
    h2 {{ break-after: avoid; page-break-after: avoid; }}
    table {{ page-break-inside: auto; }}
  }}
</style>
</head>
<body>
  <h1>R&amp;D QSAR-отчёт</h1>
  <div class="muted">Сформирован: {esc(payload.get('generated_at', ''))}</div>

  <div class="badges">
    <div class="badge" style="background:{status_bg};color:{status_fg};"><small>Статус DSS</small>{esc(_localize_decision_status(status))}</div>
    <div class="badge" style="background:{risk_bg};color:{risk_fg};"><small>Риск</small>{esc(_localize_risk_level(risk))}</div>
    <div class="badge" style="background:{hazard_bg};color:{hazard_fg};"><small>Опасность</small>{esc(_fmt_report_value(decision.get('hazard_score')))}</div>
    <div class="badge" style="background:{uncertainty_bg};color:{uncertainty_fg};"><small>Неопределённость</small>{esc(_fmt_report_value(decision.get('uncertainty_score')))}</div>
  </div>

  <div class="grid top-grid">
    <div class="card structure-card">
      <h2>Молекула</h2>
      <p><b>SMILES:</b> <span class="mono">{esc(meta.get('smiles') or meta.get('input') or '-')}</span></p>
      <div>{svg if svg else '<div class="muted">Структура не передана в отчёт.</div>'}</div>
    </div>
    <div class="card soft">
      <h2>Краткое решение</h2>
      <p><b>Рекомендация:</b> {esc(decision.get('recommendation', '-'))}</p>
      <p><b>Надёжность:</b> {esc(reliability.get('final_label', '-'))}; балл {esc(_fmt_report_value(reliability.get('final_score')))}</p>
      <p><b>Следующие действия:</b> {esc('; '.join(decision.get('next_actions', []) or []) or '-')}</p>
    </div>
  </div>

  <h2>Модельные прогнозы</h2>
  <table class="pred-table"><thead><tr><th>Модель</th><th>Результат</th><th>Уверенность</th><th>AD</th><th>Комментарий</th></tr></thead><tbody>{''.join(pred_rows) or '<tr><td colspan="5" class="muted">Прогнозы отсутствуют.</td></tr>'}</tbody></table>

  <div class="section-gap"></div>

  <h2>Ключевые факторы риска</h2>
  <div class="card">{evidence_html(groups['hazard'], 'Существенные факторы опасности не выделены.')}</div>

  <h2>Факторы неопределённости</h2>
  <div class="card">{evidence_html(groups['uncertainty'], 'Выраженные источники неопределённости не выделены.')}</div>

  <h2>Физико-химические дескрипторы</h2>
  <table><tbody>
    <tr><th>MolWt</th><td>{esc(_fmt_report_value(desc.get('MolWt')))}</td><th>LogP</th><td>{esc(_fmt_report_value(desc.get('MolLogP') or desc.get('LogP')))}</td></tr>
    <tr><th>TPSA</th><td>{esc(_fmt_report_value(desc.get('TPSA')))}</td><th>HBD / HBA</th><td>{esc(_fmt_report_value(desc.get('NumHDonors')))} / {esc(_fmt_report_value(desc.get('NumHAcceptors')))}</td></tr>
  </tbody></table>

  <h2>Область применимости</h2>
  <div class="card">{ad_html}</div>

  <h2>Проверки качества</h2>
  <div class="grid">
    <div class="card"><h3>Конфликты</h3>{conflicts_html}</div>
    <div class="card"><h3>Флаги качества</h3>{flags_html}</div>
  </div>

  <h2>Структурный профиль</h2>
  <div class="card">{profile_html}</div>

  <h2>Приложение: подробности моделей</h2>
  <table><thead><tr><th>Задача</th><th>Значение</th><th>Уверенность</th><th>Комментарии</th></tr></thead><tbody>{''.join(appendix_rows) or '<tr><td colspan="4" class="muted">Нет подробных комментариев.</td></tr>'}</tbody></table>

  <h2>Приложение: read-across</h2>
  <div class="card"><p>{esc(category.get('summary_ru', 'Сводка по аналогам недоступна.'))}</p>{''.join(ra_blocks) or '<div class="muted">Детализация read-across недоступна.</div>'}</div>

  <h2>Предупреждения</h2>
  <div class="card">{warning_html}</div>
</body>
</html>"""


def render_report_pdf(payload: Dict[str, Any], out_path: str) -> None:
    from html import escape
    from reportlab.lib.pagesizes import A4
    from reportlab.lib import colors
    from reportlab.lib.units import mm
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.utils import ImageReader

    meta = payload.get("meta", {}) or {}
    desc = payload.get("descriptors", {}) or {}
    preds = payload.get("predictions", []) or []
    warns = payload.get("warnings", []) or []
    decision = payload.get("decision", {}) or {}
    profile = payload.get("profile", {}) or {}
    read_across = payload.get("read_across", {}) or {}
    category = payload.get("category", {}) or {}
    reliability = payload.get("reliability", {}) or {}
    applicability_domain = payload.get("applicability_domain", {}) or {}
    svg = payload.get("svg", "") or ""

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    doc = SimpleDocTemplate(
        out_path,
        pagesize=A4,
        leftMargin=15 * mm,
        rightMargin=15 * mm,
        topMargin=14 * mm,
        bottomMargin=14 * mm,
        title="R&D QSAR-отчёт",
        author="ChemReport MVP",
    )
    font_regular, font_bold = _resolve_pdf_fonts(pdfmetrics, TTFont)
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle("ReportTitleV2", parent=styles["Title"], fontName=font_bold, fontSize=20, leading=24, textColor=colors.HexColor("#20242A"))
    h = ParagraphStyle("ReportHeadingV2", parent=styles["Heading3"], fontName=font_bold, fontSize=12.5, leading=15, textColor=colors.HexColor("#20242A"), spaceBefore=8, spaceAfter=6)
    cell = ParagraphStyle("ReportCellV2", parent=styles["Normal"], fontName=font_regular, fontSize=9.2, leading=11.3, wordWrap="CJK")
    cell_bold = ParagraphStyle("ReportCellBoldV2", parent=cell, fontName=font_bold)
    muted = ParagraphStyle("ReportMutedV2", parent=cell, textColor=colors.HexColor("#68717C"))
    small = ParagraphStyle("ReportSmallV2", parent=cell, fontSize=8.2, leading=9.8, textColor=colors.HexColor("#68717C"))

    def esc(value: Any) -> str:
        return escape(_maybe_fix_mojibake(value))

    def P(value: Any, style=cell):
        text = esc(value)
        text = text.replace("\n", "<br/>")
        return Paragraph(text or "-", style)

    def rawP(value: Any, style=cell):
        return Paragraph(str(value) if value else "-", style)

    def style_table(tbl: Table, *, header: bool = False, boxed: bool = True) -> Table:
        commands = [
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("LEFTPADDING", (0, 0), (-1, -1), 6),
            ("RIGHTPADDING", (0, 0), (-1, -1), 6),
            ("TOPPADDING", (0, 0), (-1, -1), 5),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ("ROWBACKGROUNDS", (0, 1 if header else 0), (-1, -1), [colors.white, colors.HexColor("#FAFBFC")]),
            ("INNERGRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#DDE3EA")),
        ]
        if header:
            commands += [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#EEF3F7")),
                ("FONTNAME", (0, 0), (-1, 0), font_bold),
            ]
        if boxed:
            commands.append(("BOX", (0, 0), (-1, -1), 0.8, colors.HexColor("#D7DEE6")))
        tbl.setStyle(TableStyle(commands))
        return tbl

    def badge(label: str, value: str, bg: str, fg: str):
        table = Table([[P(label, small)], [P(value, cell_bold)]], colWidths=[35 * mm])
        table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor(bg)),
            ("TEXTCOLOR", (0, 0), (-1, -1), colors.HexColor(fg)),
            ("BOX", (0, 0), (-1, -1), 0.4, colors.HexColor(bg)),
            ("LEFTPADDING", (0, 0), (-1, -1), 7),
            ("RIGHTPADDING", (0, 0), (-1, -1), 7),
            ("TOPPADDING", (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ]))
        return table

    def structure_flowable(max_width: float):
        def png_flowable(png_bytes: bytes):
            img = Image(ImageReader(io.BytesIO(png_bytes)))
            img.drawWidth = max_width
            img.drawHeight = max_width * 0.58
            return img

        if svg:
            try:
                import cairosvg
                return png_flowable(cairosvg.svg2png(bytestring=svg.encode("utf-8"), output_width=520))
            except Exception:
                pass

        smiles = meta.get("smiles") or meta.get("input")
        if smiles:
            try:
                from rdkit import Chem as _Chem
                from rdkit.Chem import Draw as _Draw

                mol = _Chem.MolFromSmiles(str(smiles))
                if mol is not None:
                    pil_img = _Draw.MolToImage(mol, size=(620, 360))
                    buf = io.BytesIO()
                    pil_img.save(buf, format="PNG")
                    return png_flowable(buf.getvalue())
            except Exception:
                pass

        return P("Структура недоступна для PDF-рендеринга.", muted)

    story = [
        Paragraph("R&amp;D QSAR-отчёт", title_style),
        P(f"Сформирован: {payload.get('generated_at', '')}", muted),
        Spacer(1, 7),
    ]

    status = decision.get("decision_status", "-")
    risk = decision.get("risk_level", "-")
    badge_tbl = Table(
        [[
            badge("DSS", _localize_decision_status(status), *_status_palette(status)),
            badge("Риск", _localize_risk_level(risk), *_risk_palette(risk)),
            badge("Опасность", _fmt_report_value(decision.get("hazard_score")), *_score_palette(decision.get("hazard_score"))),
            badge("Неопределённость", _fmt_report_value(decision.get("uncertainty_score")), *_score_palette(decision.get("uncertainty_score"))),
        ]],
        colWidths=[37 * mm, 37 * mm, 37 * mm, 47 * mm],
    )
    badge_tbl.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP"), ("LEFTPADDING", (0, 0), (-1, -1), 0), ("RIGHTPADDING", (0, 0), (-1, -1), 5)]))
    story.append(badge_tbl)
    story.append(Spacer(1, 10))

    molecule_block = Table(
        [[
            rawP(f"<b>Молекула</b><br/>SMILES: {esc(meta.get('smiles') or meta.get('input') or '-')}<br/><br/>"
                 f"MolWt: {esc(_fmt_report_value(desc.get('MolWt')))}<br/>"
                 f"LogP: {esc(_fmt_report_value(desc.get('MolLogP') or desc.get('LogP')))}<br/>"
                 f"TPSA: {esc(_fmt_report_value(desc.get('TPSA')))}<br/>"
                 f"HBD/HBA: {esc(_fmt_report_value(desc.get('NumHDonors')))} / {esc(_fmt_report_value(desc.get('NumHAcceptors')))}"),
            structure_flowable(content_w * 0.22),
        ]],
        colWidths=[content_w * 0.72, content_w * 0.28],
    )
    style_table(molecule_block, boxed=True)
    story.append(molecule_block)
    story.append(Spacer(1, 8))

    summary_rows = [
        [P("Рекомендация", cell_bold), P(decision.get("recommendation", "-"))],
        [P("Надёжность", cell_bold), P(f"{reliability.get('final_label', '-')}; балл {_fmt_report_value(reliability.get('final_score'))}")],
        [P("Следующие действия", cell_bold), P("; ".join(decision.get("next_actions", []) or []) or "-")],
    ]
    story.append(Paragraph("R&amp;D summary", h))
    story.append(style_table(Table(summary_rows, colWidths=[content_w * 0.18, content_w * 0.82]), boxed=True))

    pred_rows = [[P("Модель", cell_bold), P("Результат", cell_bold), P("Уверенность", cell_bold), P("AD / комментарий", cell_bold)]]
    for pred in preds:
        pred_rows.append([
            P(_localize_task_name(pred.get("task", "Model"))),
            P(_fmt_report_value(pred.get("value")), cell_bold),
            P(_maybe_fix_mojibake(pred.get("confidence") or "-")),
            P(f"{_prediction_ad_label(pred)}; {_prediction_comment(pred)}"),
        ])
    if len(pred_rows) == 1:
        pred_rows.append([P("-"), P("-"), P("-"), P("Прогнозы отсутствуют.", muted)])
    story.append(Paragraph("Модельные прогнозы", h))
    story.append(style_table(Table(pred_rows, colWidths=[39 * mm, 43 * mm, 35 * mm, 53 * mm]), header=True))
    story.append(Spacer(1, 6))

    groups = _decision_evidence_groups(decision)
    reason_rows = [[P("Тип", cell_bold), P("Фактор", cell_bold)]]
    for item in groups["hazard"][:5]:
        reason_rows.append([P("Риск"), P(_compact_report_text(_evidence_line(item), 230))])
    if len(reason_rows) == 1:
        reason_rows.append([P("Риск"), P("Существенные факторы опасности не выделены.", muted)])
    for item in groups["uncertainty"][:5]:
        reason_rows.append([P("Неопределённость"), P(_compact_report_text(_evidence_line(item), 230))])
    if not groups["uncertainty"]:
        reason_rows.append([P("Неопределённость"), P("Выраженные источники неопределённости не выделены.", muted)])
    story.append(Paragraph("Объяснение DSS", h))
    story.append(style_table(Table(reason_rows, colWidths=[34 * mm, 136 * mm]), header=True))

    ad_rows = [[P("Модель", cell_bold), P("AD-статус", cell_bold), P("AD score", cell_bold), P("Причина", cell_bold)]]
    for item in applicability_domain.get("items", []) or []:
        ad_rows.append([
            P(item.get("task", "-")),
            P(item.get("status_ru", item.get("status", "-")), cell_bold),
            P(_fmt_report_value(item.get("ad_score"))),
            P(_ad_reason_short(item)),
        ])
    if len(ad_rows) == 1:
        ad_rows.append([P("-"), P("-"), P("-"), P("AD-детализация недоступна.")])
    story.append(Paragraph("Область применимости", h))
    story.append(P(applicability_domain.get("summary_ru", "Оценка AD недоступна."), muted))
    story.append(style_table(Table(ad_rows, colWidths=[43 * mm, 32 * mm, 24 * mm, 71 * mm]), header=True))

    conflicts = decision.get("conflicts", []) or []
    flags = decision.get("data_quality_flags", []) or []
    q_rows = [[P("Конфликты", cell_bold), P("; ".join(_maybe_fix_mojibake(i.get("message") or i.get("code")) for i in conflicts[:5]) or "Не выявлены.")],
              [P("Флаги качества", cell_bold), P("; ".join(_maybe_fix_mojibake(i.get("message") or i.get("code")) for i in flags[:5]) or "Критичных флагов нет.")]]
    if warns:
        q_rows.append([P("Предупреждения", cell_bold), P("; ".join(_maybe_fix_mojibake(w) for w in warns[:6]))])
    story.append(Paragraph("Качество данных", h))
    story.append(style_table(Table(q_rows, colWidths=[39 * mm, 131 * mm]), boxed=True))

    profile_lines = profile.get("summary_ru", []) if profile else []
    story.append(Paragraph("Структурный профиль", h))
    story.append(style_table(Table([[P("Сводка", cell_bold), P("; ".join(_maybe_fix_mojibake(x) for x in profile_lines[:6]) or "-")]], colWidths=[39 * mm, 131 * mm]), boxed=True))

    story.append(PageBreak())
    story.append(Paragraph("Приложение: подробности моделей", h))
    appendix_rows = [[P("Задача", cell_bold), P("Значение", cell_bold), P("Комментарии", cell_bold)]]
    for pred in preds:
        appendix_rows.append([
            P(_localize_task_name(pred.get("task", "Model"))),
            P(_fmt_report_value(pred.get("value"))),
            P(_prediction_comment(pred)),
        ])
    if len(appendix_rows) == 1:
        appendix_rows.append([P("-"), P("-"), P("Нет подробных комментариев.")])
    story.append(style_table(Table(appendix_rows, colWidths=[45 * mm, 45 * mm, 80 * mm]), header=True))

    story.append(Paragraph("Приложение: read-across", h))
    story.append(P(category.get("summary_ru", "Сводка по аналогам недоступна.")))
    for target_key, target_data in (read_across.get("targets", {}) or {}).items():
        prediction = target_data.get("prediction") or {}
        story.append(Spacer(1, 6))
        story.append(P(f"{target_data.get('label_ru', target_key)}: прогноз {prediction.get('value', '-')}; уверенность {prediction.get('confidence', '-')}"))
        analogue_rows = [[P("#", cell_bold), P("Похожесть", cell_bold), P("Значение", cell_bold), P("Класс", cell_bold), P("SMILES", cell_bold)]]
        for analogue in (target_data.get("analogues", []) or [])[:8]:
            analogue_rows.append([
                P(analogue.get("rank", "")),
                P(_fmt_report_value(analogue.get("similarity"))),
                P(_fmt_report_value(analogue.get("value", analogue.get("logp")))),
                P(analogue.get("class_name", "")),
                P(analogue.get("smiles", "")),
            ])
        if len(analogue_rows) > 1:
            story.append(style_table(Table(analogue_rows, colWidths=[10 * mm, 23 * mm, 25 * mm, 34 * mm, 78 * mm]), header=True))

    doc.build(story)


def render_batch_table_pdf(df, out_path: str, title: str = "R&D приоритизационный отчёт") -> None:
    from html import escape
    from reportlab.lib.pagesizes import A4, landscape
    from reportlab.lib import colors
    from reportlab.lib.units import mm
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    status_col = _find_df_column(df, ["Статус решения", "decision_status", "DSS", "DSS status"])
    risk_col = _find_df_column(df, ["Уровень риска", "risk_level", "Риск"])
    hazard_col = _find_df_column(df, ["Балл опасности DSS", "hazard_score", "Опасность"])
    uncertainty_col = _find_df_column(df, ["Балл неопределённости DSS", "uncertainty_score", "Неопределённость"])
    reason_col = _find_df_column(df, ["Ключевые причины DSS", "Главная причина", "key reasons", "reason"])
    ad_status_col = _find_df_column(df, ["AD-статус", "AD status", "applicability domain"])
    ad_score_col = _find_df_column(df, ["AD-score", "AD score", "mean_ad_score"])
    reliability_col = _find_df_column(df, ["Метка надёжности", "reliability", "Надёжность"])
    smiles_col = _find_df_column(df, ["SMILES", "smiles", "input", "Молекула"])

    work = df.copy()
    if status_col:
        work["_dss_sort"] = work[status_col].map(_status_sort_key)
        if uncertainty_col:
            work["_unc_sort"] = work[uncertainty_col].map(lambda x: float(x) if isinstance(x, (int, float)) else 999.0)
        else:
            work["_unc_sort"] = 999.0
        if hazard_col:
            work["_haz_sort"] = work[hazard_col].map(lambda x: float(x) if isinstance(x, (int, float)) else 999.0)
        else:
            work["_haz_sort"] = 999.0
        work = work.sort_values(["_dss_sort", "_unc_sort", "_haz_sort"], kind="stable").drop(columns=["_dss_sort", "_unc_sort", "_haz_sort"], errors="ignore")

    selected = []
    for col in [smiles_col, status_col, risk_col, hazard_col, uncertainty_col, ad_status_col, ad_score_col, reason_col, reliability_col]:
        if col and col not in selected:
            selected.append(col)
    if not selected:
        selected = list(work.columns[: min(8, len(work.columns))])

    doc = SimpleDocTemplate(
        out_path,
        pagesize=landscape(A4),
        leftMargin=11 * mm,
        rightMargin=11 * mm,
        topMargin=10 * mm,
        bottomMargin=10 * mm,
        title=title,
        author="ChemReport MVP",
    )
    font_regular, font_bold = _resolve_pdf_fonts(pdfmetrics, TTFont)
    styles = getSampleStyleSheet()
    h = ParagraphStyle("BatchTitleV2", parent=styles["Title"], fontName=font_bold, fontSize=17, leading=20, textColor=colors.HexColor("#20242A"))
    sub = ParagraphStyle("BatchSubV2", parent=styles["Normal"], fontName=font_regular, fontSize=9, leading=11, textColor=colors.HexColor("#68717C"))
    cell = ParagraphStyle("BatchCellV2", parent=styles["Normal"], fontName=font_regular, fontSize=8.1, leading=9.7, wordWrap="CJK")
    bold = ParagraphStyle("BatchCellBoldV2", parent=cell, fontName=font_bold)

    def P(value: Any, style=cell):
        text = escape(_maybe_fix_mojibake(value))
        return Paragraph(text or "-", style)

    counts = {}
    if status_col:
        for value in work[status_col].tolist():
            label = _localize_decision_status(_maybe_fix_mojibake(value))
            counts[label] = counts.get(label, 0) + 1

    count_rows = [[P("Статус", bold), P("Количество", bold)]]
    for label in ["Одобрить", "Проверить вручную", "Отклонить", "Недостаточно данных"]:
        count_rows.append([P(label), P(counts.get(label, 0), bold)])
    summary = Table(count_rows, colWidths=[48 * mm, 24 * mm])
    summary.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#EEF3F7")),
        ("INNERGRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#DDE3EA")),
        ("BOX", (0, 0), (-1, -1), 0.8, colors.HexColor("#D7DEE6")),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))

    max_rows = min(len(work), 120)
    table_rows = [[P("Молекула", bold), P("DSS", bold), P("Риск", bold), P("Опасность", bold), P("Неопределённость", bold), P("AD", bold), P("AD score", bold), P("Главная причина", bold), P("Надёжность", bold)]]
    for _, row in work.head(max_rows).iterrows():
        table_rows.append([
            P(row[smiles_col] if smiles_col else ""),
            P(row[status_col] if status_col else ""),
            P(row[risk_col] if risk_col else ""),
            P(_fmt_report_value(row[hazard_col]) if hazard_col else ""),
            P(_fmt_report_value(row[uncertainty_col]) if uncertainty_col else ""),
            P(row[ad_status_col] if ad_status_col else ""),
            P(_fmt_report_value(row[ad_score_col]) if ad_score_col else ""),
            P(row[reason_col] if reason_col else ""),
            P(row[reliability_col] if reliability_col else ""),
        ])

    page_w, _ = landscape(A4)
    usable_w = page_w - doc.leftMargin - doc.rightMargin
    col_widths = [usable_w * x for x in [0.17, 0.10, 0.08, 0.08, 0.10, 0.10, 0.08, 0.21, 0.08]]
    table = Table(table_rows, colWidths=col_widths, repeatRows=1)
    table_style = [
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#EEF3F7")),
        ("FONTNAME", (0, 0), (-1, 0), font_bold),
        ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#D0D7DE")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#FAFBFC")]),
    ]
    for i, (_, row) in enumerate(work.head(max_rows).iterrows(), start=1):
        status_value = row[status_col] if status_col else ""
        bg, fg = _status_palette(status_value)
        table_style.append(("BACKGROUND", (1, i), (1, i), colors.HexColor(bg)))
        table_style.append(("TEXTCOLOR", (1, i), (1, i), colors.HexColor(fg)))
        if risk_col:
            rbg, rfg = _risk_palette(row[risk_col])
            table_style.append(("BACKGROUND", (2, i), (2, i), colors.HexColor(rbg)))
            table_style.append(("TEXTCOLOR", (2, i), (2, i), colors.HexColor(rfg)))
        if hazard_col:
            hbg, hfg = _score_palette(row[hazard_col])
            table_style.append(("BACKGROUND", (3, i), (3, i), colors.HexColor(hbg)))
            table_style.append(("TEXTCOLOR", (3, i), (3, i), colors.HexColor(hfg)))
        if uncertainty_col:
            ubg, ufg = _score_palette(row[uncertainty_col])
            table_style.append(("BACKGROUND", (4, i), (4, i), colors.HexColor(ubg)))
            table_style.append(("TEXTCOLOR", (4, i), (4, i), colors.HexColor(ufg)))
        if ad_score_col:
            sbg, sfg = _score_palette(row[ad_score_col])
            table_style.append(("BACKGROUND", (6, i), (6, i), colors.HexColor(sbg)))
            table_style.append(("TEXTCOLOR", (6, i), (6, i), colors.HexColor(sfg)))
    table.setStyle(TableStyle(table_style))

    note = "" if len(work) <= max_rows else f"В PDF показаны первые {max_rows} строк после сортировки. Полная таблица остаётся в CSV/XLSX."
    story = [
        Paragraph(escape(_maybe_fix_mojibake(title)), h),
        Paragraph(f"Сформирован: {datetime.now().isoformat(timespec='seconds')} | Молекул: {len(work)}", sub),
        Paragraph("Сортировка: сначала кандидаты Approve, затем Review, Reject и Insufficient data; внутри Approve выше стоят молекулы с меньшей неопределённостью.", sub),
        Spacer(1, 7),
        summary,
        Spacer(1, 9),
    ]
    if note:
        story.extend([Paragraph(note, sub), Spacer(1, 5)])
    story.append(table)
    doc.build(story)


def export_report_pdf(payload: Dict[str, Any], out_path: str) -> None:
    render_report_pdf(payload, out_path)
