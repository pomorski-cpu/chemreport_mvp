from __future__ import annotations

from typing import Any, Dict, Iterable, Optional


def _safe_float(value: Any) -> Optional[float]:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except Exception:
        return None


def _mean(values: Iterable[float]) -> Optional[float]:
    seq = list(values)
    if not seq:
        return None
    return sum(seq) / len(seq)


def _model_confidence(predictions: list[dict[str, Any]]) -> float:
    scores: list[float] = []
    for pred in predictions:
        prob_toxic = _safe_float(pred.get("prob_toxic"))
        if prob_toxic is not None:
            scores.append(max(prob_toxic, 1.0 - prob_toxic))
            continue

        conf = _safe_float(pred.get("confidence_score"))
        if conf is not None:
            scores.append(max(0.0, min(1.0, conf)))

    return max(0.0, min(1.0, _mean(scores) if scores else 0.35))


def _ad_reliability(predictions: list[dict[str, Any]]) -> float:
    ad_scores = [
        max(0.0, min(1.0, score))
        for score in (_safe_float(pred.get("ad_score")) for pred in predictions)
        if score is not None
    ]
    if ad_scores:
        return _mean(ad_scores) or 0.0

    flags = [pred.get("in_domain") for pred in predictions if pred.get("in_domain") is not None]
    if not flags:
        return 0.25
    if all(flags):
        return 0.6
    if any(flags):
        return 0.3
    return 0.05


def estimate_reliability(
    *,
    predictions: list[dict[str, Any]],
    analogues: list[dict[str, Any]] | None = None,
    category: dict[str, Any] | None = None,
    warnings: list[str] | None = None,
) -> Dict[str, Any]:
    analogues = analogues or []
    category = category or {}
    warnings = warnings or []

    ad_score = round(_ad_reliability(predictions), 3)
    model_confidence = round(_model_confidence(predictions), 3)
    analogue_support = round(min(1.0, len(analogues) / 3.0), 3) if analogues else 0.0
    category_consistency = round(
        max(0.0, min(1.0, _safe_float(category.get("consistency_score")) or 0.0)),
        3,
    )

    final_score = (
        0.45 * ad_score
        + 0.30 * model_confidence
        + 0.15 * analogue_support
        + 0.10 * category_consistency
    )

    has_ood = any(pred.get("in_domain") is False for pred in predictions)
    if has_ood:
        final_score -= 0.15

    if analogues == []:
        final_score = min(final_score, 0.64)

    if any("fragment" in str(item).lower() or "фрагмент" in str(item).lower() for item in warnings):
        final_score -= 0.05

    final_score = round(max(0.0, min(1.0, final_score)), 3)

    if final_score >= 0.7 and analogue_support > 0.0 and not has_ood:
        final_label = "Высокая"
    elif final_score >= 0.4 and not has_ood:
        final_label = "Средняя"
    else:
        final_label = "Низкая"

    summary_parts: list[str] = [
        f"Надёжность оценки: {final_label.lower()}."
        f" AD-компонент={ad_score:.2f}, модельная уверенность={model_confidence:.2f}."
    ]
    if analogues:
        summary_parts.append(
            f"Аналоговая поддержка доступна ({len(analogues)} аналога(ов)); "
            f"согласованность категории={category_consistency:.2f}."
        )
    else:
        summary_parts.append(
            "Данные по аналогам недоступны, поэтому надёжность ограничена модельными и AD-показателями."
        )
    if has_ood:
        summary_parts.append("Есть выход за область применимости как минимум для одной модели.")

    return {
        "ad_score": ad_score,
        "analogue_support": analogue_support,
        "category_consistency": category_consistency,
        "model_confidence": model_confidence,
        "final_score": final_score,
        "final_label": final_label,
        "summary_ru": " ".join(summary_parts),
    }
