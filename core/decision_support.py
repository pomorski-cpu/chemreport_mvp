from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from core.utils import resource_path


@dataclass(frozen=True)
class DecisionRules:
    version: str
    toxicity_labels: List[str]
    toxicity_prob_medium: float
    toxicity_prob_high: float
    logp_medium: float
    logp_high: float
    tpsa_low: float
    ood_penalty: float
    mixed_domain_penalty: float
    approve_max: float
    review_max: float
    reject_max: float


def _safe_float(x: Any) -> Optional[float]:
    if x is None or x == "":
        return None
    try:
        return float(x)
    except Exception:
        return None


class DecisionSupport:
    def __init__(self, rules_path: str = "config/decision_rules.json"):
        path: Path = resource_path(rules_path)
        with open(path, "r", encoding="utf-8") as f:
            cfg = json.load(f)

        tox = cfg.get("toxicity", {})
        logp = cfg.get("logp", {})
        tpsa = cfg.get("tpsa", {})
        dom = cfg.get("domain", {})
        th = cfg.get("decision_thresholds", {})

        self.rules = DecisionRules(
            version=str(cfg.get("version", "v1")),
            toxicity_labels=[str(x).lower() for x in tox.get("toxic_labels", ["генотоксичный", "toxic", "1"])],
            toxicity_prob_medium=float(tox.get("prob_medium", 0.55)),
            toxicity_prob_high=float(tox.get("prob_high", 0.70)),
            logp_medium=float(logp.get("medium", 3.0)),
            logp_high=float(logp.get("high", 4.5)),
            tpsa_low=float(tpsa.get("low", 20.0)),
            ood_penalty=float(dom.get("ood_penalty", 0.35)),
            mixed_domain_penalty=float(dom.get("mixed_penalty", 0.15)),
            approve_max=float(th.get("approve_max", 0.30)),
            review_max=float(th.get("review_max", 0.60)),
            reject_max=float(th.get("reject_max", 0.80)),
        )

    def evaluate(
        self,
        *,
        meta: Optional[Dict[str, Any]] = None,
        descriptors: Optional[Dict[str, Any]] = None,
        predictions: Optional[List[Dict[str, Any]]] = None,
        warnings: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        meta = meta or {}
        descriptors = descriptors or {}
        predictions = predictions or []
        warnings = warnings or []

        rationale: List[str] = []
        next_actions: List[str] = []
        score = 0.0

        tox_pred = self._find_prediction(predictions, "tox")
        logp_pred = self._find_prediction(predictions, "logp")

        toxic_score = self._toxicity_score(tox_pred, rationale)
        score += toxic_score

        phys_score = self._physchem_score(descriptors, logp_pred, rationale)
        score += phys_score

        dom_score, has_ood = self._domain_score(predictions, rationale)
        score += dom_score

        if any("fragment" in str(w).lower() for w in warnings):
            score += 0.05
            rationale.append("Обнаружено несколько фрагментов или солевая форма; надёжность прогноза может быть снижена.")

        score = max(0.0, min(1.0, score))

        if has_ood:
            decision_status = "insufficient_data"
            risk_level = "high" if score >= self.rules.review_max else "medium"
            recommendation = "Требуется ручная экспертная проверка из-за выхода прогноза за область применимости."
            next_actions.extend(
                [
                    "Проведите дополнительную валидацию на профильных референсных соединениях.",
                    "Не принимайте автоматическое решение по этой молекуле без экспертной оценки.",
                ]
            )
        elif score <= self.rules.approve_max:
            decision_status = "approve"
            risk_level = "low"
            recommendation = "Возможен предварительный автоматический допуск к следующему этапу скрининга."
            next_actions.append("Переведите соединение на следующий этап с рутинным мониторингом.")
        elif score <= self.rules.review_max:
            decision_status = "review"
            risk_level = "medium"
            recommendation = "Перед продолжением требуется ручная проверка."
            next_actions.append("Проведите целевую экспертную проверку токсичности и класса пестицида.")
        else:
            decision_status = "reject"
            risk_level = "high" if score <= self.rules.reject_max else "critical"
            recommendation = "Автоматическое одобрение не допускается: выявлен профиль высокого риска."
            next_actions.extend(
                [
                    "Передайте соединение на токсикологическую экспертизу.",
                    "Рассмотрите снижение приоритета кандидата до получения дополнительных данных.",
                ]
            )

        if not rationale:
            rationale.append("Текущий набор правил не выявил выраженных сигналов риска.")

        toxicity_summary = self._toxicity_summary(tox_pred)
        return {
            "rule_version": self.rules.version,
            "decision_status": decision_status,
            "risk_level": risk_level,
            "score": round(score, 3),
            "recommendation": recommendation,
            "rationale": rationale,
            "next_actions": next_actions,
            "meta": {
                "source": "rule_based_dss",
                "input": meta.get("input", ""),
                "toxicity": toxicity_summary,
            },
        }

    def _find_prediction(self, predictions: List[Dict[str, Any]], needle: str) -> Optional[Dict[str, Any]]:
        n = needle.lower()
        for p in predictions:
            task = str(p.get("task", "")).lower()
            if n in task:
                return p
        return None

    def _toxicity_score(self, tox_pred: Optional[Dict[str, Any]], rationale: List[str]) -> float:
        if not tox_pred:
            rationale.append("Прогноз токсичности недоступен.")
            return 0.10

        prob = _safe_float(tox_pred.get("prob_toxic"))
        if prob is None:
            prob = _safe_float(tox_pred.get("confidence_score"))

        if prob is not None and prob >= self.rules.toxicity_prob_high:
            rationale.append(f"Высокая вероятность токсичности (P(токсичности)={prob:.2f}).")
            return 0.65
        if prob is not None and prob >= self.rules.toxicity_prob_medium:
            rationale.append(f"Умеренная вероятность токсичности (P(токсичности)={prob:.2f}).")
            return 0.40

        value = str(tox_pred.get("value", "")).strip().lower()
        is_toxic = value in self.rules.toxicity_labels or value.endswith("генотоксичный")
        if is_toxic:
            rationale.append("Метка токсичности положительна, но вероятностный запас недостаточен.")
            return 0.30

        rationale.append("Сигнал токсичности низкий или отрицательный.")
        return 0.05

    def _toxicity_summary(self, tox_pred: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        if not tox_pred:
            return {}
        return {
            "label": tox_pred.get("value"),
            "prob_toxic": _safe_float(tox_pred.get("prob_toxic")),
            "threshold": _safe_float(tox_pred.get("toxicity_threshold")),
            "decision": tox_pred.get("toxicity_decision"),
            "medium_cutoff": self.rules.toxicity_prob_medium,
            "high_cutoff": self.rules.toxicity_prob_high,
        }

    def _physchem_score(
        self, descriptors: Dict[str, Any], logp_pred: Optional[Dict[str, Any]], rationale: List[str]
    ) -> float:
        score = 0.0
        logp_val = _safe_float(descriptors.get("cLogP"))
        if logp_val is None and logp_pred:
            logp_val = _safe_float(logp_pred.get("value"))
        if logp_val is not None:
            if logp_val >= self.rules.logp_high:
                score += 0.20
                rationale.append(f"Высокая липофильность (LogP={logp_val:.2f}).")
            elif logp_val >= self.rules.logp_medium:
                score += 0.10
                rationale.append(f"Умеренная липофильность (LogP={logp_val:.2f}).")

        tpsa_val = _safe_float(descriptors.get("TPSA"))
        if tpsa_val is not None and tpsa_val < self.rules.tpsa_low:
            score += 0.10
            rationale.append(f"Низкое значение TPSA (TPSA={tpsa_val:.2f}) может указывать на риск экспозиции.")

        return score

    def _domain_score(self, predictions: List[Dict[str, Any]], rationale: List[str]) -> tuple[float, bool]:
        flags = [p.get("in_domain") for p in predictions if p.get("in_domain") is not None]
        if not flags:
            rationale.append("Информация об области применимости отсутствует.")
            return 0.08, False

        ood_count = sum(1 for f in flags if f is False)
        if ood_count == 0:
            return 0.0, False
        if 0 < ood_count < len(flags):
            rationale.append("Смешанное качество области применимости: как минимум одна модель вне области.")
            return self.rules.mixed_domain_penalty, True

        rationale.append("Все модели с оценкой области применимости находятся вне области.")
        return self.rules.ood_penalty, True
