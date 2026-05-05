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
    bioactivity_prob_medium: float
    bioactivity_prob_high: float
    logp_medium: float
    logp_high: float
    tpsa_low: float
    reliability_low: float
    reliability_review: float
    model_confidence_low: float
    hazard_review: float
    hazard_reject: float
    uncertainty_review: float
    uncertainty_insufficient: float
    combined_review: float
    combined_reject: float


def _safe_float(x: Any) -> Optional[float]:
    if x is None or x == "":
        return None
    try:
        return float(x)
    except Exception:
        return None


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _norm(value: Any) -> str:
    return str(value or "").strip().lower()


class DecisionSupport:
    """Evidence-based screening layer above the local QSAR/read-across models.

    The DSS is intentionally conservative. It separates hazard signals from
    uncertainty signals and keeps the legacy output keys for older renderers.
    """

    def __init__(self, rules_path: str = "config/decision_rules.json"):
        path: Path = resource_path(rules_path)
        with open(path, "r", encoding="utf-8") as f:
            cfg = json.load(f)

        tox = cfg.get("toxicity", {})
        bio = cfg.get("bioactivity", {})
        logp = cfg.get("logp", {})
        tpsa = cfg.get("tpsa", {})
        rel = cfg.get("reliability", {})
        th = cfg.get("decision_thresholds", {})

        self.rules = DecisionRules(
            version=str(cfg.get("version", "dss-v2.0")),
            toxicity_labels=[str(x).lower() for x in tox.get("toxic_labels", ["генотоксичный", "токсичный", "toxic", "1"])],
            toxicity_prob_medium=float(tox.get("prob_medium", 0.55)),
            toxicity_prob_high=float(tox.get("prob_high", 0.70)),
            bioactivity_prob_medium=float(bio.get("prob_medium", tox.get("prob_medium", 0.55))),
            bioactivity_prob_high=float(bio.get("prob_high", tox.get("prob_high", 0.70))),
            logp_medium=float(logp.get("medium", 3.0)),
            logp_high=float(logp.get("high", 4.5)),
            tpsa_low=float(tpsa.get("low", 20.0)),
            reliability_low=float(rel.get("low", 0.40)),
            reliability_review=float(rel.get("review", 0.55)),
            model_confidence_low=float(rel.get("model_confidence_low", 0.55)),
            hazard_review=float(th.get("hazard_review", 0.35)),
            hazard_reject=float(th.get("hazard_reject", 0.65)),
            uncertainty_review=float(th.get("uncertainty_review", 0.35)),
            uncertainty_insufficient=float(th.get("uncertainty_insufficient", 0.70)),
            combined_review=float(th.get("combined_review", 0.45)),
            combined_reject=float(th.get("combined_reject", 0.80)),
        )

    def evaluate(
        self,
        *,
        meta: Optional[Dict[str, Any]] = None,
        descriptors: Optional[Dict[str, Any]] = None,
        predictions: Optional[List[Dict[str, Any]]] = None,
        warnings: Optional[List[str]] = None,
        reliability: Optional[Dict[str, Any]] = None,
        read_across: Optional[Dict[str, Any]] = None,
        category: Optional[Dict[str, Any]] = None,
        analogues: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        meta = meta or {}
        descriptors = descriptors or {}
        predictions = predictions or []
        warnings = warnings or []
        reliability = reliability or {}
        read_across = read_across or {}
        category = category or {}
        analogues = analogues or []

        evidence: List[Dict[str, Any]] = []
        data_quality_flags: List[Dict[str, Any]] = []
        conflicts: List[Dict[str, Any]] = []
        component_scores: Dict[str, float] = {
            "toxicity": 0.0,
            "bioactivity": 0.0,
            "physchem": 0.0,
            "read_across": 0.0,
            "domain": 0.0,
            "reliability": 0.0,
            "conflict": 0.0,
            "data_completeness": 0.0,
        }
        state: Dict[str, Any] = {
            "direct_toxic_prob": None,
            "direct_toxic": None,
            "bioactivity_toxic": [],
            "bioactivity_low": [],
            "read_across_toxic": False,
            "read_across_nontoxic": False,
            "has_ood": False,
            "low_reliability": False,
            "review_reliability": False,
            "ad_borderline": False,
        }

        tox_pred = self._find_direct_toxicity(predictions)
        self._score_toxicity(tox_pred, component_scores, evidence, data_quality_flags, state)
        self._score_bioactivity(predictions, component_scores, evidence, data_quality_flags, state)
        self._score_physchem(descriptors, predictions, component_scores, evidence)
        self._score_domain(predictions, component_scores, evidence, data_quality_flags, state)
        self._score_reliability(reliability, component_scores, evidence, data_quality_flags, state)
        self._score_read_across(read_across, analogues, component_scores, evidence, state)
        self._score_warnings(warnings, component_scores, evidence, data_quality_flags)
        self._detect_conflicts(component_scores, evidence, conflicts, state)

        component_scores["bioactivity"] = min(component_scores["bioactivity"], 0.70)
        component_scores["physchem"] = min(component_scores["physchem"], 0.22)
        component_scores["read_across"] = min(component_scores["read_across"], 0.15)
        component_scores["conflict"] = min(component_scores["conflict"], 0.70)

        hazard_score = _clamp01(
            component_scores["toxicity"]
            + component_scores["bioactivity"]
            + component_scores["physchem"]
            + component_scores["read_across"]
        )
        uncertainty_score = _clamp01(
            component_scores["domain"]
            + component_scores["reliability"]
            + component_scores["conflict"]
            + component_scores["data_completeness"]
        )
        score = _clamp01(hazard_score + 0.25 * uncertainty_score)

        force_insufficient = bool(state["has_ood"] or state["low_reliability"] or uncertainty_score >= self.rules.uncertainty_insufficient)
        force_review = bool(conflicts or state["review_reliability"] or state.get("ad_borderline") or uncertainty_score >= self.rules.uncertainty_review)

        if force_insufficient:
            decision_status = "insufficient_data"
            risk_level = "critical" if hazard_score >= 0.80 else ("high" if hazard_score >= self.rules.hazard_review else "medium")
            recommendation = (
                "Недостаточно надёжных данных для автоматического решения. Нужна ручная экспертная проверка "
                "с учётом области применимости, качества источников и конфликтующих сигналов."
            )
        elif hazard_score >= self.rules.hazard_reject or score >= self.rules.combined_reject:
            decision_status = "reject"
            risk_level = "critical" if hazard_score >= 0.85 else "high"
            recommendation = (
                "Автоматическое одобрение не рекомендуется: выявлен выраженный профиль опасности. "
                "Соединение следует передать на токсикологическую экспертизу или отложить до появления подтверждающих данных."
            )
        elif hazard_score >= self.rules.hazard_review or force_review or score >= self.rules.combined_review:
            decision_status = "review"
            risk_level = "high" if hazard_score >= 0.55 else "medium"
            recommendation = (
                "Требуется ручная экспертная проверка. DSS видит сигналы риска или неопределённости, "
                "которые нельзя закрывать автоматическим допуском."
            )
        else:
            decision_status = "approve"
            risk_level = "low"
            recommendation = (
                "Предварительный скрининг не выявил выраженных сигналов риска при текущей надёжности данных. "
                "Решение остаётся скрининговым, а не регуляторным заключением."
            )

        rationale = self._build_rationale(evidence, conflicts, data_quality_flags)
        next_actions = self._build_next_actions(decision_status, evidence, conflicts, data_quality_flags)

        return {
            "rule_version": self.rules.version,
            "decision_status": decision_status,
            "risk_level": risk_level,
            "score": round(score, 3),
            "hazard_score": round(hazard_score, 3),
            "uncertainty_score": round(uncertainty_score, 3),
            "component_scores": {key: round(_clamp01(value), 3) for key, value in component_scores.items()},
            "evidence": evidence,
            "data_quality_flags": data_quality_flags,
            "conflicts": conflicts,
            "recommendation": recommendation,
            "rationale": rationale,
            "next_actions": next_actions,
            "meta": {
                "source": "rule_based_dss_v2",
                "input": meta.get("input", ""),
                "toxicity": self._toxicity_summary(tox_pred),
                "category": {
                    "type": category.get("type", ""),
                    "summary_ru": category.get("summary_ru", ""),
                    "analogue_count": len(analogues),
                },
            },
        }

    def _add_evidence(
        self,
        evidence: List[Dict[str, Any]],
        component_scores: Dict[str, float],
        *,
        component: str,
        category: str,
        source: str,
        label: str,
        level: str,
        score_delta: float,
        rationale: str,
        value: Any = None,
        threshold: Any = None,
        confidence: Any = None,
    ) -> None:
        if score_delta:
            component_scores[component] += float(score_delta)
        item = {
            "category": category,
            "component": component,
            "source": source,
            "label": label,
            "level": level,
            "score_delta": round(float(score_delta), 3),
            "rationale": rationale,
        }
        if value is not None:
            item["value"] = value
        if threshold is not None:
            item["threshold"] = threshold
        if confidence is not None:
            item["confidence"] = confidence
        evidence.append(item)

    def _add_flag(
        self,
        data_quality_flags: List[Dict[str, Any]],
        *,
        code: str,
        level: str,
        message: str,
        source: str = "DSS",
    ) -> None:
        data_quality_flags.append({"code": code, "level": level, "source": source, "message": message})

    def _find_direct_toxicity(self, predictions: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        for pred in predictions:
            task = _norm(pred.get("task"))
            if "tox" in task or "токс" in task or "genotox" in task:
                if not self._bioactivity_endpoint(pred):
                    return pred
        return None

    def _find_logp_prediction(self, predictions: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        for pred in predictions:
            if "logp" in _norm(pred.get("task")):
                return pred
        return None

    def _bioactivity_endpoint(self, pred: Dict[str, Any]) -> Optional[str]:
        task = _norm(pred.get("task"))
        notes = _norm(pred.get("notes"))
        text = f"{task} {notes}"
        if "ec50" in text:
            return "EC50 водные беспозвоночные"
        if "lc50" in text:
            return "LC50 рыбы"
        if "ld50" in text:
            return "LD50 млекопитающие перорально"
        return None

    def _prob_toxic(self, pred: Optional[Dict[str, Any]]) -> Optional[float]:
        if not pred:
            return None
        prob = _safe_float(pred.get("prob_toxic"))
        if prob is None and isinstance(pred.get("toxicity_decision"), bool):
            prob = _safe_float(pred.get("confidence_score"))
        return prob

    def _label_is_negative(self, value: Any) -> bool:
        text = _norm(value)
        if not text:
            return False
        negative_markers = [
            "нетокс",
            "не токс",
            "не генотокс",
            "non-toxic",
            "non toxic",
            "not toxic",
            "negative",
            "нет",
            "0",
            "рќрµ",
        ]
        return any(marker in text for marker in negative_markers)

    def _label_is_toxic(self, value: Any) -> bool:
        text = _norm(value)
        if not text or self._label_is_negative(text):
            return False
        if text in self.rules.toxicity_labels:
            return True
        markers = ["генотокс", "токсич", "toxic", "positive", "active", "1", "рірµрѕс‚рѕрєсѓ", "с‚рѕрєсѓ"]
        return any(marker in text for marker in markers)

    def _score_toxicity(
        self,
        tox_pred: Optional[Dict[str, Any]],
        component_scores: Dict[str, float],
        evidence: List[Dict[str, Any]],
        data_quality_flags: List[Dict[str, Any]],
        state: Dict[str, Any],
    ) -> None:
        if not tox_pred:
            self._add_flag(
                data_quality_flags,
                code="missing_direct_toxicity",
                level="medium",
                message="Прямая модель токсичности не дала результата.",
                source="Toxicity",
            )
            self._add_evidence(
                evidence,
                component_scores,
                component="data_completeness",
                category="uncertainty",
                source="Toxicity",
                label="Нет прямого прогноза токсичности",
                level="medium",
                score_delta=0.10,
                rationale="Отсутствие прямого токсикологического сигнала повышает неопределённость DSS.",
            )
            return

        prob = self._prob_toxic(tox_pred)
        value = tox_pred.get("value")
        state["direct_toxic_prob"] = prob
        if prob is not None:
            state["direct_toxic"] = prob >= self.rules.toxicity_prob_medium
            if prob >= self.rules.toxicity_prob_high:
                self._add_evidence(
                    evidence,
                    component_scores,
                    component="toxicity",
                    category="hazard",
                    source="Toxicity",
                    label="Высокая вероятность токсичности",
                    level="high",
                    score_delta=0.70,
                    value=round(prob, 3),
                    threshold=self.rules.toxicity_prob_high,
                    confidence=round(prob, 3),
                    rationale=f"Прямая модель токсичности дала P(toxic)={prob:.2f}, что выше порога высокого риска {self.rules.toxicity_prob_high:.2f}.",
                )
            elif prob >= self.rules.toxicity_prob_medium:
                self._add_evidence(
                    evidence,
                    component_scores,
                    component="toxicity",
                    category="hazard",
                    source="Toxicity",
                    label="Умеренная вероятность токсичности",
                    level="medium",
                    score_delta=0.40,
                    value=round(prob, 3),
                    threshold=self.rules.toxicity_prob_medium,
                    confidence=round(prob, 3),
                    rationale=f"Прямая модель токсичности дала P(toxic)={prob:.2f}, что требует ручной проверки.",
                )
            else:
                self._add_evidence(
                    evidence,
                    component_scores,
                    component="toxicity",
                    category="support",
                    source="Toxicity",
                    label="Низкая вероятность токсичности",
                    level="low",
                    score_delta=0.0,
                    value=round(prob, 3),
                    threshold=self.rules.toxicity_prob_medium,
                    confidence=round(prob, 3),
                    rationale=f"Прямая модель токсичности не показывает выраженного toxic-сигнала (P={prob:.2f}).",
                )
            return

        is_toxic = self._label_is_toxic(value)
        state["direct_toxic"] = is_toxic
        if is_toxic:
            self._add_evidence(
                evidence,
                component_scores,
                component="toxicity",
                category="hazard",
                source="Toxicity",
                label="Положительная токсикологическая метка",
                level="medium",
                score_delta=0.40,
                value=value,
                rationale="Модель вернула токсичную метку без вероятностного запаса, поэтому DSS трактует сигнал как умеренный.",
            )
        else:
            self._add_evidence(
                evidence,
                component_scores,
                component="toxicity",
                category="support",
                source="Toxicity",
                label="Токсикологическая метка не указывает на токсичность",
                level="low",
                score_delta=0.0,
                value=value,
                rationale="Прямая токсикологическая метка не содержит положительного toxic-сигнала.",
            )

    def _score_bioactivity(
        self,
        predictions: List[Dict[str, Any]],
        component_scores: Dict[str, float],
        evidence: List[Dict[str, Any]],
        data_quality_flags: List[Dict[str, Any]],
        state: Dict[str, Any],
    ) -> None:
        bio_preds = [(self._bioactivity_endpoint(pred), pred) for pred in predictions]
        bio_preds = [(endpoint, pred) for endpoint, pred in bio_preds if endpoint]
        if not bio_preds:
            self._add_flag(
                data_quality_flags,
                code="missing_bioactivity",
                level="low",
                message="Биоактивные endpoint-модели EC50/LC50/LD50 отсутствуют в текущем прогоне.",
                source="Bioactivity",
            )
            return

        for endpoint, pred in bio_preds:
            prob = self._prob_toxic(pred)
            decision = pred.get("toxicity_decision")
            if prob is not None:
                toxic = prob >= self.rules.bioactivity_prob_medium
                if toxic:
                    state["bioactivity_toxic"].append({"endpoint": endpoint, "prob": prob})
                elif prob <= 0.35:
                    state["bioactivity_low"].append({"endpoint": endpoint, "prob": prob})

                if prob >= self.rules.bioactivity_prob_high:
                    self._add_evidence(
                        evidence,
                        component_scores,
                        component="bioactivity",
                        category="hazard",
                        source=endpoint,
                        label=f"Высокий bioactivity-сигнал: {endpoint}",
                        level="high",
                        score_delta=0.35,
                        value=round(prob, 3),
                        threshold=self.rules.bioactivity_prob_high,
                        confidence=round(prob, 3),
                        rationale=f"{endpoint}: вероятность toxic-класса {prob:.2f}; для EC50/LC50 это соответствует порогу <=1 мг/л, для LD50 oral - <=300 мг/кг.",
                    )
                elif prob >= self.rules.bioactivity_prob_medium:
                    self._add_evidence(
                        evidence,
                        component_scores,
                        component="bioactivity",
                        category="hazard",
                        source=endpoint,
                        label=f"Умеренный bioactivity-сигнал: {endpoint}",
                        level="medium",
                        score_delta=0.18,
                        value=round(prob, 3),
                        threshold=self.rules.bioactivity_prob_medium,
                        confidence=round(prob, 3),
                        rationale=f"{endpoint}: вероятность toxic-класса {prob:.2f}; сигнал недостаточно силён для отклонения, но требует проверки.",
                    )
                else:
                    self._add_evidence(
                        evidence,
                        component_scores,
                        component="bioactivity",
                        category="support",
                        source=endpoint,
                        label=f"Низкий bioactivity toxic-сигнал: {endpoint}",
                        level="low",
                        score_delta=0.0,
                        value=round(prob, 3),
                        threshold=self.rules.bioactivity_prob_medium,
                        confidence=round(prob, 3),
                        rationale=f"{endpoint}: модель не показывает toxic-класс при текущем пороге.",
                    )
            elif decision is True:
                state["bioactivity_toxic"].append({"endpoint": endpoint, "prob": None})
                self._add_evidence(
                    evidence,
                    component_scores,
                    component="bioactivity",
                    category="hazard",
                    source=endpoint,
                    label=f"Bioactivity-модель дала toxic-класс: {endpoint}",
                    level="medium",
                    score_delta=0.18,
                    rationale=f"{endpoint}: toxic-класс получен без вероятности, поэтому вклад ограничен.",
                )
            elif decision is False:
                state["bioactivity_low"].append({"endpoint": endpoint, "prob": None})

    def _score_physchem(
        self,
        descriptors: Dict[str, Any],
        predictions: List[Dict[str, Any]],
        component_scores: Dict[str, float],
        evidence: List[Dict[str, Any]],
    ) -> None:
        logp_val = _safe_float(descriptors.get("cLogP"))
        logp_pred = self._find_logp_prediction(predictions)
        if logp_val is None and logp_pred:
            logp_val = _safe_float(logp_pred.get("value"))
        if logp_val is not None:
            if logp_val >= self.rules.logp_high:
                self._add_evidence(
                    evidence,
                    component_scores,
                    component="physchem",
                    category="hazard",
                    source="LogP",
                    label="Высокая липофильность",
                    level="high",
                    score_delta=0.15,
                    value=round(logp_val, 3),
                    threshold=self.rules.logp_high,
                    rationale=f"LogP={logp_val:.2f} выше порога {self.rules.logp_high:.1f}; это может усиливать экспозицию и биоаккумуляционный риск.",
                )
            elif logp_val >= self.rules.logp_medium:
                self._add_evidence(
                    evidence,
                    component_scores,
                    component="physchem",
                    category="hazard",
                    source="LogP",
                    label="Умеренно повышенная липофильность",
                    level="medium",
                    score_delta=0.08,
                    value=round(logp_val, 3),
                    threshold=self.rules.logp_medium,
                    rationale=f"LogP={logp_val:.2f} выше скринингового порога {self.rules.logp_medium:.1f}.",
                )

        tpsa_val = _safe_float(descriptors.get("TPSA"))
        if tpsa_val is not None and tpsa_val < self.rules.tpsa_low:
            self._add_evidence(
                evidence,
                component_scores,
                component="physchem",
                category="hazard",
                source="TPSA",
                label="Низкая полярная поверхность",
                level="medium",
                score_delta=0.08,
                value=round(tpsa_val, 3),
                threshold=self.rules.tpsa_low,
                rationale=f"TPSA={tpsa_val:.2f} ниже {self.rules.tpsa_low:.1f}; для скрининга это трактуется как фактор возможной проницаемости.",
            )

    def _score_domain(
        self,
        predictions: List[Dict[str, Any]],
        component_scores: Dict[str, float],
        evidence: List[Dict[str, Any]],
        data_quality_flags: List[Dict[str, Any]],
        state: Dict[str, Any],
    ) -> None:
        flags = [pred.get("in_domain") for pred in predictions if pred.get("in_domain") is not None]
        if not flags:
            self._add_flag(
                data_quality_flags,
                code="missing_ad",
                level="medium",
                message="Нет информации об области применимости для рассчитанных моделей.",
                source="Applicability domain",
            )
            self._add_evidence(
                evidence,
                component_scores,
                component="data_completeness",
                category="uncertainty",
                source="Applicability domain",
                label="AD-информация отсутствует",
                level="medium",
                score_delta=0.10,
                rationale="Без AD-оценки DSS не может полноценно проверить применимость моделей к молекуле.",
            )
            return

        if any(flag is False for flag in flags):
            state["has_ood"] = True
            self._add_flag(
                data_quality_flags,
                code="out_of_domain",
                level="high",
                message="Как минимум одна модель находится вне области применимости.",
                source="Applicability domain",
            )
            self._add_evidence(
                evidence,
                component_scores,
                component="domain",
                category="uncertainty",
                source="Applicability domain",
                label="Выход за область применимости",
                level="high",
                score_delta=0.70,
                rationale="Политика DSS консервативна: любой in_domain=False переводит молекулу в ручную проверку/недостаточность данных.",
            )

        borderline = [
            pred for pred in predictions
            if str(pred.get("ad_status") or "").lower() == "borderline"
        ]
        if borderline:
            state["ad_borderline"] = True
            self._add_flag(
                data_quality_flags,
                code="borderline_ad",
                level="medium",
                message=f"Пограничная область применимости для {len(borderline)} модели(ей).",
                source="Applicability domain",
            )
            labels = ", ".join(str(pred.get("task", "model")) for pred in borderline[:4])
            self._add_evidence(
                evidence,
                component_scores,
                component="domain",
                category="uncertainty",
                source="Applicability domain",
                label="Пограничная область применимости",
                level="medium",
                score_delta=0.25,
                rationale=f"Модели в пограничной зоне AD: {labels}. Такие прогнозы можно использовать для скрининга, но не для автоматического approve.",
            )

    def _score_reliability(
        self,
        reliability: Dict[str, Any],
        component_scores: Dict[str, float],
        evidence: List[Dict[str, Any]],
        data_quality_flags: List[Dict[str, Any]],
        state: Dict[str, Any],
    ) -> None:
        if not reliability:
            self._add_flag(
                data_quality_flags,
                code="missing_reliability",
                level="medium",
                message="Сводная оценка надёжности недоступна.",
                source="Reliability",
            )
            self._add_evidence(
                evidence,
                component_scores,
                component="data_completeness",
                category="uncertainty",
                source="Reliability",
                label="Нет сводной оценки надёжности",
                level="medium",
                score_delta=0.10,
                rationale="DSS не получил итоговую reliability-оценку workflow.",
            )
            return

        final_score = _safe_float(reliability.get("final_score"))
        model_confidence = _safe_float(reliability.get("model_confidence"))
        if final_score is not None:
            if final_score < self.rules.reliability_low:
                state["low_reliability"] = True
                self._add_flag(
                    data_quality_flags,
                    code="low_reliability",
                    level="high",
                    message=f"Сводная надёжность {final_score:.2f} ниже критического порога {self.rules.reliability_low:.2f}.",
                    source="Reliability",
                )
                self._add_evidence(
                    evidence,
                    component_scores,
                    component="reliability",
                    category="uncertainty",
                    source="Reliability",
                    label="Критически низкая надёжность",
                    level="high",
                    score_delta=0.65,
                    value=round(final_score, 3),
                    threshold=self.rules.reliability_low,
                    rationale="Критически низкая reliability запрещает автоматическое одобрение даже при слабом hazard-сигнале.",
                )
            elif final_score < self.rules.reliability_review:
                state["review_reliability"] = True
                self._add_flag(
                    data_quality_flags,
                    code="questionable_reliability",
                    level="medium",
                    message=f"Сводная надёжность {final_score:.2f} ниже порога уверенного автоматического решения.",
                    source="Reliability",
                )
                self._add_evidence(
                    evidence,
                    component_scores,
                    component="reliability",
                    category="uncertainty",
                    source="Reliability",
                    label="Сомнительная надёжность",
                    level="medium",
                    score_delta=0.35,
                    value=round(final_score, 3),
                    threshold=self.rules.reliability_review,
                    rationale="Надёжность ниже порога review, поэтому DSS не должен автоматически одобрять молекулу.",
                )

        if model_confidence is not None and model_confidence < self.rules.model_confidence_low:
            state["review_reliability"] = True
            self._add_flag(
                data_quality_flags,
                code="low_model_confidence",
                level="medium",
                message=f"Средняя уверенность моделей {model_confidence:.2f} ниже порога {self.rules.model_confidence_low:.2f}.",
                source="Reliability",
            )
            self._add_evidence(
                evidence,
                component_scores,
                component="reliability",
                category="uncertainty",
                source="Reliability",
                label="Низкая уверенность моделей",
                level="medium",
                score_delta=0.20,
                value=round(model_confidence, 3),
                threshold=self.rules.model_confidence_low,
                rationale="Низкая уверенность моделей повышает неопределённость итогового решения.",
            )

    def _parse_read_across_confidence(self, prediction: Dict[str, Any]) -> Optional[float]:
        direct = _safe_float(prediction.get("confidence_score"))
        if direct is not None:
            return direct
        text = _norm(prediction.get("confidence"))
        if "выс" in text or "high" in text:
            return 0.85
        if "сред" in text or "medium" in text:
            return 0.60
        if "низ" in text or "low" in text:
            return 0.40
        return None

    def _score_read_across(
        self,
        read_across: Dict[str, Any],
        analogues: List[Dict[str, Any]],
        component_scores: Dict[str, float],
        evidence: List[Dict[str, Any]],
        state: Dict[str, Any],
    ) -> None:
        targets = read_across.get("targets", {}) or {}
        for target_key, target_data in targets.items():
            prediction = target_data.get("prediction") or {}
            label = str(target_data.get("label_ru") or target_key)
            task_text = _norm(f"{target_key} {label} {prediction.get('task', '')}")
            if not any(marker in task_text for marker in ["tox", "токс", "ec50", "lc50", "ld50"]):
                continue

            value = prediction.get("value")
            confidence = self._parse_read_across_confidence(prediction)
            is_toxic = self._label_is_toxic(value) or prediction.get("toxicity_decision") is True
            is_nontoxic = self._label_is_negative(value) or prediction.get("toxicity_decision") is False
            if is_toxic:
                state["read_across_toxic"] = True
                delta = 0.12 if (confidence is None or confidence >= 0.70) else 0.07
                self._add_evidence(
                    evidence,
                    component_scores,
                    component="read_across",
                    category="hazard",
                    source=f"Read-across: {label}",
                    label="Токсичный сигнал по аналогам",
                    level="medium" if delta < 0.12 else "high",
                    score_delta=delta,
                    value=value,
                    confidence=confidence,
                    rationale=f"Read-across для «{label}» согласуется с токсичным/опасным классом; это усиливает hazard, но не заменяет прямую модель.",
                )
            elif is_nontoxic:
                state["read_across_nontoxic"] = True
                self._add_evidence(
                    evidence,
                    component_scores,
                    component="read_across",
                    category="support",
                    source=f"Read-across: {label}",
                    label="Аналоги не указывают на токсичность",
                    level="low",
                    score_delta=0.0,
                    value=value,
                    confidence=confidence,
                    rationale=f"Read-across для «{label}» не показывает toxic-сигнал.",
                )

        if analogues and not targets:
            self._add_evidence(
                evidence,
                component_scores,
                component="read_across",
                category="support",
                source="Read-across",
                label="Найдены структурные аналоги",
                level="low",
                score_delta=0.0,
                value=len(analogues),
                rationale="Аналоги найдены, но без endpoint-специфичного токсикологического вывода.",
            )

    def _score_warnings(
        self,
        warnings: List[str],
        component_scores: Dict[str, float],
        evidence: List[Dict[str, Any]],
        data_quality_flags: List[Dict[str, Any]],
    ) -> None:
        text = " ".join(str(w) for w in warnings).lower()
        if any(marker in text for marker in ["fragment", "фрагмент", "salt", "соль"]):
            self._add_flag(
                data_quality_flags,
                code="fragment_or_salt",
                level="medium",
                message="Есть предупреждение о фрагментах или солевой форме.",
                source="Workflow",
            )
            self._add_evidence(
                evidence,
                component_scores,
                component="data_completeness",
                category="uncertainty",
                source="Workflow warnings",
                label="Фрагменты или солевая форма",
                level="medium",
                score_delta=0.10,
                rationale="Фрагментированная структура или соль может искажать дескрипторы и область применимости.",
            )

    def _detect_conflicts(
        self,
        component_scores: Dict[str, float],
        evidence: List[Dict[str, Any]],
        conflicts: List[Dict[str, Any]],
        state: Dict[str, Any],
    ) -> None:
        direct_prob = state.get("direct_toxic_prob")
        bio_toxic = state.get("bioactivity_toxic") or []
        bio_low = state.get("bioactivity_low") or []

        if direct_prob is not None and direct_prob <= 0.35 and bio_toxic:
            self._add_conflict(
                conflicts,
                component_scores,
                evidence,
                code="toxicity_vs_bioactivity",
                message="Прямая токсичность низкая, но bioactivity-модели дают toxic-сигнал.",
                sources=["Toxicity", ", ".join(item["endpoint"] for item in bio_toxic)],
            )
        if direct_prob is not None and direct_prob >= self.rules.toxicity_prob_high and bio_low and not bio_toxic:
            self._add_conflict(
                conflicts,
                component_scores,
                evidence,
                code="toxicity_high_bioactivity_low",
                message="Прямая токсичность высокая, но bioactivity-endpoint модели не подтверждают toxic-класс.",
                sources=["Toxicity", ", ".join(item["endpoint"] for item in bio_low)],
            )
        if direct_prob is not None and direct_prob <= 0.35 and state.get("read_across_toxic"):
            self._add_conflict(
                conflicts,
                component_scores,
                evidence,
                code="toxicity_vs_read_across",
                message="Read-across указывает на токсичность при низкой вероятности прямой модели.",
                sources=["Toxicity", "Read-across"],
            )
        if direct_prob is not None and direct_prob >= self.rules.toxicity_prob_high and state.get("read_across_nontoxic"):
            self._add_conflict(
                conflicts,
                component_scores,
                evidence,
                code="toxic_model_read_across_nontoxic",
                message="Прямая модель токсичности и read-across дают разные выводы.",
                sources=["Toxicity", "Read-across"],
            )

    def _add_conflict(
        self,
        conflicts: List[Dict[str, Any]],
        component_scores: Dict[str, float],
        evidence: List[Dict[str, Any]],
        *,
        code: str,
        message: str,
        sources: List[str],
    ) -> None:
        conflicts.append({"code": code, "level": "medium", "message": message, "sources": sources})
        self._add_evidence(
            evidence,
            component_scores,
            component="conflict",
            category="uncertainty",
            source="; ".join(sources),
            label="Конфликт сигналов",
            level="medium",
            score_delta=0.35,
            rationale=message,
        )

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

    def _build_rationale(
        self,
        evidence: List[Dict[str, Any]],
        conflicts: List[Dict[str, Any]],
        flags: List[Dict[str, Any]],
    ) -> List[str]:
        lines: List[str] = []
        hazard_items = sorted(
            [item for item in evidence if item.get("category") == "hazard" and item.get("score_delta", 0) > 0],
            key=lambda item: item.get("score_delta", 0),
            reverse=True,
        )
        uncertainty_items = sorted(
            [item for item in evidence if item.get("category") == "uncertainty" and item.get("score_delta", 0) > 0],
            key=lambda item: item.get("score_delta", 0),
            reverse=True,
        )
        for item in hazard_items[:4]:
            lines.append(str(item.get("rationale") or item.get("label")))
        for item in uncertainty_items[:3]:
            lines.append(str(item.get("rationale") or item.get("label")))
        for conflict in conflicts[:2]:
            lines.append(f"Конфликт: {conflict.get('message')}")
        if not lines:
            support = [item for item in evidence if item.get("category") == "support"]
            if support:
                lines.append("Основные модели не показали выраженных hazard-сигналов при текущих порогах.")
            else:
                lines.append("DSS не получил достаточно специфичных сигналов для расширенного объяснения.")
        if flags and not uncertainty_items:
            lines.append(flags[0].get("message", "Есть флаги качества данных."))
        return lines

    def _build_next_actions(
        self,
        decision_status: str,
        evidence: List[Dict[str, Any]],
        conflicts: List[Dict[str, Any]],
        flags: List[Dict[str, Any]],
    ) -> List[str]:
        actions: List[str] = []
        if decision_status == "approve":
            actions.append("Перевести соединение на следующий этап скрининга с обычным контролем качества данных.")
        elif decision_status == "reject":
            actions.append("Передать соединение на токсикологическую экспертизу и проверить исходные endpoint-прогнозы.")
            actions.append("Рассмотреть снижение приоритета кандидата до появления подтверждающих экспериментальных данных.")
        elif decision_status == "insufficient_data":
            actions.append("Проверить область применимости, надёжность аналогов и корректность исходной структуры.")
            actions.append("Не использовать автоматический вывод DSS как основание для допуска без экспертной проверки.")
        else:
            actions.append("Провести ручную проверку ключевых hazard-сигналов и источников неопределённости.")

        if conflicts:
            actions.append("Разобрать конфликтующие источники отдельно: прямые модели, bioactivity и read-across.")
        if any(flag.get("code") == "out_of_domain" for flag in flags):
            actions.append("Подобрать более близкие референсные соединения или расширить applicability domain.")
        if any(item.get("component") == "bioactivity" and item.get("category") == "hazard" for item in evidence):
            actions.append("Проверить EC50/LC50/LD50 endpoint и токсикологические пороги, использованные для binary toxic-класса.")
        return actions
