# core/tox_predictor.py
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict

import joblib
from rdkit import Chem

from core.featurizer_rdkit_inchi import build_feature_df
from core.utils import resource_path


def _normalize_multilabel_items(labels: list[Any] | tuple[Any, ...] | set[Any] | str | None) -> set[str]:
    if labels is None:
        return set()
    if isinstance(labels, str):
        raw_items = labels.replace("|", ";").split(";")
    else:
        raw_items = list(labels)
    items = {str(item).strip() for item in raw_items if str(item).strip()}
    main_items = {item for item in items if item.lower() != "other"}
    return main_items or items


def multilabel_hit_metrics(
    predicted_labels: list[Any] | tuple[Any, ...] | set[Any] | str | None,
    true_labels: list[Any] | tuple[Any, ...] | set[Any] | str | None,
) -> Dict[str, Any]:
    predicted = _normalize_multilabel_items(predicted_labels)
    truth = _normalize_multilabel_items(true_labels)
    intersection = predicted & truth
    union = predicted | truth
    return {
        "predicted_labels": sorted(predicted),
        "true_labels": sorted(truth),
        "any_hit": bool(intersection),
        "all_hit": bool(truth) and truth.issubset(predicted),
        "exact_match": bool(truth) and predicted == truth,
        "jaccard": round((len(intersection) / len(union)) if union else 0.0, 3),
        "extra_label_count": len(predicted - truth),
    }


@dataclass(frozen=True)
class ToxPaths:
    pipeline_pkl: str = "models/mlp_tox_pipeline.pkl"          # РїРѕРјРµРЅСЏР№ РїРѕРґ СЂРµР°Р»СЊРЅРѕРµ РёРјСЏ
    meta_json: str = "models/mlp_tox_pipeline_meta.json"       # РїРѕРјРµРЅСЏР№ РїРѕРґ СЂРµР°Р»СЊРЅРѕРµ РёРјСЏ


class ToxPredictor:
    def __init__(self, paths: ToxPaths | None = None):
        self._paths = paths or ToxPaths()

        self.model = None
        self.meta: Dict[str, Any] = {}
        self.feature_cols: list[str] = []
        self.class_names: Dict[str, str] = {}
        self.classes: list[int] = []
        self.toxic_class_id: int | None = None
        self.non_toxic_class_id: int | None = None
        self.decision_threshold: float = 0.5

        self._load_assets()

    def _is_toxic_task(self) -> bool:
        task_markers = [
            str(self.meta.get("target_name", "")).lower(),
            str(self.meta.get("name", "")).lower(),
        ]
        return any(("tox" in marker) or ("генотокс" in marker) or ("genotox" in marker) for marker in task_markers)

    def _load_assets(self) -> None:
        self.model = joblib.load(resource_path(self._paths.pipeline_pkl))
        self._force_single_thread(self.model)

        with open(resource_path(self._paths.meta_json), "r", encoding="utf-8-sig") as f:
            self.meta = json.load(f)

        self.feature_cols = list(self.meta["feature_cols"])
        class_names = self.meta.get("class_names")
        self.class_names = class_names if isinstance(class_names, dict) else {}
        self.classes = [int(x) for x in self.meta.get("classes", [])]
        self.decision_threshold = float(
            self.meta.get("decision_threshold", self.meta.get("toxicity_threshold", 0.5))
        )
        self.toxic_class_id = self._resolve_toxic_class_id()
        self.non_toxic_class_id = self._resolve_non_toxic_class_id()

    def _force_single_thread(self, model: Any) -> None:
        if hasattr(model, "n_jobs"):
            try:
                model.n_jobs = 1
            except Exception:
                pass
        for _, step in getattr(model, "steps", []) or []:
            self._force_single_thread(step)

    def _normalize_label(self, cls_id: int) -> str:
        return str(self.class_names.get(str(cls_id), self.class_names.get(cls_id, str(cls_id))))

    def _is_toxic_label(self, label: str) -> bool:
        t = str(label).strip().lower()
        if not t:
            return False
        is_negative = t.startswith("не ") or t.startswith("non-") or t.startswith("not ")
        has_toxic_token = ("токс" in t) or ("toxic" in t) or ("genotox" in t)
        return has_toxic_token and not is_negative

    def _resolve_toxic_class_id(self) -> int | None:
        if not self._is_toxic_task():
            return None

        ids: list[int] = []
        if self.class_names:
            for raw_key in self.class_names.keys():
                try:
                    ids.append(int(raw_key))
                except Exception:
                    pass
        if not ids:
            ids = list(self.classes)

        for cls_id in ids:
            if self._is_toxic_label(self._normalize_label(cls_id)):
                return cls_id
        if len(ids) == 2 and 1 in ids:
            return 1
        return None

    def _resolve_non_toxic_class_id(self) -> int | None:
        if self.toxic_class_id is None:
            return None
        for cls_id in self.classes:
            if int(cls_id) != int(self.toxic_class_id):
                return int(cls_id)
        return None

    def _class_confidence(self, top_prob: float | None, second_prob: float | None) -> tuple[str, float | None]:
        if top_prob is None:
            return "", None

        runner_up = second_prob or 0.0
        pairwise_score = top_prob / (top_prob + runner_up) if (top_prob + runner_up) > 0 else 0.0
        margin = top_prob - runner_up

        if pairwise_score >= 0.80 and margin >= 0.15:
            return "Высокая", pairwise_score
        if pairwise_score >= 0.65 and margin >= 0.07:
            return "Средняя", pairwise_score
        return "Низкая", pairwise_score

    def _is_multilabel_task(self) -> bool:
        return str(self.meta.get("task_type", "")).lower() == "multilabel_classification"

    def _multilabel_probabilities(self, Xdf) -> list[float]:
        if hasattr(self.model, "predict_proba"):
            raw = self.model.predict_proba(Xdf)
            if hasattr(raw, "tolist"):
                raw = raw.tolist()
            if raw and isinstance(raw[0], list):
                return [float(x) for x in raw[0]]
            return [float(x) for x in raw]

        raw_pred = self.model.predict(Xdf)
        if hasattr(raw_pred, "tolist"):
            raw_pred = raw_pred.tolist()
        if raw_pred and isinstance(raw_pred[0], list):
            return [float(x) for x in raw_pred[0]]
        return [float(x) for x in raw_pred]

    def _predict_multilabel(self, Xdf) -> Dict[str, Any]:
        labels = list(self.meta.get("labels") or [])
        if not labels:
            labels = [self.class_names.get(str(i), str(i)) for i in range(len(self.classes))]

        probs = self._multilabel_probabilities(Xdf)
        thresholds_map = self.meta.get("label_thresholds") if isinstance(self.meta.get("label_thresholds"), dict) else {}
        thresholds = [float(thresholds_map.get(label, 0.5)) for label in labels]

        selected_idx = [i for i, prob in enumerate(probs[: len(labels)]) if prob >= thresholds[i]]
        if not selected_idx and probs:
            selected_idx = [int(max(range(min(len(labels), len(probs))), key=lambda i: probs[i]))]

        selected_labels = [labels[i] for i in selected_idx]
        main_labels = [label for label in selected_labels if str(label).strip().lower() != "other"]
        if main_labels:
            selected_labels = main_labels
            selected_idx = [labels.index(label) for label in selected_labels]

        separator = str(self.meta.get("multilabel_separator", "; "))
        value = separator.join(selected_labels)
        selected_probs = [probs[i] for i in selected_idx] if selected_idx else []
        confidence_score = float(sum(selected_probs) / len(selected_probs)) if selected_probs else None

        if confidence_score is None:
            conf_txt = ""
        elif confidence_score >= 0.75:
            conf_txt = "Высокая"
        elif confidence_score >= 0.55:
            conf_txt = "Средняя"
        else:
            conf_txt = "Низкая"

        selected_bits = [str(label) for label in selected_labels]
        label_probabilities = {
            str(label): float(probs[i])
            for i, label in enumerate(labels[: len(probs)])
        }
        notes_bits = [
            "метки: " + (", ".join(selected_bits) if selected_bits else value),
        ]

        return {
            "task": self.meta.get("target_name", "Pesticide Class"),
            "value": value,
            "confidence": conf_txt,
            "prob_toxic": None,
            "toxicity_threshold": None,
            "toxicity_decision": None,
            "confidence_score": confidence_score,
            "predicted_labels": selected_bits,
            "label_probabilities": label_probabilities,
            "multilabel_success_metric": "at_least_one_hit",
            "ad_distance": None,
            "ad_threshold": None,
            "ad_ratio": None,
            "ad_score": None,
            "in_domain": None,
            "notes": "; ".join(notes_bits),
        }

    def predict(self, mol: Chem.Mol, *, features_df=None) -> Dict[str, Any]:
        # 1) features (СЃС‚СЂРѕРіРѕ РІ С‚РѕРј Р¶Рµ РїРѕСЂСЏРґРєРµ)
        Xdf = features_df if features_df is not None else build_feature_df(mol)
        Xdf = Xdf.reindex(columns=self.feature_cols, fill_value=0.0)

        if self._is_multilabel_task():
            return self._predict_multilabel(Xdf)

        # 2) prediction
        y = int(self.model.predict(Xdf)[0])

        proba = None
        prob_toxic = None
        second_proba = None
        pred_class = y
        class_prob_map: Dict[int, float] = {}
        if hasattr(self.model, "predict_proba"):
            p = self.model.predict_proba(Xdf)[0]
            classes = list(getattr(self.model, "classes_", range(len(p))))
            class_probs = [(int(cls), float(prob)) for cls, prob in zip(classes, p)]
            class_prob_map = {cls: prob for cls, prob in class_probs}
            class_probs.sort(key=lambda x: x[1], reverse=True)

            if class_probs:
                pred_class = class_probs[0][0]
                proba = class_probs[0][1]
                if len(class_probs) > 1:
                    second_proba = class_probs[1][1]

            if self.toxic_class_id is not None:
                prob_toxic = class_prob_map.get(int(self.toxic_class_id))
            elif len(class_probs) == 2 and len(class_prob_map) == 2:
                # Binary fallback: assume larger class id is the positive/toxic class.
                fallback_toxic = sorted(class_prob_map.keys())[-1]
                prob_toxic = class_prob_map.get(fallback_toxic)

        decision_is_toxic = None
        if prob_toxic is not None:
            decision_is_toxic = prob_toxic >= self.decision_threshold
        elif self.toxic_class_id is not None:
            decision_is_toxic = int(pred_class) == int(self.toxic_class_id)

        final_class = pred_class
        if decision_is_toxic is True and self.toxic_class_id is not None:
            final_class = int(self.toxic_class_id)
        elif decision_is_toxic is False and self.non_toxic_class_id is not None:
            final_class = int(self.non_toxic_class_id)

        label = self.class_names.get(str(final_class), self.class_names.get(final_class, str(final_class)))

        conf_txt = ""
        confidence_score = max(prob_toxic, 1.0 - prob_toxic) if prob_toxic is not None else proba
        if prob_toxic is not None and self.toxic_class_id is not None:
            conf_txt = (
                f"P(токсичности)={prob_toxic:.3f}; "
                f"порог={self.decision_threshold:.3f}; "
                f"решение={'токсично' if decision_is_toxic else 'нетоксично'}"
            )
        elif proba is not None:
            conf_txt, confidence_score = self._class_confidence(proba, second_proba)

        notes_bits = []
        if self.toxic_class_id is not None:
            selected_prob = class_prob_map.get(int(final_class), proba)
            if selected_prob is not None:
                notes_bits.append(f"P({label})={selected_prob:.3f}")
        elif class_prob_map:
            ranked = sorted(class_prob_map.items(), key=lambda item: item[1], reverse=True)
            top_class, top_prob = ranked[0]
            top_label = self.class_names.get(str(top_class), self.class_names.get(top_class, str(top_class)))
            notes_bits.append(f"наиболее вероятный класс: {top_label} (P={top_prob:.3f})")
            top3 = []
            for class_id, prob in ranked[:3]:
                class_label = self.class_names.get(str(class_id), self.class_names.get(class_id, str(class_id)))
                top3.append(f"{class_label}: {prob:.3f}")
            notes_bits.append("топ-3 вероятности: " + ", ".join(top3))
        notes = "; ".join(notes_bits)

        return {
            "task": self.meta.get("target_name", "Toxicity"),
            "value": label,
            "confidence": conf_txt,
            "prob_toxic": prob_toxic,
            "toxicity_threshold": self.decision_threshold if self.toxic_class_id is not None else None,
            "toxicity_decision": decision_is_toxic,
            "confidence_score": confidence_score,
            "ad_distance": None,
            "ad_threshold": None,
            "ad_ratio": None,
            "ad_score": None,
            "in_domain": None,
            "notes": notes,
        }
