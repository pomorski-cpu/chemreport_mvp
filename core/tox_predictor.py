# core/tox_predictor.py
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict

import joblib
from rdkit import Chem

from core.featurizer_rdkit_inchi import build_feature_df
from core.utils import resource_path


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

    def predict(self, mol: Chem.Mol, *, features_df=None) -> Dict[str, Any]:
        # 1) features (СЃС‚СЂРѕРіРѕ РІ С‚РѕРј Р¶Рµ РїРѕСЂСЏРґРєРµ)
        Xdf = features_df if features_df is not None else build_feature_df(mol)
        Xdf = Xdf.reindex(columns=self.feature_cols, fill_value=0.0)

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
        confidence_score = prob_toxic if prob_toxic is not None else proba
        if prob_toxic is not None and self.toxic_class_id is not None:
            conf_txt = (
                f"P(toxic)={prob_toxic:.3f}; "
                f"threshold={self.decision_threshold:.3f}; "
                f"decision={'toxic' if decision_is_toxic else 'non-toxic'}"
            )
        elif proba is not None:
            conf_txt, confidence_score = self._class_confidence(proba, second_proba)

        notes_bits = []
        if self.toxic_class_id is not None:
            selected_prob = class_prob_map.get(int(final_class), proba)
            if selected_prob is not None:
                notes_bits.append(f"P({label})={selected_prob:.3f}")
        else:
            class_one_prob = class_prob_map.get(1)
            if class_one_prob is not None:
                notes_bits.append(f"1 vs other={class_one_prob:.3f}")
            if proba is not None:
                notes_bits.append(f"top={proba:.3f}")
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

