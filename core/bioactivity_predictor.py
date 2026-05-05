from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Dict

import joblib
import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import Descriptors, Fragments, MACCSkeys, rdFingerprintGenerator

from core.utils import resource_path


DESC_FUNCS = {
    "MolWt": Descriptors.MolWt,
    "MolLogP": Descriptors.MolLogP,
    "TPSA": Descriptors.TPSA,
    "NumHAcceptors": Descriptors.NumHAcceptors,
    "NumHDonors": Descriptors.NumHDonors,
    "NumRotatableBonds": Descriptors.NumRotatableBonds,
    "RingCount": Descriptors.RingCount,
    "NumAromaticRings": Descriptors.NumAromaticRings,
    "NumAliphaticRings": Descriptors.NumAliphaticRings,
    "HeavyAtomCount": Descriptors.HeavyAtomCount,
    "FractionCSP3": Descriptors.FractionCSP3,
    "BertzCT": Descriptors.BertzCT,
    "BalabanJ": Descriptors.BalabanJ,
}

ATOMS = ["C", "N", "O", "S", "P", "F", "Cl", "Br", "I"]
SMARTS = {
    "smarts_halogen": "[F,Cl,Br,I]",
    "smarts_trifluoromethyl": "C(F)(F)F",
    "smarts_nitro": "[N+](=O)[O-]",
    "smarts_nitrile": "C#N",
    "smarts_phosphate": "P(=O)",
    "smarts_thiophosphate": "P(=S)",
    "smarts_carbamate": "N-C(=O)-O",
    "smarts_urea": "N-C(=O)-N",
    "smarts_amide": "C(=O)-N",
    "smarts_phenyl": "c1ccccc1",
    "smarts_pyridine": "n1ccccc1",
    "smarts_triazine": "n1cncnc1",
    "smarts_pyrazole": "n1nccc1",
}

RULE_LABELS_RU = {
    "toxic if EC50 <= 1 mg/L": "токсично при EC50 <= 1 мг/л",
    "toxic if LC50 <= 1 mg/L": "токсично при LC50 <= 1 мг/л",
    "toxic if oral LD50 <= 300 mg/kg": "токсично при пероральном LD50 <= 300 мг/кг",
}
DATASET_LABELS_RU = {
    "primary_only": "только основные PPDB-колонки",
    "clean_hybrid": "очищенный гибридный набор",
    "primary_tropical_fallback": "основные PPDB-колонки + совместимый fallback",
    "referencepoints_strict": "строгий набор referencepoints",
}
FEATURE_SCHEMA_LABELS_RU = {
    "bioactivity_v2_rdkit_descriptors_fragments_smarts_no_fingerprints": (
        "RDKit-дескрипторы, фрагменты, SMARTS; без fingerprint-признаков"
    ),
    "bioactivity_v2_rdkit_maccs_morgan_no_aux": (
        "RDKit-дескрипторы, фрагменты, SMARTS, MACCS и Morgan; без auxiliary-признаков"
    ),
}


def _label_ru(mapping: dict[str, str], value: Any, fallback: str = "-") -> str:
    text = str(value or "").strip()
    if not text:
        return fallback
    return mapping.get(text, text)
SMARTS_PATTERNS = {key: Chem.MolFromSmarts(value) for key, value in SMARTS.items()}
FRAGMENT_FUNCS = {
    name: func for name, func in vars(Fragments).items()
    if name.startswith("fr_") and callable(func)
}
MORGAN_GENERATORS: dict[int, Any] = {}
BIOACTIVITY_REGRESSION_ARTIFACT_CACHE: dict[str, Any] = {}


@dataclass(frozen=True)
class BioactivityPaths:
    pipeline_pkl: str
    meta_json: str


def _finite_frame(df: pd.DataFrame) -> pd.DataFrame:
    return df.replace([np.inf, -np.inf], np.nan).fillna(0.0)


def _morgan_generator(n_bits: int = 1024):
    bits = int(n_bits)
    if bits not in MORGAN_GENERATORS:
        MORGAN_GENERATORS[bits] = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=bits)
    return MORGAN_GENERATORS[bits]


def _needs_prefix(feature_cols: list[str] | tuple[str, ...] | None, prefix: str) -> bool:
    return any(str(col).startswith(prefix) for col in (feature_cols or []))


def _featurize_uncached(
    smiles: str,
    *,
    include_maccs: bool = False,
    include_morgan: bool = False,
    morgan_bits: int = 1024,
) -> Dict[str, Any]:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError("Invalid SMILES for bioactivity featurization")

    row: Dict[str, Any] = {"canonical_smiles": Chem.MolToSmiles(mol, canonical=True)}
    for name, func in DESC_FUNCS.items():
        try:
            row[f"desc_{name}"] = float(func(mol))
        except Exception:
            row[f"desc_{name}"] = 0.0

    counts = Counter(atom.GetSymbol() for atom in mol.GetAtoms())
    for atom in ATOMS:
        row[f"atom_{atom}"] = int(counts.get(atom, 0))

    for name, pattern in SMARTS_PATTERNS.items():
        row[name] = int(len(mol.GetSubstructMatches(pattern))) if pattern is not None else 0

    for name, func in FRAGMENT_FUNCS.items():
        try:
            row[name] = int(func(mol))
        except Exception:
            row[name] = 0

    if include_maccs:
        maccs = MACCSkeys.GenMACCSKeys(mol)
        for bit in range(1, maccs.GetNumBits()):
            row[f"maccs_{bit}"] = int(maccs.GetBit(bit))

    if include_morgan:
        fp = _morgan_generator(morgan_bits).GetFingerprint(mol)
        arr = np.zeros((int(morgan_bits),), dtype=np.int8)
        DataStructs.ConvertToNumpyArray(fp, arr)
        for bit, value in enumerate(arr):
            row[f"morgan_{bit}"] = int(value)

    return row


@lru_cache(maxsize=4096)
def _cached_feature_items(
    smiles: str,
    include_maccs: bool,
    include_morgan: bool,
    morgan_bits: int,
) -> tuple[tuple[str, Any], ...]:
    row = _featurize_uncached(
        smiles,
        include_maccs=include_maccs,
        include_morgan=include_morgan,
        morgan_bits=morgan_bits,
    )
    return tuple(row.items())


def featurize_mol_v2(mol: Chem.Mol, feature_cols: list[str] | tuple[str, ...] | None = None) -> Dict[str, Any]:
    smiles = Chem.MolToSmiles(mol, canonical=True)
    include_maccs = _needs_prefix(feature_cols, "maccs_")
    include_morgan = _needs_prefix(feature_cols, "morgan_")
    morgan_bits = 1024
    if feature_cols:
        morgan_indices = [
            int(str(col).split("_", 1)[1])
            for col in feature_cols
            if str(col).startswith("morgan_") and str(col).split("_", 1)[1].isdigit()
        ]
        if morgan_indices:
            morgan_bits = max(morgan_indices) + 1
    return dict(_cached_feature_items(smiles, include_maccs, include_morgan, morgan_bits))


def build_bioactivity_feature_df(mol: Chem.Mol, feature_cols: list[str] | None = None) -> pd.DataFrame:
    row = featurize_mol_v2(mol, feature_cols=feature_cols)
    df = pd.DataFrame([row])
    if feature_cols is not None:
        df = df.reindex(columns=feature_cols, fill_value=0.0)
    numeric_cols = [col for col in df.columns if col != "canonical_smiles"]
    if numeric_cols:
        df[numeric_cols] = _finite_frame(df[numeric_cols])
    return df


class BioactivityBinaryPredictor:
    def __init__(self, paths: BioactivityPaths):
        self._paths = paths
        self.model = None
        self.meta: Dict[str, Any] = {}
        self.model_feature_cols: list[str] = []
        self.feature_cols: list[str] = []
        self.classes: list[int] = []
        self.positive_class_id = 1
        self.negative_class_id = 0
        self.decision_threshold = 0.5
        self.class_names: Dict[str, str] = {}
        self._load_assets()

    def _load_assets(self) -> None:
        bundle = joblib.load(resource_path(self._paths.pipeline_pkl))
        with open(resource_path(self._paths.meta_json), "r", encoding="utf-8-sig") as f:
            self.meta = json.load(f)

        if isinstance(bundle, dict) and "model" in bundle:
            self.model = bundle["model"]
            self.model_feature_cols = list(bundle.get("feature_cols") or self.meta["feature_cols"])
        else:
            self.model = bundle
            self.model_feature_cols = list(self.meta["feature_cols"])
        if hasattr(self.model, "n_jobs"):
            self.model.n_jobs = 1

        self.classes = [int(x) for x in self.meta.get("classes", [0, 1])]
        self.positive_class_id = int(self.meta.get("positive_class_id", 1))
        self.negative_class_id = int(self.meta.get("negative_class_id", 0))
        self.decision_threshold = float(self.meta.get("decision_threshold", 0.5))
        class_names = self.meta.get("class_names") or {"0": "Нетоксично по GHS-порогу", "1": "Токсично по GHS-порогу"}
        self.class_names = {str(k): str(v) for k, v in class_names.items()}

        # DSSWorkflow checks predictor.feature_cols against the shared legacy feature table.
        # This predictor builds its own v2 features, so exposing them there would create a false warning.
        self.feature_cols = []

    def _label(self, class_id: int) -> str:
        return self.class_names.get(str(class_id), str(class_id))

    def predict(self, mol: Chem.Mol, *, features_df=None) -> Dict[str, Any]:
        xdf = build_bioactivity_feature_df(mol, self.model_feature_cols)
        y_model = int(self.model.predict(xdf)[0])

        prob_toxic = None
        class_prob_map: Dict[int, float] = {}
        if hasattr(self.model, "predict_proba"):
            proba = self.model.predict_proba(xdf)[0]
            classes = list(getattr(self.model, "classes_", range(len(proba))))
            class_prob_map = {int(cls): float(prob) for cls, prob in zip(classes, proba)}
            prob_toxic = class_prob_map.get(self.positive_class_id)
            if prob_toxic is None and len(class_prob_map) == 2:
                prob_toxic = class_prob_map.get(sorted(class_prob_map)[-1])

        if prob_toxic is None:
            decision_is_toxic = int(y_model) == self.positive_class_id
            prob_toxic = 1.0 if decision_is_toxic else 0.0
        else:
            decision_is_toxic = prob_toxic >= self.decision_threshold

        final_class = self.positive_class_id if decision_is_toxic else self.negative_class_id
        label = self._label(final_class)
        confidence_score = max(float(prob_toxic), 1.0 - float(prob_toxic))
        rule_text = _label_ru(RULE_LABELS_RU, self.meta.get("ghs_binary_rule"))
        direction = "выше" if decision_is_toxic else "ниже"
        notes_bits = [
            f"порог: {rule_text}",
            f"вероятность {direction} порога решения",
        ]

        return {
            "task": self.meta.get("target_name", "Биоактивность: бинарная классификация"),
            "value": label,
            "confidence": (
                f"P(токсичности)={float(prob_toxic):.3f}; "
                f"порог={self.decision_threshold:.3f}; "
                f"решение={'токсично' if decision_is_toxic else 'нетоксично'}"
            ),
            "prob_toxic": float(prob_toxic),
            "toxicity_threshold": self.decision_threshold,
            "toxicity_decision": bool(decision_is_toxic),
            "confidence_score": confidence_score,
            "ad_distance": None,
            "ad_threshold": None,
            "ad_ratio": None,
            "ad_score": None,
            "in_domain": None,
            "notes": "; ".join(bit for bit in notes_bits if bit),
        }


@dataclass(frozen=True)
class BioactivityRegressionPaths:
    artifact_joblib: str
    task_key: str


REGRESSION_TASK_META = {
    "ec50_aquatic_invertebrates_acute_48h": {
        "task": "Биоактивность: EC50 регрессия, водные беспозвоночные",
        "endpoint": "EC50",
        "unit": "мг/л",
        "threshold": 1.0,
        "threshold_text": "GHS acute aquatic toxic: EC50 <= 1 мг/л",
        "ad45_r2": 0.758,
        "ad45_coverage": 0.187,
        "ad55_r2": 0.847,
        "ad55_coverage": 0.052,
        "full_r2": 0.429,
    },
    "ld50_mammals_acute_oral": {
        "task": "Биоактивность: LD50 регрессия, млекопитающие перорально",
        "endpoint": "LD50 oral",
        "unit": "мг/кг",
        "threshold": 300.0,
        "threshold_text": "GHS acute oral toxic: LD50 <= 300 мг/кг",
        "ad45_r2": 0.685,
        "ad45_coverage": 0.339,
        "ad55_r2": 0.683,
        "ad55_coverage": 0.109,
        "full_r2": 0.556,
    },
}


def _regression_fp(mol: Chem.Mol):
    return _morgan_generator(2048).GetFingerprint(mol)


class BioactivityRegressionPredictor:
    """AD-gated research regression predictor for EC50/LD50.

    The numeric value is deliberately not exposed as `prob_toxic`: DSS should not
    treat regression confidence as toxic-class probability.
    """

    def __init__(self, paths: BioactivityRegressionPaths):
        self._paths = paths
        self.task_key = paths.task_key
        self.task_meta = dict(REGRESSION_TASK_META.get(self.task_key, {}))
        self.meta: Dict[str, Any] = {
            "task_key": self.task_key,
            "target_name": self.task_meta.get("task", self.task_key),
            "model_family": "universal_consensus_regression",
            "ad_min_tanimoto": 0.45,
            "ad_high_tanimoto": 0.55,
        }
        self.feature_cols: list[str] = []
        self.model_feature_cols: list[str] = []
        self._experts: list[dict[str, Any]] = []
        self._load_assets()

    def _load_assets(self) -> None:
        artifact_path = str(resource_path(self._paths.artifact_joblib))
        if artifact_path not in BIOACTIVITY_REGRESSION_ARTIFACT_CACHE:
            BIOACTIVITY_REGRESSION_ARTIFACT_CACHE[artifact_path] = joblib.load(artifact_path)
        bundle = BIOACTIVITY_REGRESSION_ARTIFACT_CACHE[artifact_path]
        seed = int(bundle.get("runtime_seed", 42))
        experts = list((bundle.get("artifacts") or {}).get((self.task_key, seed), []) or [])
        if not experts:
            raise ValueError(f"No regression experts found for task {self.task_key!r}")
        self._experts = experts
        feature_cols: list[str] = []
        for expert in experts:
            for col in expert.get("feature_cols", []) or []:
                if col not in feature_cols:
                    feature_cols.append(str(col))
            model = expert.get("estimator")
            if hasattr(model, "n_jobs"):
                model.n_jobs = 1
        self.model_feature_cols = feature_cols
        self.feature_cols = []
        metrics = (bundle.get("metrics") or {}).get(self.task_key, {})
        if metrics:
            self.meta["research_metrics"] = metrics

    def _predict_experts(self, mol: Chem.Mol) -> list[dict[str, Any]]:
        parts: list[dict[str, Any]] = []
        qfp = _regression_fp(mol)
        for expert in self._experts:
            cols = [str(c) for c in expert.get("feature_cols", []) or []]
            if not cols:
                continue
            xdf = build_bioactivity_feature_df(mol, cols)
            model = expert.get("estimator")
            try:
                pred = float(model.predict(xdf)[0])
            except Exception:
                continue
            sims: list[float] = []
            for train_fp in expert.get("train_fps", []) or []:
                try:
                    sims.append(float(DataStructs.TanimotoSimilarity(qfp, train_fp)))
                except Exception:
                    pass
            sim = max(sims) if sims else None
            sim_for_weight = 0.35 if sim is None else sim
            ad_weight = max(0.05, (sim_for_weight - 0.30) / 0.40)
            quality_weight = max(0.05, min(1.0, (float(expert.get("valid_r2") or 0.0) + 0.2) / 1.2))
            if expert.get("expert_name") == "global_all":
                ad_weight = max(ad_weight, 0.25)
            parts.append({
                "expert": str(expert.get("expert_name") or "expert"),
                "model": str(expert.get("model_name") or type(model).__name__),
                "prediction": pred,
                "max_tanimoto": sim,
                "weight": float(ad_weight * quality_weight),
                "valid_r2": float(expert.get("valid_r2")) if expert.get("valid_r2") is not None else None,
            })
        return parts

    def _trust_label(self, sim: float | None) -> tuple[str, str, bool]:
        if sim is not None and sim >= 0.55:
            return "высокая", "числу можно доверять в пределах исследовательской модели", True
        if sim is not None and sim >= 0.45:
            return "рабочая", "число применимо, но желательно сверить с аналогами", True
        if sim is not None and sim >= 0.35:
            return "низкая", "число пограничное: использовать как ориентир, не как вывод", False
        return "очень низкая", "молекула вне AD: числу доверять нельзя, нужен ручной разбор", False

    def predict(self, mol: Chem.Mol, *, features_df=None) -> Dict[str, Any]:
        parts = self._predict_experts(mol)
        if not parts:
            return {
                "task": self.task_meta.get("task", self.task_key),
                "value": "прогноз недоступен",
                "confidence": "регрессионная модель не смогла построить прогноз",
                "confidence_score": None,
                "in_domain": None,
            "notes": "Регрессионный прогноз недоступен: не удалось рассчитать значение.",
            }
        weights = np.array([max(float(p.get("weight") or 0.0), 1e-6) for p in parts], dtype=float)
        preds = np.array([float(p["prediction"]) for p in parts], dtype=float)
        log_value = float(np.average(preds, weights=weights))
        consensus_std = float(np.sqrt(np.average((preds - log_value) ** 2, weights=weights))) if len(preds) > 1 else 0.0
        value = float(10 ** log_value)
        sims = [float(p["max_tanimoto"]) for p in parts if p.get("max_tanimoto") is not None]
        best_sim = max(sims) if sims else None
        trust_label, trust_explanation, in_domain = self._trust_label(best_sim)
        ad_min = 0.45
        distance = (1.0 - best_sim) if best_sim is not None else None
        threshold = 1.0 - ad_min
        ratio = (distance / threshold) if distance is not None and threshold > 0 else None
        threshold_value = float(self.task_meta.get("threshold", 0.0) or 0.0)
        toxic_by_value = bool(threshold_value and value <= threshold_value)
        unit = self.task_meta.get("unit", "")
        endpoint = self.task_meta.get("endpoint", self.task_key)
        value_text = f"{endpoint} ≈ {value:.3g} {unit} (log10={log_value:.3f})"
        value_text += " — ниже токсикологического порога" if toxic_by_value else " — выше токсикологического порога"
        confidence = (
            f"доверие: {trust_label}; AD Tanimoto={best_sim:.3f} при пороге 0.45; "
            f"разброс consensus={consensus_std:.3f} log; экспертов={len(parts)}"
            if best_sim is not None else
            f"доверие: {trust_label}; AD similarity недоступна; разброс consensus={consensus_std:.3f} log; экспертов={len(parts)}"
        )
        threshold_side = "ниже" if toxic_by_value else "выше"
        notes = (
            f"значение {threshold_side} токсикологического порога; "
            f"доверие: {trust_label}; {trust_explanation}"
        )
        return {
            "task": self.task_meta.get("task", self.task_key),
            "value": value_text,
            "confidence": confidence,
            "confidence_score": float(best_sim) if best_sim is not None else None,
            "regression_value": value,
            "regression_log10_value": log_value,
            "regression_unit": unit,
            "regression_endpoint": endpoint,
            "regression_consensus_std_log10": consensus_std,
            "regression_toxic_by_threshold": toxic_by_value,
            "regression_threshold": threshold_value,
            "prob_toxic": None,
            "toxicity_decision": None,
            "ad_distance": round(distance, 4) if distance is not None else None,
            "ad_threshold": round(threshold, 4),
            "ad_ratio": round(ratio, 4) if ratio is not None else None,
            "ad_score": float(best_sim) if best_sim is not None else None,
            "ad_tanimoto": float(best_sim) if best_sim is not None else None,
            "ad_min_tanimoto": ad_min,
            "in_domain": bool(in_domain),
            "notes": notes,
        }
