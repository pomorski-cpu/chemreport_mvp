from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Dict, Optional

import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import Descriptors, rdFingerprintGenerator
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.neighbors import NearestNeighbors

from core.utils import resource_path


@dataclass
class ReferenceBundle:
    path: str
    X: np.ndarray
    feature_cols: list[str]
    smiles: list[str]
    labels: list[str]
    scaffolds: list[str]
    descriptor_names: list[str]
    descriptor_min: np.ndarray | None
    descriptor_max: np.ndarray | None


def _safe_float(value: Any) -> Optional[float]:
    if value is None or value == "":
        return None
    try:
        value = float(value)
    except Exception:
        return None
    if not np.isfinite(value):
        return None
    return value


def _canonical_smiles(mol: Chem.Mol) -> str:
    return Chem.MolToSmiles(mol, canonical=True) if mol is not None else ""


def _murcko(mol: Chem.Mol) -> str:
    try:
        scaffold = MurckoScaffold.GetScaffoldForMol(mol)
        if scaffold is not None and scaffold.GetNumAtoms() > 0:
            return Chem.MolToSmiles(scaffold, canonical=True)
    except Exception:
        pass
    return "acyclic_" + _canonical_smiles(mol)


@lru_cache(maxsize=4096)
def _fp_from_smiles(smiles: str):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=1024)
    return generator.GetFingerprint(mol)


def _query_fp(mol: Chem.Mol):
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=1024)
    return generator.GetFingerprint(mol)


class ApplicabilityDomainService:
    """Test-layer applicability-domain assessment for one molecule.

    The service is intentionally conservative and transparent. It uses real
    reference arrays when a model exposes them, otherwise it returns `unknown`
    with molecular flags instead of pretending that AD is available.
    """

    def __init__(self, *, k: int = 5, threshold_q: float = 0.95):
        self.k = int(k)
        self.threshold_q = float(threshold_q)
        self._ref_cache: dict[str, ReferenceBundle | None] = {}

    def evaluate_prediction(
        self,
        *,
        mol: Chem.Mol,
        task: str,
        predictor: Any,
        prediction: Dict[str, Any],
        features_df,
    ) -> Dict[str, Any]:
        task = str(task or prediction.get("task") or "model")
        flags = self._molecular_flags(mol)

        existing = self._existing_ad(task, prediction, flags)
        if existing is not None:
            return existing

        ref = self._load_reference(predictor)
        if ref is None:
            return {
                "task": task,
                "status": "unknown",
                "status_ru": "AD неизвестна",
                "ad_score": None,
                "in_domain": None,
                "reason": "Для модели нет reference-файла с обучающим пространством.",
                "method": "no_reference",
                "flags": flags,
                "nearest": [],
            }

        query_x = self._query_feature_vector(mol, predictor, ref, features_df)
        if query_x is None:
            return {
                "task": task,
                "status": "unknown",
                "status_ru": "AD неизвестна",
                "ad_score": None,
                "in_domain": None,
                "reason": "Не удалось построить признаки в пространстве reference-файла.",
                "method": "feature_mismatch",
                "reference_path": ref.path,
                "flags": flags,
                "nearest": [],
            }

        distance_info = self._distance_ad(query_x, ref)
        similarity_info = self._similarity_ad(mol, ref)
        scaffold = _murcko(mol)
        scaffold_in_ref = bool(scaffold and scaffold in set(ref.scaffolds))
        descriptor_flags = self._descriptor_flags(mol, ref)

        status, in_domain = self._combine_status(distance_info, similarity_info, scaffold_in_ref, flags, descriptor_flags)
        score = self._combined_score(distance_info, similarity_info, status)
        reason = self._reason(status, distance_info, similarity_info, scaffold_in_ref, flags, descriptor_flags)

        return {
            "task": task,
            "status": status,
            "status_ru": self._status_ru(status),
            "ad_score": score,
            "in_domain": in_domain,
            "reason": reason,
            "method": "knn_distance+tanimoto+scaffold+descriptor_range",
            "reference_path": ref.path,
            "reference_size": int(ref.X.shape[0]),
            "distance": distance_info,
            "similarity": similarity_info,
            "scaffold": scaffold,
            "scaffold_in_reference": scaffold_in_ref,
            "descriptor_flags": descriptor_flags,
            "flags": flags,
            "nearest": similarity_info.get("nearest", []),
        }

    def apply_to_prediction(self, prediction: Dict[str, Any], ad: Dict[str, Any]) -> None:
        prediction["applicability_domain"] = ad
        prediction["ad_status"] = ad.get("status")
        prediction["ad_status_ru"] = ad.get("status_ru")
        if prediction.get("ad_score") is None:
            prediction["ad_score"] = ad.get("ad_score")
        distance = ad.get("distance") or {}
        if prediction.get("ad_distance") is None:
            prediction["ad_distance"] = distance.get("distance")
        if prediction.get("ad_threshold") is None:
            prediction["ad_threshold"] = distance.get("threshold")
        if prediction.get("ad_ratio") is None:
            prediction["ad_ratio"] = distance.get("ratio")
        if prediction.get("in_domain") is None and ad.get("in_domain") is not None:
            prediction["in_domain"] = bool(ad["in_domain"])
        self._calibrate_confidence_by_ad(prediction, ad)

    def _calibrate_confidence_by_ad(self, prediction: Dict[str, Any], ad: Dict[str, Any]) -> None:
        base_score = _safe_float(prediction.get("confidence_score"))
        if "model_confidence" not in prediction:
            prediction["model_confidence"] = prediction.get("confidence")
        if "model_confidence_score" not in prediction:
            prediction["model_confidence_score"] = base_score

        status = str(ad.get("status") or "unknown")
        status_ru = ad.get("status_ru") or self._status_ru(status)
        ad_score = _safe_float(ad.get("ad_score"))
        reason = str(ad.get("reason") or "").strip()

        note_parts = [f"AD: {status_ru}"]

        if base_score is None:
            self._append_note(prediction, "; ".join(note_parts))
            return

        adjusted = base_score
        calibration = "model_probability"
        if status == "out_of_domain" or ad.get("in_domain") is False:
            adjusted = min(base_score, 0.35)
            calibration = "confidence_capped_by_out_of_domain_ad"
            note_parts.append("уверенность снижена: молекула вне области применимости")
        elif status == "borderline":
            adjusted = min(base_score, 0.60)
            calibration = "confidence_capped_by_borderline_ad"
            note_parts.append("уверенность ограничена: пограничная область применимости")
        elif status == "unknown":
            adjusted = min(base_score, 0.55)
            calibration = "confidence_capped_by_unknown_ad"
            note_parts.append("уверенность ограничена: AD недоступна")
        elif status == "in_domain" and ad_score is not None:
            adjusted = (0.75 * base_score) + (0.25 * ad_score)
            calibration = "model_probability_weighted_with_ad_score"

        adjusted = max(0.0, min(1.0, float(adjusted)))
        prediction["confidence_score"] = round(adjusted, 3)
        prediction["confidence"] = self._confidence_label(adjusted)
        prediction["confidence_calibration"] = calibration
        prediction["ad_adjusted_confidence"] = True
        self._append_note(prediction, "; ".join(note_parts))

    @staticmethod
    def _confidence_label(score: float) -> str:
        if score >= 0.75:
            return "Высокая"
        if score >= 0.55:
            return "Средняя"
        return "Низкая"

    @staticmethod
    def _append_note(prediction: Dict[str, Any], note: str) -> None:
        note = str(note or "").strip()
        if not note:
            return
        existing = str(prediction.get("notes") or "").strip()
        if note in existing:
            return
        prediction["notes"] = (existing + "; " if existing else "") + note

    def summarize(self, results: list[Dict[str, Any]]) -> Dict[str, Any]:
        if not results:
            return {
                "overall_status": "unknown",
                "overall_status_ru": "AD неизвестна",
                "summary_ru": "Оценка области применимости недоступна.",
                "items": [],
            }
        rank = {"out_of_domain": 3, "borderline": 2, "unknown": 1, "in_domain": 0}
        worst = max(results, key=lambda item: rank.get(item.get("status"), 1))
        known_scores = [_safe_float(item.get("ad_score")) for item in results]
        known_scores = [score for score in known_scores if score is not None]
        mean_score = round(float(np.mean(known_scores)), 3) if known_scores else None
        counts: dict[str, int] = {}
        for item in results:
            counts[item.get("status", "unknown")] = counts.get(item.get("status", "unknown"), 0) + 1
        summary = (
            f"Сводная AD-оценка: {self._status_ru(worst.get('status'))}. "
            f"Моделей: {len(results)}; внутри AD={counts.get('in_domain', 0)}, "
            f"погранично={counts.get('borderline', 0)}, вне AD={counts.get('out_of_domain', 0)}, "
            f"неизвестно={counts.get('unknown', 0)}."
        )
        if mean_score is not None:
            summary += f" Средний AD score={mean_score:.2f}."
        return {
            "overall_status": worst.get("status", "unknown"),
            "overall_status_ru": self._status_ru(worst.get("status")),
            "mean_ad_score": mean_score,
            "counts": counts,
            "summary_ru": summary,
            "items": results,
        }

    def _existing_ad(self, task: str, prediction: Dict[str, Any], flags: list[Dict[str, Any]]) -> Dict[str, Any] | None:
        if prediction.get("in_domain") is None and prediction.get("ad_score") is None:
            return None
        in_domain = prediction.get("in_domain")
        ratio = _safe_float(prediction.get("ad_ratio"))
        score = _safe_float(prediction.get("ad_score"))
        if in_domain is False:
            status = "out_of_domain"
        elif ratio is not None and ratio > 0.85:
            status = "borderline"
        elif in_domain is True:
            status = "in_domain"
        else:
            status = "unknown"
        return {
            "task": task,
            "status": status,
            "status_ru": self._status_ru(status),
            "ad_score": score,
            "in_domain": in_domain,
            "reason": "AD рассчитана внутри модели.",
            "method": "model_native_ad",
            "distance": {
                "distance": _safe_float(prediction.get("ad_distance")),
                "threshold": _safe_float(prediction.get("ad_threshold")),
                "ratio": ratio,
            },
            "flags": flags,
            "nearest": [],
        }

    def _load_reference(self, predictor: Any) -> ReferenceBundle | None:
        meta = getattr(predictor, "meta", {}) or {}
        ref_file = meta.get("reference_file")
        if not ref_file:
            ad_meta = meta.get("ad") if isinstance(meta.get("ad"), dict) else {}
            ref_file = ad_meta.get("reference_file") or ad_meta.get("file")
        if not ref_file:
            task_key = str(meta.get("task_key") or "")
            ref_file = {
                "ec50_aquatic_invertebrates_acute_48h": "bioactivity_ec50_aquatic_binary_v2_rf_nofp_ref.npz",
                "lc50_fish_acute_96h": "bioactivity_lc50_fish_binary_v2_rf_nofp_ref.npz",
                "ld50_mammals_acute_oral": "bioactivity_ld50_mammals_oral_binary_v2_rf_nofp_ref.npz",
            }.get(task_key)
        if not ref_file:
            return None
        ref_path = resource_path(ref_file if str(ref_file).startswith("models/") else f"models/{ref_file}")
        cache_key = str(ref_path)
        if cache_key in self._ref_cache:
            return self._ref_cache[cache_key]
        if not ref_path.exists():
            self._ref_cache[cache_key] = None
            return None
        try:
            z = np.load(ref_path, allow_pickle=True)
            X_key = "X_ref" if "X_ref" in z.files else "X_ref_scaled" if "X_ref_scaled" in z.files else None
            if X_key is None:
                self._ref_cache[cache_key] = None
                return None
            feature_cols = [str(x) for x in z["feature_cols"].tolist()] if "feature_cols" in z.files else []
            smiles = [str(x) for x in z["smiles"].tolist()] if "smiles" in z.files else []
            labels = [str(x) for x in z["labels"].tolist()] if "labels" in z.files else []
            scaffolds = [str(x) for x in z["scaffolds"].tolist()] if "scaffolds" in z.files else []
            descriptor_names = [str(x) for x in z["descriptor_names"].tolist()] if "descriptor_names" in z.files else []
            descriptor_min = z["descriptor_min"].astype(float) if "descriptor_min" in z.files else None
            descriptor_max = z["descriptor_max"].astype(float) if "descriptor_max" in z.files else None
            ref = ReferenceBundle(
                path=str(ref_path),
                X=z[X_key].astype(np.float32),
                feature_cols=feature_cols,
                smiles=smiles,
                labels=labels,
                scaffolds=scaffolds,
                descriptor_names=descriptor_names,
                descriptor_min=descriptor_min,
                descriptor_max=descriptor_max,
            )
        except Exception:
            ref = None
        self._ref_cache[cache_key] = ref
        return ref

    def _query_feature_vector(self, mol: Chem.Mol, predictor: Any, ref: ReferenceBundle, features_df):
        try:
            if hasattr(predictor, "model_feature_cols"):
                from core.bioactivity_predictor import build_bioactivity_feature_df

                xdf = build_bioactivity_feature_df(mol, ref.feature_cols)
            else:
                xdf = features_df.reindex(columns=ref.feature_cols, fill_value=0.0)
            arr = xdf.to_numpy(dtype=np.float32)
            if arr.shape[1] != ref.X.shape[1]:
                return None
            return arr
        except Exception:
            return None

    def _distance_ad(self, query_x: np.ndarray, ref: ReferenceBundle) -> Dict[str, Any]:
        X = ref.X
        k = min(max(1, self.k), max(1, len(X) - 1))
        nn = NearestNeighbors(n_neighbors=min(k + 1, len(X)), metric="euclidean")
        nn.fit(X)
        d_train, _ = nn.kneighbors(X, return_distance=True)
        if d_train.shape[1] > 1:
            train_dc = d_train[:, 1:].mean(axis=1)
        else:
            train_dc = d_train[:, :1].mean(axis=1)
        threshold = float(np.quantile(train_dc, self.threshold_q))
        d_query, idx = nn.kneighbors(query_x, n_neighbors=min(k, len(X)), return_distance=True)
        distance = float(d_query.mean())
        ratio = float(distance / threshold) if threshold > 0 else None
        return {
            "distance": round(distance, 4),
            "threshold": round(threshold, 4),
            "ratio": round(ratio, 4) if ratio is not None else None,
            "k": int(k),
            "threshold_q": self.threshold_q,
            "neighbor_indices": [int(i) for i in idx[0].tolist()],
        }

    def _similarity_ad(self, mol: Chem.Mol, ref: ReferenceBundle) -> Dict[str, Any]:
        if not ref.smiles:
            return {"top_similarity": None, "nearest": []}
        qfp = _query_fp(mol)
        rows = []
        for i, smiles in enumerate(ref.smiles):
            fp = _fp_from_smiles(smiles)
            if fp is None:
                continue
            sim = float(DataStructs.TanimotoSimilarity(qfp, fp))
            rows.append((sim, i, smiles))
        rows.sort(reverse=True, key=lambda x: x[0])
        nearest = []
        for sim, i, smiles in rows[:5]:
            nearest.append({
                "rank": len(nearest) + 1,
                "similarity": round(sim, 3),
                "smiles": smiles,
                "label": ref.labels[i] if i < len(ref.labels) else "",
                "scaffold": ref.scaffolds[i] if i < len(ref.scaffolds) else "",
            })
        return {"top_similarity": nearest[0]["similarity"] if nearest else None, "nearest": nearest}

    def _descriptor_flags(self, mol: Chem.Mol, ref: ReferenceBundle) -> list[Dict[str, Any]]:
        if ref.descriptor_min is None or ref.descriptor_max is None or not ref.descriptor_names:
            return []
        values = self._descriptor_values(mol)
        flags = []
        for i, name in enumerate(ref.descriptor_names):
            value = values.get(name)
            if value is None or i >= len(ref.descriptor_min) or i >= len(ref.descriptor_max):
                continue
            lo = float(ref.descriptor_min[i])
            hi = float(ref.descriptor_max[i])
            pad = max(1e-9, 0.05 * (hi - lo))
            if value < lo - pad or value > hi + pad:
                flags.append({
                    "code": "descriptor_out_of_range",
                    "descriptor": name,
                    "value": round(float(value), 3),
                    "min": round(lo, 3),
                    "max": round(hi, 3),
                    "message": f"{name}={value:.2f} вне диапазона train [{lo:.2f}; {hi:.2f}]",
                })
        return flags

    def _descriptor_values(self, mol: Chem.Mol) -> Dict[str, float]:
        return {
            "MolWt": float(Descriptors.MolWt(mol)),
            "MolLogP": float(Descriptors.MolLogP(mol)),
            "TPSA": float(Descriptors.TPSA(mol)),
            "HeavyAtomCount": float(Descriptors.HeavyAtomCount(mol)),
            "NumHDonors": float(Descriptors.NumHDonors(mol)),
            "NumHAcceptors": float(Descriptors.NumHAcceptors(mol)),
            "NumRotatableBonds": float(Descriptors.NumRotatableBonds(mol)),
            "RingCount": float(Descriptors.RingCount(mol)),
        }

    def _molecular_flags(self, mol: Chem.Mol) -> list[Dict[str, Any]]:
        flags = []
        smiles = _canonical_smiles(mol)
        if "." in smiles:
            flags.append({"code": "multi_fragment", "level": "medium", "message": "SMILES содержит несколько фрагментов."})
        allowed = {"H", "B", "C", "N", "O", "F", "P", "S", "Cl", "Br", "I", "Si"}
        rare = sorted({atom.GetSymbol() for atom in mol.GetAtoms() if atom.GetSymbol() not in allowed})
        if rare:
            flags.append({"code": "rare_elements", "level": "high", "message": "Редкие элементы: " + ", ".join(rare)})
        if mol.GetNumHeavyAtoms() > 80:
            flags.append({"code": "large_molecule", "level": "medium", "message": "Очень крупная молекула для текущих QSAR-моделей."})
        return flags

    def _combine_status(self, distance_info, similarity_info, scaffold_in_ref, flags, descriptor_flags) -> tuple[str, Optional[bool]]:
        ratio = _safe_float(distance_info.get("ratio"))
        top_sim = _safe_float(similarity_info.get("top_similarity"))
        high_flags = any(flag.get("level") == "high" for flag in flags)
        if ratio is not None and ratio > 1.25:
            return "out_of_domain", False
        if top_sim is not None and top_sim < 0.30:
            return "out_of_domain", False
        if high_flags or len(descriptor_flags) >= 3:
            return "out_of_domain", False
        if ratio is not None and ratio > 1.0:
            return "borderline", True
        if top_sim is not None and top_sim < 0.45:
            return "borderline", True
        if not scaffold_in_ref:
            return "borderline", True
        if descriptor_flags:
            return "borderline", True
        return "in_domain", True

    def _combined_score(self, distance_info, similarity_info, status: str) -> Optional[float]:
        ratio = _safe_float(distance_info.get("ratio"))
        d_score = None if ratio is None else max(0.0, min(1.0, 1.0 - (ratio / 1.25)))
        sim = _safe_float(similarity_info.get("top_similarity"))
        values = [v for v in [d_score, sim] if v is not None]
        if not values:
            return None
        score = float(np.mean(values))
        if status == "borderline":
            score = min(score, 0.55)
        elif status == "out_of_domain":
            score = min(score, 0.25)
        return round(max(0.0, min(1.0, score)), 3)

    def _reason(self, status, distance_info, similarity_info, scaffold_in_ref, flags, descriptor_flags) -> str:
        parts = []
        ratio = distance_info.get("ratio")
        if ratio is not None:
            parts.append(f"kNN ratio={ratio:.2f}")
        top_sim = similarity_info.get("top_similarity")
        if top_sim is not None:
            parts.append(f"max Tanimoto={top_sim:.2f}")
        parts.append("scaffold есть в train" if scaffold_in_ref else "новый scaffold относительно train")
        if descriptor_flags:
            parts.append(f"descriptor flags={len(descriptor_flags)}")
        if flags:
            parts.append("; ".join(flag.get("message", flag.get("code", "")) for flag in flags))
        if status == "in_domain":
            return "Молекула похожа на обучающую область: " + ", ".join(parts)
        if status == "borderline":
            return "Пограничная применимость: " + ", ".join(parts)
        if status == "out_of_domain":
            return "Вне области применимости: " + ", ".join(parts)
        return "AD не удалось оценить."

    def _status_ru(self, status: Any) -> str:
        return {
            "in_domain": "в области применимости",
            "borderline": "пограничная зона",
            "out_of_domain": "вне области применимости",
            "unknown": "AD неизвестна",
            None: "AD неизвестна",
        }.get(status, str(status))
