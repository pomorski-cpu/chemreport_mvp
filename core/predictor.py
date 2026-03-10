# core/predictor.py
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import joblib
import numpy as np
from rdkit import Chem
from sklearn.neighbors import NearestNeighbors

from core.featurizer_rdkit_inchi import build_feature_df
from core.utils import resource_path


@dataclass(frozen=True)
class SVRPaths:
    pipeline_pkl: str = "models/svr_logp_pipeline.pkl"
    meta_json: str = "models/svr_logp_pipeline_meta.json"
    ref_npz: str = "models/svr_logp_ref.npz"


class SVRPredictor:
    """
    SVR pipeline + Applicability Domain (kNN average distance), inspired by qsar_ad:
      - Dc(x) = mean distance to k nearest neighbors in training space
      - ad_threshold computed from training Dc distribution (e.g., 95th percentile)
      - in_domain = Dc <= ad_threshold
    """

    def __init__(
        self,
        models_dir: str = "models",  # оставлено для совместимости, фактически не нужно
        *,
        k_ad: int = 5,
        threshold_q: float = 0.95,
        calib_sample: int = 2000,
        metric: str = "euclidean",
        paths: SVRPaths | None = None,
    ):
        self.models_dir = models_dir
        self.k_ad = int(k_ad)
        self.threshold_q = float(threshold_q)
        self.calib_sample = int(calib_sample)
        self.metric = metric

        self._paths = paths or SVRPaths()

        # loaded assets
        self.model = None  # sklearn Pipeline
        self.meta: Dict[str, Any] = {}
        self.feature_cols: list[str] = []
        self.X_ref_scaled: Optional[np.ndarray] = None

        # AD
        self._nn: Optional[NearestNeighbors] = None
        self.ad_threshold: Optional[float] = None

        self._load_assets()
        self._calibrate_ad()

    # ---------------- IO ----------------

    def _load_assets(self) -> None:
        # --- load model ---
        model_path = resource_path(self._paths.pipeline_pkl)
        self.model = joblib.load(model_path)

        # --- load meta ---
        meta_path = resource_path(self._paths.meta_json)
        with open(meta_path, "r", encoding="utf-8") as f:
            self.meta = json.load(f)
        self.feature_cols = list(self.meta["feature_cols"])

        # --- load training reference (already scaled) ---
        ref_path = resource_path(self._paths.ref_npz)
        if ref_path.exists():
            ref = np.load(ref_path)
            # ожидаем ключ X_ref_scaled
            self.X_ref_scaled = ref["X_ref_scaled"].astype(np.float32)
        else:
            self.X_ref_scaled = None

    # ---------------- AD ----------------

    def _calibrate_ad(self) -> None:
        """
        Fit NearestNeighbors on X_ref_scaled and compute:
          - Dc_train[i] = mean distance to k nearest neighbors of train point i (excluding itself)
          - ad_threshold = quantile(Dc_train, threshold_q)
        """
        X = self.X_ref_scaled
        if X is None or len(X) < max(20, self.k_ad + 2):
            self._nn = None
            self.ad_threshold = None
            return

        self._nn = NearestNeighbors(
            n_neighbors=self.k_ad + 1,  # +1 because nearest is itself (distance 0)
            metric=self.metric,
            algorithm="auto",
        )
        self._nn.fit(X)

        n = len(X)
        m = min(n, self.calib_sample)
        if m < n:
            rng = np.random.RandomState(42)
            idx = rng.choice(n, size=m, replace=False)
            X_cal = X[idx]
        else:
            X_cal = X

        dists, _ = self._nn.kneighbors(X_cal, n_neighbors=self.k_ad + 1, return_distance=True)
        Dc = dists[:, 1:].mean(axis=1).astype(np.float32)  # drop self

        self.ad_threshold = float(np.quantile(Dc, self.threshold_q))

    def _ad_metrics(
        self, x_scaled: np.ndarray
    ) -> Tuple[Optional[float], Optional[bool], Optional[float], Optional[float]]:
        """
        Returns:
          Dc (avg kNN distance),
          in_domain (Dc <= threshold),
          ratio (Dc / threshold),
          confidence_score in [0..1]
        """
        if self._nn is None or self.ad_threshold is None:
            return None, None, None, None

        dists, _ = self._nn.kneighbors(x_scaled, n_neighbors=self.k_ad, return_distance=True)
        Dc = float(dists.mean())

        th = float(self.ad_threshold)
        ratio = (Dc / th) if th > 0 else None
        in_domain = (Dc <= th) if ratio is not None else None

        conf = (1.0 - min(1.0, Dc / th)) if th > 0 else None
        return Dc, in_domain, ratio, conf

    def _confidence_label(self, conf_score: Optional[float], in_domain: Optional[bool]) -> str:
        if conf_score is None or in_domain is None:
            return ""
        if in_domain and conf_score >= 0.50:
            return "Высокая"
        if in_domain and conf_score >= 0.20:
            return "Средняя"
        return "Низкая"

    # ---------------- Predict ----------------

    def predict(self, mol: Chem.Mol, *, features_df=None) -> Dict[str, Any]:
        if self.model is None:
            raise RuntimeError("SVRPredictor: model is not loaded")

        # 1) features
        Xdf = features_df if features_df is not None else build_feature_df(mol)
        Xdf = Xdf.reindex(columns=self.feature_cols, fill_value=0.0)

        # 2) regression
        y = self.model.predict(Xdf)

        # 3) AD: use SAME scaler as training
        scaler = getattr(self.model, "named_steps", {}).get("scaler", None)
        if scaler is None:
            x_scaled = Xdf.to_numpy(dtype=np.float32)
        else:
            x_scaled = scaler.transform(Xdf).astype(np.float32)

        Dc, in_domain, ratio, conf_score = self._ad_metrics(x_scaled)
        label = self._confidence_label(conf_score, in_domain)

        conf_txt = ""
        if label and conf_score is not None and Dc is not None and self.ad_threshold is not None:
            conf_txt = f"{label} (оценка={conf_score:.2f}, Dc={Dc:.2f}, порог={self.ad_threshold:.2f})"

        notes = self.meta.get("name", "SVR")
        if in_domain is False:
            notes = f"{notes}; вне области применимости"

        return {
            "task": self.meta.get("target_name", "SVR"),
            "value": float(y[0]),
            "confidence": conf_txt,
            "ad_distance": Dc,
            "ad_threshold": self.ad_threshold,
            "ad_ratio": ratio,
            "ad_score": conf_score,
            "confidence_score": conf_score,
            "in_domain": in_domain,
            "notes": notes,
        }
