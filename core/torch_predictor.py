from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from rdkit import Chem
from sklearn.neighbors import NearestNeighbors

from core.featurizer_rdkit_inchi import build_feature_df
from core.utils import resource_path


class TorchPredictorError(RuntimeError):
    pass


class MLP(nn.Module):
    def __init__(self, in_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        return self.net(x)


def _extract_state_dict(obj: Any) -> Optional[Dict[str, torch.Tensor]]:
    if isinstance(obj, dict):
        if "state_dict" in obj and isinstance(obj["state_dict"], dict):
            return obj["state_dict"]
        if "model_state_dict" in obj and isinstance(obj["model_state_dict"], dict):
            return obj["model_state_dict"]
        if obj and all(isinstance(v, torch.Tensor) for v in obj.values()):
            return obj
    return None


class TorchPredictor:
    def __init__(
        self,
        models_dir: str = "models",
        *,
        model_file: str = "mlp_regression_state.pt",
        meta_file: str = "mlp_regression_meta.json",
        scaler_file: str = "mlp_regression_scaler.joblib",
        ref_file: str = "mlp_logp_ref.npz",
        device: str = "cpu",
        round_value: int = 3,
        k_ad: int = 5,
        threshold_q: float = 0.95,
        calib_sample: int = 2000,
        ad_metric: str = "euclidean",
    ):
        self.models_dir = models_dir
        self.device = torch.device(device)
        self.round_value = int(round_value)
        self.k_ad = int(k_ad)
        self.threshold_q = float(threshold_q)
        self.calib_sample = int(calib_sample)
        self.ad_metric = ad_metric

        model_rel = f"{models_dir}/{model_file}"
        meta_rel = f"{models_dir}/{meta_file}"
        scaler_rel = f"{models_dir}/{scaler_file}"
        ref_rel = f"{models_dir}/{ref_file}"

        self.model_path: Path = resource_path(model_rel)
        self.meta_path: Path = resource_path(meta_rel)
        self.scaler_path: Path = resource_path(scaler_rel)
        self.ref_path: Path = resource_path(ref_rel)

        if not self.model_path.exists():
            raise TorchPredictorError(f"Файл модели не найден: {self.model_path}")
        if not self.meta_path.exists():
            raise TorchPredictorError(f"Файл метаданных не найден: {self.meta_path}")

        with open(self.meta_path, "r", encoding="utf-8") as f:
            self.meta = json.load(f)

        self.feature_cols: List[str] = self.meta["feature_cols"]
        self.task = self.meta.get("target_name", "LogP")
        self.name = self.meta.get("name", "MLP (PyTorch)")

        self.scaler = None
        if self.scaler_path.exists():
            import joblib

            self.scaler = joblib.load(self.scaler_path)
        else:
            raise TorchPredictorError(
                f"Scaler не найден: {self.scaler_path}\n"
                "Модель обучалась со StandardScaler. Сохраните scaler, использованный для финальной модели."
            )

        obj = torch.load(self.model_path, map_location=self.device)
        state = _extract_state_dict(obj)
        if state is None:
            raise TorchPredictorError(
                "Этот файл не является state_dict/checkpoint. "
                "Если вы сохраняли state_dict, проверьте код сохранения."
            )

        in_dim = len(self.feature_cols)
        self.model = MLP(in_dim=in_dim)
        self.model.load_state_dict(state, strict=True)
        self.model.to(self.device)
        self.model.eval()

        self.X_ref_scaled: Optional[np.ndarray] = None
        self._nn: Optional[NearestNeighbors] = None
        self.ad_threshold: Optional[float] = None

        self._load_ad_reference()
        self._calibrate_ad()

    def _load_ad_reference(self) -> None:
        if not self.ref_path.exists():
            self.X_ref_scaled = None
            return

        ref = np.load(self.ref_path, allow_pickle=True)
        X_ref: Optional[np.ndarray] = None
        if "X_ref_scaled" in ref:
            X_ref = ref["X_ref_scaled"].astype(np.float32)
        elif "X_ref" in ref:
            X_ref = ref["X_ref"].astype(np.float32)

        if X_ref is None:
            self.X_ref_scaled = None
            return

        # AD should be computed in the same feature space as the model.
        # If an old/incompatible reference is provided, disable AD gracefully.
        if X_ref.ndim != 2 or X_ref.shape[1] != len(self.feature_cols):
            self.X_ref_scaled = None
            return

        self.X_ref_scaled = X_ref

    def _calibrate_ad(self) -> None:
        X = self.X_ref_scaled
        if X is None or len(X) < max(20, self.k_ad + 2):
            self._nn = None
            self.ad_threshold = None
            return

        self._nn = NearestNeighbors(
            n_neighbors=self.k_ad + 1,
            metric=self.ad_metric,
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
        Dc = dists[:, 1:].mean(axis=1).astype(np.float32)
        self.ad_threshold = float(np.quantile(Dc, self.threshold_q))

    def _ad_metrics(
        self, x_scaled: np.ndarray
    ) -> Tuple[Optional[float], Optional[bool], Optional[float], Optional[float]]:
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

    def _make_x(self, mol: Chem.Mol, features_df=None) -> np.ndarray:
        Xdf = features_df if features_df is not None else build_feature_df(mol)
        Xdf = Xdf.reindex(columns=self.feature_cols, fill_value=0.0)
        return self.scaler.transform(Xdf).astype(np.float32)

    def predict(self, mol: Chem.Mol, *, features_df=None):
        x = self._make_x(mol, features_df=features_df)

        with torch.no_grad():
            y = (
                self.model(torch.tensor(x, dtype=torch.float32, device=self.device))
                .cpu()
                .numpy()[0, 0]
            )

        Dc, in_domain, ratio, conf_score = self._ad_metrics(x)
        label = self._confidence_label(conf_score, in_domain)
        conf_txt = ""
        if label and conf_score is not None and Dc is not None and self.ad_threshold is not None:
            conf_txt = f"{label} (оценка={conf_score:.2f}, Dc={Dc:.2f}, порог={self.ad_threshold:.2f})"

        return {
            "task": self.task,
            "value": float(y),
            "confidence": conf_txt,
            "ad_distance": Dc,
            "ad_threshold": self.ad_threshold,
            "ad_ratio": ratio,
            "ad_score": conf_score,
            "confidence_score": conf_score,
            "in_domain": in_domain,
            "source": "torch",
            "notes": self.name + ("; вне области применимости" if in_domain is False else ""),
        }
