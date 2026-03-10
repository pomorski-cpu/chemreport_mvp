import json
import joblib
import torch
import numpy as np
from dataclasses import dataclass
from core.utils import resource_path


@dataclass
class ModelsBundle:
    svr_pipeline: object
    svr_ref: np.ndarray
    mlp_scaler: object
    mlp_state: dict
    mlp_meta: dict
    svr_meta: dict


def load_models(device: str = "cpu") -> ModelsBundle:
    return ModelsBundle(
        svr_pipeline=joblib.load(
            resource_path("models/svr_logp_pipeline.pkl")
        ),
        svr_ref=np.load(
            resource_path("models/svr_logp_ref.npz")
        ),
        mlp_scaler=joblib.load(
            resource_path("models/mlp_regression_scaler.joblib")
        ),
        mlp_state=torch.load(
            resource_path("models/mlp_regression_state.pt"),
            map_location=device,
        ),
        mlp_meta=json.load(
            open(resource_path("models/mlp_regression_meta.json"), encoding="utf-8")
        ),
        svr_meta=json.load(
            open(resource_path("models/svr_logp_pipeline_meta.json"), encoding="utf-8")
        ),
    )
