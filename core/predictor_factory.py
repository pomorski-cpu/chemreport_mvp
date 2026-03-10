from __future__ import annotations
from typing import Any, Dict

from core.model_registry import ModelRegistry
from core.predictor import SVRPredictor, SVRPaths
from core.tox_predictor import ToxPredictor, ToxPaths
from core.utils import resource_path


class PredictorFactory:
    def __init__(self, registry_path: str = "models/registry.json"):
        self.registry = ModelRegistry(registry_path)

    def create(self, task_key: str):
        cfg: Dict[str, Any] = self.registry.resolve(task_key)
        t = cfg.pop("type")
        cfg.pop("version", None)
        models_dir = cfg.get("models_dir", "models")

        if t == "sklearn_svr":
            # Support both schemas:
            # 1) pipeline_pkl/meta_json/ref_npz
            # 2) legacy model_file/meta_file (fallback to known defaults)
            pipeline_pkl = cfg.get("pipeline_pkl")
            meta_json = cfg.get("meta_json")
            ref_npz = cfg.get("ref_npz")

            if not pipeline_pkl or pipeline_pkl.endswith("svr_model.pkl"):
                pipeline_pkl = f"{models_dir}/svr_logp_pipeline.pkl"
            if not meta_json or meta_json.endswith("svr_meta.json"):
                meta_json = f"{models_dir}/svr_logp_pipeline_meta.json"
            if not ref_npz:
                ref_npz = f"{models_dir}/svr_logp_ref.npz"

            return SVRPredictor(
                models_dir=models_dir,
                paths=SVRPaths(
                    pipeline_pkl=pipeline_pkl,
                    meta_json=meta_json,
                    ref_npz=ref_npz,
                ),
            )

        if t == "torch_state":
            from core.torch_predictor import TorchPredictor
            return TorchPredictor(**cfg)

        if t == "sklearn_pipeline":
            pipeline_pkl = cfg.get("pipeline_pkl", "randomforest_tox_pipeline.pkl")
            meta_json = cfg.get("meta_json", "randomforest_tox_pipeline_meta.json")
            pipeline_path = resource_path(f"{models_dir}/{pipeline_pkl}")
            meta_path = resource_path(f"{models_dir}/{meta_json}")
            if not pipeline_path.exists():
                if resource_path(f"{models_dir}/randomforest_tox_pipeline.pkl").exists():
                    pipeline_pkl = "randomforest_tox_pipeline.pkl"
                elif resource_path(f"{models_dir}/knn_tox_pipeline.pkl").exists():
                    pipeline_pkl = "knn_tox_pipeline.pkl"
                else:
                    pipeline_pkl = "bagging_tox_pipeline.pkl"
            if not meta_path.exists():
                if resource_path(f"{models_dir}/randomforest_tox_pipeline_meta.json").exists():
                    meta_json = "randomforest_tox_pipeline_meta.json"
                elif resource_path(f"{models_dir}/knn_tox_pipeline_meta.json").exists():
                    meta_json = "knn_tox_pipeline_meta.json"
                else:
                    meta_json = "bagging_tox_pipeline_meta.json"
            return ToxPredictor(
                paths=ToxPaths(
                    pipeline_pkl=f"{models_dir}/{pipeline_pkl}",
                    meta_json=f"{models_dir}/{meta_json}",
                )
            )

        raise ValueError(f"Unknown model type '{t}' for task '{task_key}'")
