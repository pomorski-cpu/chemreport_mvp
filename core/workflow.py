from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

from rdkit import Chem

from core.descriptors import compute_basic_descriptors
from core.featurizer_rdkit_inchi import build_feature_df
from core.logging_utils import configure_logging, get_logger
from core.profiling import profile_molecule
from core.reliability import estimate_reliability
from core.report import build_report_payload

configure_logging()
logger = get_logger("workflow")


@dataclass(frozen=True)
class PredictorSpec:
    task: str
    predictor: Any
    coverage_name: str


class DSSWorkflow:
    def __init__(
        self,
        predictor_specs: List[PredictorSpec],
        *,
        decision_support: Any | None = None,
        read_across: Any | None = None,
    ):
        self.predictor_specs = predictor_specs
        self.decision_support = decision_support
        self.read_across = read_across

    def analyze_molecule(
        self,
        mol: Chem.Mol,
        *,
        meta: Dict[str, Any] | None = None,
        warnings: List[str] | None = None,
        svg: str | None = None,
    ) -> Dict[str, Any]:
        meta = meta or {}
        warnings_out = list(warnings or [])
        query_smiles = str(meta.get("smiles") or Chem.MolToSmiles(mol, canonical=True))
        logger.info("Workflow started for molecule: %s", query_smiles)

        descriptors = compute_basic_descriptors(mol)
        features_df = build_feature_df(mol)
        logger.info(
            "Features built for %s: descriptor_count=%s feature_columns=%s",
            query_smiles,
            len(descriptors),
            len(features_df.columns),
        )
        predictions = self._predict_all(mol, features_df, warnings_out)
        profile = profile_molecule(mol)

        if "." in str(meta.get("smiles", "")):
            warnings_out.append("Обнаружено несколько фрагментов; интерпретация результатов ограничена.")

        analogues: list[dict[str, Any]] = []
        category = {
            "type": "analogue_data_unavailable",
            "members": [],
            "consistency_score": 0.0,
            "summary_ru": "Категория не сформирована: данные по аналогам недоступны в текущем MVP.",
        }
        read_across_payload: dict[str, Any] = {}
        if self.read_across is not None:
            logger.info("Running read-across for molecule: %s", query_smiles)
            read_across_result = self.read_across.analyze(mol, meta=meta)
            read_across_payload = dict(read_across_result)
            analogues = list(read_across_result.get("analogues", []) or [])
            category = dict(read_across_result.get("category", {}) or category)
            warnings_out.extend(read_across_result.get("warnings", []) or [])
            logger.info(
                "Read-across completed for %s: targets=%s primary_analogues=%s warnings=%s",
                query_smiles,
                len(read_across_payload.get("targets", {}) or {}),
                len(analogues),
                len(read_across_result.get("warnings", []) or []),
            )

        reliability = estimate_reliability(
            predictions=predictions,
            analogues=analogues,
            category=category,
            warnings=warnings_out,
        )

        decision = {}
        if self.decision_support is not None:
            decision = self.decision_support.evaluate(
                meta=meta,
                descriptors=descriptors,
                predictions=predictions,
                warnings=warnings_out,
            )

        logger.info(
            "Workflow finished for %s: predictions=%s warnings=%s profile_lines=%s reliability=%s",
            query_smiles,
            len(predictions),
            len(warnings_out),
            len(profile.get("summary_ru", []) or []),
            reliability.get("final_label", ""),
        )

        payload = build_report_payload(
            meta=meta,
            descriptors=descriptors,
            predictions=predictions,
            warnings=warnings_out,
            decision=decision,
            profile=profile,
            analogues=analogues,
            category=category,
            read_across=read_across_payload,
            reliability=reliability,
            svg=svg,
        )

        return {
            "meta": meta,
            "descriptors": descriptors,
            "predictions": predictions,
            "warnings": warnings_out,
            "profile": profile,
            "analogues": analogues,
            "category": category,
            "read_across": read_across_payload,
            "reliability": reliability,
            "decision": decision,
            "payload": payload,
            "features_df": features_df,
        }

    def _predict_all(self, mol: Chem.Mol, features_df, warnings_out: list[str]) -> list[dict[str, Any]]:
        predictions: list[dict[str, Any]] = []
        for spec in self.predictor_specs:
            logger.info("Running predictor %s (%s)", spec.task, spec.coverage_name)
            coverage_warning = self._feature_coverage_warning(spec.coverage_name, spec.predictor, features_df)
            if coverage_warning:
                warnings_out.append(coverage_warning)
                logger.warning("%s", coverage_warning)

            try:
                pred = dict(spec.predictor.predict(mol, features_df=features_df) or {})
                pred["task"] = spec.task
                logger.info(
                    "Predictor %s finished: value=%s confidence=%s in_domain=%s",
                    spec.task,
                    pred.get("value"),
                    pred.get("confidence", ""),
                    pred.get("in_domain"),
                )
            except Exception as exc:
                pred = self._prediction_error(spec.task, spec.coverage_name, exc)

            if pred.get("in_domain") is False:
                warnings_out.append(
                    f"Для задачи «{spec.task}» результат находится вне области применимости."
                )

            predictions.append(pred)

        return predictions

    def _feature_coverage_warning(self, predictor_name: str, predictor: Any, features_df) -> str | None:
        expected = list(getattr(predictor, "feature_cols", []) or [])
        if not expected:
            return None

        missing = [column for column in expected if column not in features_df.columns]
        logger.info(
            "Feature coverage for %s: missing=%s/%s%s",
            predictor_name,
            len(missing),
            len(expected),
            f" sample={missing[:5]}" if missing else "",
        )
        if len(missing) < max(4, int(0.2 * len(expected))):
            return None

        return (
            f"Для модели {predictor_name} покрытие признаков снижено: "
            f"отсутствует {len(missing)} из {len(expected)} ожидаемых признаков."
        )

    def _prediction_error(self, task: str, predictor_name: str, exc: Exception) -> dict[str, Any]:
        logger.exception("Prediction failed for %s (%s)", task, predictor_name)
        return {
            "task": task,
            "value": "",
            "confidence": "",
            "confidence_score": None,
            "ad_distance": None,
            "ad_threshold": None,
            "ad_ratio": None,
            "ad_score": None,
            "in_domain": None,
            "notes": f"{predictor_name}: ошибка прогноза ({type(exc).__name__}: {exc})",
        }
