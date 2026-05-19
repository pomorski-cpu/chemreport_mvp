from rdkit import Chem
import numpy as np
import pandas as pd

from core.applicability_domain import ApplicabilityDomainService, ReferenceBundle


def test_applicability_domain_unknown_without_reference():
    class Predictor:
        meta = {}

    mol = Chem.MolFromSmiles("CCO")
    service = ApplicabilityDomainService()
    result = service.evaluate_prediction(
        mol=mol,
        task="Dummy",
        predictor=Predictor(),
        prediction={"task": "Dummy", "value": 1},
        features_df=None,
    )
    assert result["status"] == "unknown"
    assert result["method"] == "no_reference"


def test_applicability_domain_preserves_native_ad():
    class Predictor:
        meta = {}

    mol = Chem.MolFromSmiles("CCO")
    service = ApplicabilityDomainService()
    result = service.evaluate_prediction(
        mol=mol,
        task="LogP",
        predictor=Predictor(),
        prediction={
            "task": "LogP",
            "ad_score": 0.7,
            "ad_distance": 1.0,
            "ad_threshold": 2.0,
            "ad_ratio": 0.5,
            "in_domain": True,
        },
        features_df=None,
    )
    assert result["status"] == "in_domain"
    assert result["method"] == "model_native_ad"


def test_applicability_summary_counts_statuses():
    service = ApplicabilityDomainService()
    summary = service.summarize(
        [
            {"status": "in_domain", "ad_score": 0.8},
            {"status": "borderline", "ad_score": 0.4},
            {"status": "unknown", "ad_score": None},
        ]
    )
    assert summary["overall_status"] == "borderline"
    assert summary["counts"]["in_domain"] == 1
    assert summary["counts"]["unknown"] == 1


def _reference(smiles: list[str]) -> ReferenceBundle:
    return ReferenceBundle(
        path="test_ref.npz",
        X=np.array([[0.0], [0.2], [0.4]], dtype=np.float32),
        feature_cols=["x"],
        smiles=smiles,
        canonical_smiles=[Chem.MolToSmiles(Chem.MolFromSmiles(item), canonical=True) for item in smiles],
        labels=["a", "b", "c"],
        scaffolds=["acyclic_" + item for item in smiles],
        descriptor_names=[],
        descriptor_min=None,
        descriptor_max=None,
    )


def test_exact_reference_match_sets_high_ad_and_promotes_confidence():
    class Predictor:
        meta = {}

    mol = Chem.MolFromSmiles("CCO")
    service = ApplicabilityDomainService()
    service._load_reference = lambda predictor: _reference(["CCO", "CCN", "CCC"])
    result = service.evaluate_prediction(
        mol=mol,
        task="Dummy",
        predictor=Predictor(),
        prediction={"task": "Dummy", "value": 1, "confidence_score": 0.60},
        features_df=pd.DataFrame({"x": [0.0]}),
    )
    prediction = {"task": "Dummy", "value": 1, "confidence": "base", "confidence_score": 0.60}
    service.apply_to_prediction(prediction, result)

    assert result["method"] == "exact_reference_match"
    assert result["ad_score"] == 1.0
    assert result["in_domain"] is True
    assert prediction["confidence_score"] == 0.75
    assert prediction["model_confidence_score"] == 0.60


def test_high_similarity_does_not_promote_weak_model_confidence():
    class Predictor:
        meta = {}

    mol = Chem.MolFromSmiles("CCO")
    service = ApplicabilityDomainService()
    service._load_reference = lambda predictor: _reference(["CCCC", "CCN", "CCC"])
    service._similarity_ad = lambda mol, ref: {"top_similarity": 0.90, "nearest": [{"similarity": 0.90}]}
    result = service.evaluate_prediction(
        mol=mol,
        task="Dummy",
        predictor=Predictor(),
        prediction={"task": "Dummy", "value": 1, "confidence_score": 0.40},
        features_df=pd.DataFrame({"x": [0.0]}),
    )
    prediction = {"task": "Dummy", "value": 1, "confidence": "base", "confidence_score": 0.40}
    service.apply_to_prediction(prediction, result)

    assert result["similarity_tier"] == "high"
    assert result["ad_score"] >= 0.85
    assert prediction["confidence_score"] == 0.40


def test_medium_similarity_caps_confidence_at_medium_level():
    class Predictor:
        meta = {}

    mol = Chem.MolFromSmiles("CCO")
    service = ApplicabilityDomainService()
    service._load_reference = lambda predictor: _reference(["CCCC", "CCN", "CCC"])
    service._similarity_ad = lambda mol, ref: {"top_similarity": 0.75, "nearest": [{"similarity": 0.75}]}
    result = service.evaluate_prediction(
        mol=mol,
        task="Dummy",
        predictor=Predictor(),
        prediction={"task": "Dummy", "value": 1, "confidence_score": 0.90},
        features_df=pd.DataFrame({"x": [0.0]}),
    )
    prediction = {"task": "Dummy", "value": 1, "confidence": "base", "confidence_score": 0.90}
    service.apply_to_prediction(prediction, result)

    assert result["similarity_tier"] == "medium"
    assert result["ad_score"] >= 0.65
    assert prediction["confidence_score"] == 0.70
