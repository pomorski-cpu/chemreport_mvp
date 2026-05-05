from rdkit import Chem

from core.applicability_domain import ApplicabilityDomainService


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
