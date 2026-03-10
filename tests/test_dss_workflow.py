from rdkit import Chem

from core.profiling import profile_molecule
from core.reliability import estimate_reliability
from core.workflow import DSSWorkflow, PredictorSpec


def mol(smiles: str):
    molecule = Chem.MolFromSmiles(smiles)
    assert molecule is not None
    return molecule


class StubPredictor:
    def __init__(self, output: dict):
        self.output = output
        self.feature_cols = ["MolWeight"]

    def predict(self, mol, *, features_df=None):
        assert features_df is not None
        return dict(self.output)


class BrokenPredictor:
    feature_cols = ["MolWeight"]

    def predict(self, mol, *, features_df=None):
        raise RuntimeError("boom")


class StubDecisionSupport:
    def evaluate(self, *, meta=None, descriptors=None, predictions=None, warnings=None):
        return {
            "rule_version": "test",
            "decision_status": "review",
            "risk_level": "medium",
            "score": 0.42,
            "recommendation": "Требуется ручная проверка.",
            "rationale": ["Тестовый DSS-ответ."],
            "next_actions": ["Проверить результаты вручную."],
            "meta": {"source": "test"},
        }


class StubReadAcross:
    def analyze(self, mol, *, meta=None):
        return {
            "predictions": [
                {
                    "task": "Прогноз по аналогам (LogP)",
                    "value": 1.11,
                    "confidence": "Средняя (аналогов=2, sim=0.71)",
                    "confidence_score": 0.71,
                    "in_domain": None,
                    "notes": "Тестовый прогноз по аналогам.",
                },
                {
                    "task": "Прогноз по аналогам (класс пестицида)",
                    "value": "Гербицид",
                    "confidence": "Высокая (аналогов=2, sim=0.71)",
                    "confidence_score": 0.91,
                    "in_domain": None,
                    "notes": "Тестовое голосование по аналогам.",
                },
            ],
            "analogues": [
                {
                    "rank": 1,
                    "smiles": "CCO",
                    "similarity": 1.0,
                    "value": 1.1,
                    "class_name": "Гербицид",
                    "n_obs": 1,
                },
                {
                    "rank": 2,
                    "smiles": "CCCO",
                    "similarity": 0.42,
                    "value": 1.2,
                    "class_name": "Гербицид",
                    "n_obs": 1,
                },
            ],
            "category": {
                "type": "pesticide_class_context",
                "members": ["Гербицид"],
                "consistency_score": 1.0,
                "summary_ru": "По ближайшим аналогам преобладает класс «Гербицид».",
                "dominant_class": "Гербицид",
            },
            "targets": {
                "logp": {
                    "label_ru": "LogP",
                    "prediction": {
                        "task": "Прогноз по аналогам (LogP)",
                        "value": 1.11,
                        "confidence": "Средняя (аналогов=2, sim=0.71)",
                    },
                    "analogues": [
                        {
                            "rank": 1,
                            "smiles": "CCO",
                            "similarity": 1.0,
                            "value": 1.1,
                            "class_name": "Гербицид",
                        }
                    ],
                    "category": {
                        "summary_ru": "По ближайшим аналогам преобладает класс «Гербицид».",
                    },
                }
            },
            "warnings": [],
        }


def test_profile_detects_aromatic_and_halogenated_features():
    profile = profile_molecule(mol("c1ccccc1Cl"))

    assert profile["aromatic"] is True
    assert profile["halogenated"] is True
    assert any("аромат" in line.lower() for line in profile["summary_ru"])


def test_reliability_is_capped_without_analogues():
    reliability = estimate_reliability(
        predictions=[
            {"task": "LogP", "ad_score": 0.95, "confidence_score": 0.90, "in_domain": True},
        ],
        analogues=[],
        category={},
        warnings=[],
    )

    assert reliability["final_score"] <= 0.64
    assert reliability["final_label"] in {"Низкая", "Средняя"}


def test_workflow_returns_payload_and_handles_predictor_failure():
    workflow = DSSWorkflow(
        [
            PredictorSpec(
                task="LogP",
                predictor=StubPredictor(
                    {
                        "task": "LogP",
                        "value": 1.23,
                        "confidence": "ok",
                        "confidence_score": 0.8,
                        "ad_score": 0.7,
                        "in_domain": True,
                        "notes": "stub",
                    }
                ),
                coverage_name="SVR",
            ),
            PredictorSpec(
                task="Toxicity",
                predictor=BrokenPredictor(),
                coverage_name="Tox",
            ),
        ],
        decision_support=StubDecisionSupport(),
        read_across=StubReadAcross(),
    )

    result = workflow.analyze_molecule(
        mol("CCO"),
        meta={"input": "CCO", "smiles": "CCO", "source": "test"},
        warnings=[],
    )

    assert "profile" in result
    assert "reliability" in result
    assert "payload" in result
    assert result["payload"]["profile"] == result["profile"]
    assert result["payload"]["reliability"] == result["reliability"]
    assert result["decision"]["decision_status"] == "review"
    assert "ошибка прогноза" in result["predictions"][1]["notes"]
    assert result["analogues"]
    assert result["category"]["consistency_score"] == 1.0
    assert all("аналогам" not in pred["task"].lower() for pred in result["predictions"])
    assert result["reliability"]["analogue_support"] > 0
    assert "logp" in result["read_across"]["targets"]
    assert any(
        pred["task"] == "Прогноз по аналогам (LogP)"
        for pred in result["read_across"]["predictions"]
    )
