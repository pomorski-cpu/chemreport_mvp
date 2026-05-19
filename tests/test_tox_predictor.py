from rdkit import Chem
import pytest

from core.tox_predictor import ToxPredictor, multilabel_hit_metrics


class _FakeClassifierModel:
    classes_ = [0, 1, 2, 3, 4]

    def predict(self, Xdf):
        return [0]

    def predict_proba(self, Xdf):
        return [[0.62, 0.08, 0.12, 0.09, 0.09]]


class _FakeToxicBinaryModel:
    classes_ = [0, 1]

    def predict(self, Xdf):
        return [1]

    def predict_proba(self, Xdf):
        return [[0.18, 0.82]]


class _FakeNonToxicBinaryModel:
    classes_ = [0, 1]

    def predict(self, Xdf):
        return [0]

    def predict_proba(self, Xdf):
        return [[0.82, 0.18]]


class _FakeMultilabelModel:
    def predict_proba(self, Xdf):
        return [[0.90, 0.80]]


def test_non_toxic_classifier_does_not_emit_toxic_confidence():
    predictor = ToxPredictor.__new__(ToxPredictor)
    predictor.model = _FakeClassifierModel()
    predictor.meta = {
        "name": "RandomForest (Pesticide class)",
        "target_name": "Pesticide Class",
    }
    predictor.feature_cols = []
    predictor.class_names = {
        "0": "Прочее",
        "1": "Гербицид",
        "2": "Инсектицид",
        "3": "Микробиоцид",
        "4": "Фунгицид",
    }
    predictor.classes = [0, 1, 2, 3, 4]
    predictor.decision_threshold = 0.5
    predictor.toxic_class_id = predictor._resolve_toxic_class_id()
    predictor.non_toxic_class_id = predictor._resolve_non_toxic_class_id()

    mol = Chem.MolFromSmiles("c1ccccc1")
    assert mol is not None

    out = predictor.predict(mol)

    assert out["value"] == "Прочее"
    assert out["confidence"] == "Высокая"
    assert "P(токсичности)" not in out["confidence"]
    assert "P(class=" not in out["confidence"]
    assert out["prob_toxic"] is None
    assert out["confidence_score"] == 0.62 / (0.62 + 0.12)
    assert "наиболее вероятный класс" in out["notes"]
    assert "топ-3 вероятности" in out["notes"]
    assert "0.620" in out["notes"]


def test_toxicity_notes_show_selected_class_probability():
    predictor = ToxPredictor.__new__(ToxPredictor)
    predictor.model = _FakeToxicBinaryModel()
    predictor.meta = {
        "name": "RandomForest (tox)",
        "target_name": "Toxicity",
    }
    predictor.feature_cols = []
    predictor.class_names = {
        "0": "Не генотоксичный",
        "1": "Генотоксичный",
    }
    predictor.classes = [0, 1]
    predictor.decision_threshold = 0.5
    predictor.toxic_class_id = predictor._resolve_toxic_class_id()
    predictor.non_toxic_class_id = predictor._resolve_non_toxic_class_id()

    mol = Chem.MolFromSmiles("CCN")
    assert mol is not None

    out = predictor.predict(mol)

    assert out["value"] == "Генотоксичный"
    assert out["notes"] == "P(Генотоксичный)=0.820"


def test_binary_toxicity_confidence_is_decision_confidence_not_toxic_probability():
    predictor = ToxPredictor.__new__(ToxPredictor)
    predictor.model = _FakeNonToxicBinaryModel()
    predictor.meta = {
        "name": "RandomForest (tox)",
        "target_name": "Toxicity",
    }
    predictor.feature_cols = []
    predictor.class_names = {
        "0": "Не генотоксичный",
        "1": "Генотоксичный",
    }
    predictor.classes = [0, 1]
    predictor.decision_threshold = 0.5
    predictor.toxic_class_id = predictor._resolve_toxic_class_id()
    predictor.non_toxic_class_id = predictor._resolve_non_toxic_class_id()

    mol = Chem.MolFromSmiles("CCN")
    assert mol is not None

    out = predictor.predict(mol)

    assert out["prob_toxic"] == 0.18
    assert out["toxicity_decision"] is False
    assert out["confidence_score"] == pytest.approx(0.82)


def test_multilabel_hit_metrics_report_any_hit_and_strict_diagnostics():
    metrics = multilabel_hit_metrics(
        predicted_labels=["Инсектицид", "Микробиоцид"],
        true_labels=["Инсектицид", "Фунгицид"],
    )

    assert metrics["any_hit"] is True
    assert metrics["all_hit"] is False
    assert metrics["exact_match"] is False
    assert metrics["jaccard"] == 0.333
    assert metrics["extra_label_count"] == 1


def test_multilabel_prediction_removes_other_when_main_label_is_present():
    predictor = ToxPredictor.__new__(ToxPredictor)
    predictor.model = _FakeMultilabelModel()
    predictor.meta = {
        "task_type": "multilabel_classification",
        "target_name": "Pesticide Class",
        "labels": ["Other", "Herbicide"],
        "label_thresholds": {"Other": 0.5, "Herbicide": 0.5},
    }
    predictor.class_names = {}
    predictor.classes = []

    out = predictor._predict_multilabel(None)

    assert out["value"] == "Herbicide"
    assert out["predicted_labels"] == ["Herbicide"]
    assert out["confidence_score"] == pytest.approx(0.80)
