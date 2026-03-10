from rdkit import Chem

from core.tox_predictor import ToxPredictor


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
    assert "P(toxic)" not in out["confidence"]
    assert "P(class=" not in out["confidence"]
    assert out["prob_toxic"] is None
    assert out["confidence_score"] == 0.62 / (0.62 + 0.12)
    assert out["notes"] == "1 vs other=0.080; top=0.620"


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
