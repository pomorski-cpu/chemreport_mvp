import math
import pytest
from rdkit import Chem

from core.predictor import SVRPredictor
from core.torch_predictor import TorchPredictor


def mol(smiles: str):
    m = Chem.MolFromSmiles(smiles)
    assert m is not None, f"Invalid SMILES in test: {smiles}"
    return m


@pytest.fixture(scope="session")
def svr():
    return SVRPredictor(models_dir="models")


@pytest.fixture(scope="session")
def mlp():
    return TorchPredictor(
        models_dir="models",
        model_file="mlp_regression_state.pt",
        meta_file="mlp_regression_meta.json",
        scaler_file="mlp_regression_scaler.joblib",
        device="cpu",
    )


@pytest.mark.parametrize("smiles", [
    "CC(=O)Oc1ccccc1C(=O)O",       # аспирин
    "CCO",                         # этанол
    "c1ccccc1",                    # бензол
    "CCCCCCCC",                    # октан
    "CCN(CC)CC",                   # триэтиламин
])
def test_predict_returns_numeric_and_finite(svr, mlp, smiles):
    m = mol(smiles)

    out_svr = svr.predict(m)
    assert "value" in out_svr
    assert isinstance(out_svr["value"], (int, float))
    assert math.isfinite(out_svr["value"])

    out_mlp = mlp.predict(m)
    assert "value" in out_mlp
    assert isinstance(out_mlp["value"], (int, float))
    assert math.isfinite(out_mlp["value"])


def test_prediction_is_deterministic_same_input(svr, mlp):
    m = mol("CC(=O)Oc1ccccc1C(=O)O")  # aspirin
    out1 = svr.predict(m)["value"]
    out2 = svr.predict(m)["value"]
    assert out1 == out2, "SVR should be deterministic."

    out1 = mlp.predict(m)["value"]
    out2 = mlp.predict(m)["value"]
    assert out1 == out2, "Torch MLP should be deterministic in eval()."


def test_svr_ad_fields_present_and_reasonable(svr):
    m = mol("CCO")
    out = svr.predict(m)

    # если ты уже выводишь AD/Confidence
    # допускаем, что может быть пусто если ref не загружен, но тогда это тоже сигнал
    if out.get("ad_distance") is None or out.get("ad_score") is None:
        pytest.skip("SVR AD reference is missing (svr_logp_ref.npz not loaded).")

    assert isinstance(out["ad_distance"], (int, float))
    assert out["ad_distance"] >= 0

    assert isinstance(out["ad_score"], (int, float))
    assert 0 <= out["ad_score"] <= 1

    # label должен соответствовать score (если ты это так зашил)
    conf = str(out.get("confidence", ""))
    assert conf != "", "Confidence string is empty but AD is present."


def test_similarity_effect_on_ad_score(svr):
    """
    Базовый sanity: разные по структуре молекулы должны давать валидные AD score,
    и метрика должна быть чувствительна (не NaN/не выход за диапазон).
    """
    m_sim = mol("CCO")                # этанол
    m_far = mol("c1ccc2cccc3cccc(c1)c23")  # фенантрен-подобное (жирный ароматик)

    out_sim = svr.predict(m_sim)
    out_far = svr.predict(m_far)

    if out_sim.get("ad_score") is None or out_far.get("ad_score") is None:
        pytest.skip("SVR AD reference is missing.")

    assert isinstance(out_sim["ad_score"], (int, float))
    assert isinstance(out_far["ad_score"], (int, float))
    assert 0 <= out_sim["ad_score"] <= 1
    assert 0 <= out_far["ad_score"] <= 1
    assert out_sim["ad_score"] != out_far["ad_score"], (
        "AD scores for structurally different molecules unexpectedly совпали."
    )


def test_torch_ad_fields_present_and_reasonable(mlp):
    m = mol("CCO")
    out = mlp.predict(m)

    if out.get("ad_distance") is None or out.get("ad_score") is None:
        pytest.skip("Torch AD reference is missing (mlp_logp_ref.npz not loaded).")

    assert isinstance(out["ad_distance"], (int, float))
    assert out["ad_distance"] >= 0

    assert isinstance(out["ad_score"], (int, float))
    assert 0 <= out["ad_score"] <= 1

    conf = str(out.get("confidence", ""))
    assert conf != "", "Confidence string is empty but AD is present."
