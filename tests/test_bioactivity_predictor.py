import math

from rdkit import Chem

from core.bioactivity_predictor import build_bioactivity_feature_df
from core.predictor_factory import PredictorFactory


TASKS = [
    "bioactivity_ec50_aquatic_binary",
    "bioactivity_lc50_fish_binary",
    "bioactivity_ld50_mammals_oral_binary",
]


def mol(smiles: str):
    molecule = Chem.MolFromSmiles(smiles)
    assert molecule is not None
    return molecule


def test_bioactivity_v2_default_feature_schema_has_no_fingerprints():
    xdf = build_bioactivity_feature_df(mol("CCOc1ccc2nc(S(N)(=O)=O)sc2c1"))

    assert "desc_MolWt" in xdf.columns
    assert not any(col.startswith("maccs_") for col in xdf.columns)
    assert not any(col.startswith("morgan_") for col in xdf.columns)
    assert xdf.drop(columns=["canonical_smiles"]).map(math.isfinite).all().all()


def test_bioactivity_binary_predictors_load_active_no_fingerprint_models():
    factory = PredictorFactory("models/registry.json")

    for task_key in TASKS:
        predictor = factory.create(task_key)
        assert predictor.meta["feature_schema"].endswith("no_fingerprints")
        assert not any(col.startswith("maccs_") for col in predictor.model_feature_cols)
        assert not any(col.startswith("morgan_") for col in predictor.model_feature_cols)

        out = predictor.predict(mol("CCOc1ccc2nc(S(N)(=O)=O)sc2c1"))
        assert out["value"] in {"Токсично по GHS-порогу", "Нетоксично по GHS-порогу"}
        assert 0.0 <= out["prob_toxic"] <= 1.0
        assert out["toxicity_threshold"] == 0.5
        assert isinstance(out["toxicity_decision"], bool)
        assert math.isfinite(out["confidence_score"])
        assert "P(токсичности)=" in out["confidence"]



def test_bioactivity_regression_predictors_report_ad_trust():
    factory = PredictorFactory("models/registry.json")

    for task_key in [
        "bioactivity_ec50_aquatic_regression",
        "bioactivity_ld50_mammals_oral_regression",
    ]:
        predictor = factory.create(task_key)
        out = predictor.predict(mol("CCOc1ccc2nc(S(N)(=O)=O)sc2c1"))

        assert math.isfinite(out["regression_value"])
        assert math.isfinite(out["regression_log10_value"])
        assert out["regression_unit"] in {"мг/л", "мг/кг"}
        assert out["prob_toxic"] is None
        assert out["toxicity_decision"] is None
        assert "AD Tanimoto" in out["confidence"]
        assert out["in_domain"] in {True, False}
        assert out["ad_min_tanimoto"] == 0.45

