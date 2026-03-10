import pytest
from rdkit import Chem

from core.predictor import SVRPredictor
from core.torch_predictor import TorchPredictor
from core.featurizer_rdkit_inchi import build_feature_df


def mol(smiles: str):
    m = Chem.MolFromSmiles(smiles)
    assert m is not None
    return m


def test_features_have_expected_columns_for_svr():
    p = SVRPredictor(models_dir="models")
    m = mol("CCO")

    Xdf = build_feature_df(m)
    # reindex в predictor делает fill_value=0 — но тут важно что колонок хватает
    missing = [c for c in p.feature_cols if c not in Xdf.columns]
    # допускаем, что featurizer может не отдавать часть колонок — но тогда predictor их заполнит 0
    # это нормально, но если missing слишком много — модель почти "пустая"
    assert len(missing) <= max(3, int(0.1 * len(p.feature_cols))), (
        f"Too many missing feature columns for SVR: {len(missing)} / {len(p.feature_cols)}"
    )


def test_features_have_expected_columns_for_mlp():
    p = TorchPredictor(
        models_dir="models",
        model_file="mlp_regression_state.pt",
        meta_file="mlp_regression_meta.json",
        scaler_file="mlp_regression_scaler.joblib",
        device="cpu",
    )
    m = mol("CCO")
    Xdf = build_feature_df(m)
    missing = [c for c in p.feature_cols if c not in Xdf.columns]
    assert len(missing) <= max(3, int(0.1 * len(p.feature_cols))), (
        f"Too many missing feature columns for MLP: {len(missing)} / {len(p.feature_cols)}"
    )
