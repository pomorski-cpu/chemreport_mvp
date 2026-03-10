import json

from rdkit import Chem

from core.read_across import ReadAcrossService
from core.read_across_sqlite import build_read_across_sqlite


def test_read_across_returns_prediction_and_category(tmp_path):
    logp_path = tmp_path / "logp.csv"
    logp_path.write_text(
        "MolWeight,LogP,SMILES\n"
        "46.07,1.10,CCO\n"
        "60.10,1.40,CCCO\n"
        "78.11,2.10,c1ccccc1\n",
        encoding="utf-8",
    )

    category_path = tmp_path / "category.csv"
    category_path.write_text(
        "SMILES,Class\n"
        "CCO,Гербицид\n"
        "CCCO,Гербицид\n"
        "c1ccccc1,Фунгицид\n",
        encoding="utf-8",
    )
    db_path = tmp_path / "read_across.db"
    build_read_across_sqlite(
        db_path,
        category_csv=category_path,
        logp_csv=logp_path,
        pesticide_csv=category_path,
    )

    config_path = tmp_path / "read_across.json"
    config_path.write_text(
        json.dumps(
            {
                "version": "test-ra-sqlite",
                "database": str(db_path),
                "category_table": "category_dataset",
                "category_dataset": "",
                "category_smiles_col": "SMILES",
                "category_value_col": "Class",
                "fingerprint_radius": 2,
                "fingerprint_bits": 512,
                "top_k": 3,
                "min_similarity": 0.2,
                "fallback_similarity": 0.0,
                "weight_power": 2.0,
                "cache_file": str(tmp_path / "ra_cache.pkl.gz"),
                "targets": [
                    {
                        "key": "logp",
                        "label_ru": "LogP",
                        "prediction_task": "Прогноз по аналогам (LogP)",
                        "table": "logp_dataset",
                        "dataset": "",
                        "smiles_col": "SMILES",
                        "value_col": "LogP",
                        "mode": "regression",
                        "enabled": True,
                    },
                    {
                        "key": "pesticide_class",
                        "label_ru": "Класс пестицида",
                        "prediction_task": "Прогноз по аналогам (класс пестицида)",
                        "table": "pesticide_class_dataset",
                        "dataset": "",
                        "smiles_col": "SMILES",
                        "value_col": "Class",
                        "mode": "classification",
                        "enabled": True,
                    },
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    service = ReadAcrossService(str(config_path))
    mol = Chem.MolFromSmiles("CCO")
    assert mol is not None

    result = service.analyze(mol, meta={"smiles": "CCO"})

    assert result["predictions"]
    assert any(item["task"] == "Прогноз по аналогам (LogP)" for item in result["predictions"])
    assert any(item["task"] == "Прогноз по аналогам (класс пестицида)" for item in result["predictions"])
    assert result["analogues"]
    assert result["analogues"][0]["smiles"] == "CCO"
    assert result["analogues"][0]["similarity"] == 1.0
    assert result["category"]["consistency_score"] > 0
    assert "Гербицид" in result["category"]["summary_ru"]
    assert "logp" in result["targets"]
    assert "pesticide_class" in result["targets"]
    assert service.cache_info()["exists"] is True


def test_read_across_can_clear_cache(tmp_path):
    logp_path = tmp_path / "logp.csv"
    logp_path.write_text("LogP,SMILES\n1.1,CCO\n", encoding="utf-8")
    category_path = tmp_path / "category.csv"
    category_path.write_text("SMILES,Class\nCCO,Гербицид\n", encoding="utf-8")
    db_path = tmp_path / "read_across.db"
    build_read_across_sqlite(
        db_path,
        category_csv=category_path,
        logp_csv=logp_path,
    )
    cache_path = tmp_path / "cache.pkl.gz"
    config_path = tmp_path / "read_across.json"
    config_path.write_text(
        json.dumps(
            {
                "version": "test-ra-clear-sqlite",
                "database": str(db_path),
                "category_table": "category_dataset",
                "category_dataset": "",
                "category_smiles_col": "SMILES",
                "category_value_col": "Class",
                "fingerprint_radius": 2,
                "fingerprint_bits": 256,
                "top_k": 3,
                "min_similarity": 0.1,
                "fallback_similarity": 0.0,
                "weight_power": 2.0,
                "cache_file": str(cache_path),
                "targets": [
                    {
                        "key": "logp",
                        "label_ru": "LogP",
                        "prediction_task": "Прогноз по аналогам (LogP)",
                        "table": "logp_dataset",
                        "dataset": "",
                        "smiles_col": "SMILES",
                        "value_col": "LogP",
                        "mode": "regression",
                        "enabled": True,
                    }
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    service = ReadAcrossService(str(config_path))
    mol = Chem.MolFromSmiles("CCO")
    assert mol is not None
    service.analyze(mol, meta={"smiles": "CCO"})
    assert service.cache_info()["exists"] is True

    service.clear_cache()
    assert service.cache_info()["exists"] is False


def test_read_across_public_config_without_datasets_does_not_crash(tmp_path):
    config_path = tmp_path / "read_across.json"
    config_path.write_text(
        json.dumps(
            {
                "version": "public-safe",
                "database": "",
                "category_table": "",
                "category_dataset": "",
                "category_smiles_col": "SMILES",
                "category_value_col": "Class",
                "fingerprint_radius": 2,
                "fingerprint_bits": 256,
                "top_k": 3,
                "min_similarity": 0.1,
                "fallback_similarity": 0.0,
                "weight_power": 2.0,
                "cache_file": str(tmp_path / "cache.pkl.gz"),
                "targets": [
                    {
                        "key": "logp",
                        "label_ru": "LogP",
                        "prediction_task": "Прогноз по аналогам (LogP)",
                        "table": "logp_dataset",
                        "dataset": "",
                        "smiles_col": "SMILES",
                        "value_col": "LogP",
                        "mode": "regression",
                        "enabled": False,
                    }
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    service = ReadAcrossService(str(config_path))
    mol = Chem.MolFromSmiles("CCO")
    assert mol is not None

    result = service.analyze(mol, meta={"smiles": "CCO"})

    assert result["predictions"] == []
    assert result["analogues"] == []
    assert result["category"]["type"] == "analogue_data_unavailable"
