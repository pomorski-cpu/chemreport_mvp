from core.decision_support import DecisionSupport


def _dss():
    return DecisionSupport("config/decision_rules.json")


def _reliable():
    return {"final_score": 0.72, "model_confidence": 0.82}


def test_high_prob_toxic_rejects_and_returns_v2_payload():
    decision = _dss().evaluate(
        meta={"input": "CCO"},
        descriptors={"cLogP": 2.1, "TPSA": 32.0},
        predictions=[
            {"task": "LogP", "value": 2.1, "in_domain": True},
            {"task": "Toxicity", "value": "Генотоксичный", "prob_toxic": 0.92, "in_domain": True},
        ],
        warnings=[],
        reliability=_reliable(),
    )

    assert decision["decision_status"] == "reject"
    assert decision["risk_level"] in {"high", "critical"}
    assert decision["hazard_score"] >= 0.65
    assert decision["component_scores"]["toxicity"] >= 0.65
    assert decision["evidence"]
    assert "uncertainty_score" in decision
    assert "data_quality_flags" in decision


def test_bioactivity_toxic_signal_drives_review():
    decision = _dss().evaluate(
        meta={"input": "CCO"},
        descriptors={"cLogP": 1.4, "TPSA": 45.0},
        predictions=[
            {"task": "Toxicity", "value": "Нетоксично", "prob_toxic": 0.12, "in_domain": True},
            {"task": "Биоактивность: EC50, водные беспозвоночные", "value": "toxic", "prob_toxic": 0.81, "toxicity_decision": True, "in_domain": True},
        ],
        warnings=[],
        reliability=_reliable(),
    )

    assert decision["decision_status"] in {"review", "reject"}
    assert decision["component_scores"]["bioactivity"] >= 0.35
    assert any("EC50" in item["source"] for item in decision["evidence"])


def test_multiple_bioactivity_toxic_signals_can_reject():
    decision = _dss().evaluate(
        meta={"input": "CCCl"},
        descriptors={"cLogP": 2.0, "TPSA": 30.0},
        predictions=[
            {"task": "Toxicity", "value": "Нетоксично", "prob_toxic": 0.30, "in_domain": True},
            {"task": "Биоактивность: EC50, водные беспозвоночные", "prob_toxic": 0.83, "toxicity_decision": True, "in_domain": True},
            {"task": "Биоактивность: LC50, рыбы", "prob_toxic": 0.78, "toxicity_decision": True, "in_domain": True},
        ],
        warnings=[],
        reliability=_reliable(),
    )

    assert decision["decision_status"] in {"reject", "review"}
    assert decision["hazard_score"] >= 0.65


def test_out_of_domain_forces_insufficient_data():
    decision = _dss().evaluate(
        meta={"input": "c1ccccc1"},
        descriptors={"cLogP": 2.0, "TPSA": 45.0},
        predictions=[
            {"task": "LogP", "value": 2.0, "in_domain": False, "ad_ratio": 1.4},
            {"task": "Toxicity", "value": "Нетоксично", "prob_toxic": 0.12, "in_domain": True},
        ],
        warnings=[],
        reliability=_reliable(),
    )

    assert decision["decision_status"] == "insufficient_data"
    assert decision["uncertainty_score"] >= 0.70
    assert any(flag["code"] == "out_of_domain" for flag in decision["data_quality_flags"])


def test_low_reliability_prevents_approve():
    decision = _dss().evaluate(
        meta={"input": "CCO"},
        descriptors={"cLogP": 1.0, "TPSA": 50.0},
        predictions=[
            {"task": "Toxicity", "value": "Нетоксично", "prob_toxic": 0.08, "in_domain": True},
        ],
        warnings=[],
        reliability={"final_score": 0.48, "model_confidence": 0.50},
    )

    assert decision["decision_status"] in {"review", "insufficient_data"}
    assert decision["decision_status"] != "approve"
    assert any(flag["code"] in {"questionable_reliability", "low_model_confidence"} for flag in decision["data_quality_flags"])


def test_conflicting_models_force_at_least_review():
    decision = _dss().evaluate(
        meta={"input": "CCN"},
        descriptors={"cLogP": 1.5, "TPSA": 35.0},
        predictions=[
            {"task": "Toxicity", "value": "Нетоксично", "prob_toxic": 0.10, "in_domain": True},
            {"task": "Биоактивность: LD50, млекопитающие перорально", "prob_toxic": 0.88, "toxicity_decision": True, "in_domain": True},
        ],
        warnings=[],
        reliability=_reliable(),
    )

    assert decision["decision_status"] in {"review", "reject"}
    assert decision["conflicts"]
    assert decision["component_scores"]["conflict"] > 0


def test_low_risk_with_good_reliability_can_approve():
    decision = _dss().evaluate(
        meta={"input": "CCO"},
        descriptors={"cLogP": 1.1, "TPSA": 48.0},
        predictions=[
            {"task": "LogP", "value": 1.1, "in_domain": True},
            {"task": "Toxicity", "value": "Нетоксично", "prob_toxic": 0.08, "in_domain": True},
            {"task": "Биоактивность: LC50, рыбы", "prob_toxic": 0.10, "toxicity_decision": False, "in_domain": True},
        ],
        warnings=[],
        reliability={"final_score": 0.80, "model_confidence": 0.82},
    )

    assert decision["decision_status"] == "approve"
    assert decision["hazard_score"] < 0.35
    assert decision["uncertainty_score"] < 0.35


def test_read_across_toxic_strengthens_hazard():
    decision = _dss().evaluate(
        meta={"input": "CCO"},
        descriptors={"cLogP": 1.2, "TPSA": 42.0},
        predictions=[
            {"task": "Toxicity", "value": "Нетоксично", "prob_toxic": 0.30, "in_domain": True},
        ],
        warnings=[],
        reliability=_reliable(),
        read_across={
            "targets": {
                "toxicity": {
                    "label_ru": "Генотоксичность",
                    "prediction": {"value": "Генотоксичный", "confidence": "Высокая", "confidence_score": 0.86},
                    "analogues": [],
                }
            }
        },
    )

    assert decision["component_scores"]["read_across"] > 0
    assert decision["decision_status"] in {"review", "reject"}
    assert any("Read-across" in item["source"] for item in decision["evidence"])
