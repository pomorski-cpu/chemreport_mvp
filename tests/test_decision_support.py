from core.decision_support import DecisionSupport


def test_high_toxicity_leads_to_reject():
    dss = DecisionSupport("config/decision_rules.json")
    decision = dss.evaluate(
        meta={"input": "CCO"},
        descriptors={"cLogP": 2.1, "TPSA": 32.0},
        predictions=[
            {"task": "LogP", "value": 2.1, "in_domain": True},
            {"task": "Toxicity", "value": "Генотоксичный", "prob_toxic": 0.92, "in_domain": None},
        ],
        warnings=[],
    )

    assert decision["decision_status"] == "reject"
    assert decision["risk_level"] in {"high", "critical"}
    assert decision["score"] >= 0.6


def test_out_of_domain_forces_insufficient_data():
    dss = DecisionSupport("config/decision_rules.json")
    decision = dss.evaluate(
        meta={"input": "c1ccccc1"},
        descriptors={"cLogP": 2.0, "TPSA": 45.0},
        predictions=[
            {"task": "LogP", "value": 2.0, "in_domain": False, "ad_ratio": 1.4},
            {"task": "Toxicity", "value": "Не генотоксичный", "prob_toxic": 0.12, "in_domain": None},
        ],
        warnings=[],
    )

    assert decision["decision_status"] == "insufficient_data"
    assert decision["risk_level"] in {"medium", "high", "critical"}


def test_low_risk_profile_can_be_approved():
    dss = DecisionSupport("config/decision_rules.json")
    decision = dss.evaluate(
        meta={"input": "CCO"},
        descriptors={"cLogP": 1.1, "TPSA": 48.0},
        predictions=[
            {"task": "LogP", "value": 1.1, "in_domain": True},
            {"task": "Toxicity", "value": "Не генотоксичный", "prob_toxic": 0.08, "in_domain": None},
        ],
        warnings=[],
    )

    assert decision["decision_status"] in {"approve", "review"}
    assert decision["score"] <= 0.6
