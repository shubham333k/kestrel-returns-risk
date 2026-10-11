"""
tests/test_service.py — Unit & End-to-End HTTP Tests for Returns Risk API
==========================================================================
Exercises:
1. Direct function unit tests (healthz, valid predict, unknown category, sanity).
2. FastAPI TestClient HTTP lifecycle:
   - GET /healthz
   - GET / (UI HTML response)
   - Valid POST /predict (structure, score bounds, reasons, disclaimer)
   - Input validation errors (out of range, missing fields, negative numbers)
   - Graceful handling of unknown SKUs and categories without stack traces
"""

import json
from pathlib import Path
import pytest
import pandas as pd
import joblib
from fastapi.testclient import TestClient

from kestrel.data_paths import find_data_file
from service.app import (
    app,
    OrderRiskRequest,
    predict_risk,
    health_check,
    prepare_features,
    generate_reasons,
    state,
)


@pytest.fixture(autouse=True)
def setup_state():
    base_dir = Path(__file__).parent.parent
    artifacts_dir = base_dir / "artifacts"
    state["pipeline"] = joblib.load(artifacts_dir / "model.joblib")
    with open(artifacts_dir / "metadata.json", "r") as f:
        state["metadata"] = json.load(f)
    products_path = find_data_file("products.csv")
    state["products"] = pd.read_csv(products_path)


# ── Direct Endpoint Unit Tests ────────────────────────────────────────────────
def test_healthz():
    res = health_check()
    assert res["status"] == "ok"
    assert res["model_loaded"] is True


def test_valid_predict():
    req = OrderRiskRequest(
        order_id="KO2610504",
        sku="KH-AF-01",
        sales_channel="app",
        payment_mode="cod",
        discount_pct=15.0,
        qty=1,
        order_value_inr=4500.0,
        promised_delivery_days=5,
        is_gift="N",
        customer_prior_orders=2,
        customer_prior_returns=1,
    )
    res = predict_risk(req)
    assert res.order_id == "KO2610504"
    assert 0.0 <= res.return_risk_score <= 1.0
    assert res.risk_band in ["HIGH", "MEDIUM", "LOW"]
    assert len(res.reasons) > 0
    assert "Risk estimate for human review" in res.note


def test_unseen_category():
    req = OrderRiskRequest(
        order_id="TEST_UNSEEN",
        sku="KH-UNKNOWN-99",
        sales_channel="unknown_channel_xyz",
        payment_mode="bitcoin_crypto",
        discount_pct=10.0,
        qty=1,
        order_value_inr=3000.0,
        promised_delivery_days=3,
        is_gift="N",
        customer_prior_orders=0,
        customer_prior_returns=0,
    )
    res = predict_risk(req)
    assert 0.0 <= res.return_risk_score <= 1.0


def test_extreme_values():
    req = OrderRiskRequest(
        order_id="TEST_EXTREME",
        sku="KH-AF-01",
        sales_channel="app",
        payment_mode="cod",
        discount_pct=100.0,
        qty=999,
        order_value_inr=999999.0,
        promised_delivery_days=90,
        is_gift="Y",
        customer_prior_orders=100,
        customer_prior_returns=100,
    )
    res = predict_risk(req)
    assert 0.0 <= res.return_risk_score <= 1.0


def test_risky_vs_safe_sanity():
    risky_req = OrderRiskRequest(
        order_id="RISKY",
        sales_channel="marketplace",
        payment_mode="cod",
        discount_pct=50.0,
        is_gift="Y",
        customer_prior_orders=3,
        customer_prior_returns=3,
    )
    safe_req = OrderRiskRequest(
        order_id="SAFE",
        sales_channel="app",
        payment_mode="prepaid_upi",
        discount_pct=0.0,
        is_gift="N",
        customer_prior_orders=5,
        customer_prior_returns=0,
    )
    res_risky = predict_risk(risky_req)
    res_safe = predict_risk(safe_req)
    assert res_risky.return_risk_score > res_safe.return_risk_score


# ── HTTP TestClient End-to-End Tests ──────────────────────────────────────────
def test_http_healthz_and_ui():
    """Tests GET /healthz and GET / via TestClient HTTP request."""
    with TestClient(app) as client:
        resp_h = client.get("/healthz")
        assert resp_h.status_code == 200
        data_h = resp_h.json()
        assert data_h["status"] == "ok"
        assert data_h["model_loaded"] is True

        resp_ui = client.get("/")
        assert resp_ui.status_code == 200
        assert "<title>Kestrel Home" in resp_ui.text
        assert "Risk estimate for human review" in resp_ui.text


def test_http_valid_predict():
    """Tests POST /predict end-to-end via HTTP client."""
    payload = {
        "order_id": "KO2610504",
        "sku": "KH-AF-01",
        "sales_channel": "app",
        "payment_mode": "cod",
        "discount_pct": 15.0,
        "qty": 1,
        "order_value_inr": 4500.0,
        "promised_delivery_days": 5,
        "is_gift": "N",
        "customer_prior_orders": 2,
        "customer_prior_returns": 1,
    }
    with TestClient(app) as client:
        resp = client.post("/predict", json=payload)
        assert resp.status_code == 200
        data = resp.json()
        assert data["order_id"] == "KO2610504"
        assert 0.0 <= data["return_risk_score"] <= 1.0
        assert data["risk_band"] in ["HIGH", "MEDIUM", "LOW"]
        assert "threshold" in data
        assert len(data["reasons"]) > 0
        assert "Risk estimate for human review" in data["note"]


def test_http_validation_errors():
    """Tests that out-of-range or missing fields return clean HTTP 422 errors without stack traces."""
    with TestClient(app) as client:
        # Discount > 100%
        resp = client.post("/predict", json={"order_id": "ERR_1", "discount_pct": 150.0})
        assert resp.status_code == 422
        assert "detail" in resp.json()
        assert "Traceback" not in resp.text

        # Negative discount
        resp = client.post("/predict", json={"order_id": "ERR_2", "discount_pct": -5.0})
        assert resp.status_code == 422
        assert "detail" in resp.json()

        # Zero quantity (qty >= 1 required)
        resp = client.post("/predict", json={"order_id": "ERR_3", "qty": 0})
        assert resp.status_code == 422
        assert "detail" in resp.json()

        # Missing order_id
        resp = client.post("/predict", json={"sku": "KH-AF-01"})
        assert resp.status_code == 422
        assert "detail" in resp.json()


def test_http_unknown_sku_and_category_handling():
    """Tests that uncatalogued SKUs and unseen categories are handled gracefully with HTTP 200."""
    payload = {
        "order_id": "TEST_HTTP_UNKNOWN",
        "sku": "KH-NONEXISTENT-999",
        "sales_channel": "snapchat_store",
        "payment_mode": "cryptocurrency",
        "discount_pct": 5.0,
        "qty": 1,
        "order_value_inr": 2500.0,
        "promised_delivery_days": 4,
        "is_gift": "N",
        "customer_prior_orders": 0,
        "customer_prior_returns": 0,
    }
    with TestClient(app) as client:
        resp = client.post("/predict", json=payload)
        assert resp.status_code == 200
        data = resp.json()
        assert data["order_id"] == "TEST_HTTP_UNKNOWN"
        assert 0.0 <= data["return_risk_score"] <= 1.0
        assert len(data["reasons"]) > 0
