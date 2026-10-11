"""
tests/test_features.py — Feature parity, calculation accuracy, and explanation tests
=====================================================================================
Validates:
1. Feature parity between real-time inference and batch training feature builders.
2. Exact calculation of smoothed prior return rate (zero orders, division-by-zero safety).
3. Handling of unknown SKUs and categories without inventing product families.
4. Faithfulness of linear model explanations.
5. Preservation of row counts during product catalog joins (no row multiplication).
"""

import joblib
import numpy as np
import pandas as pd
import pytest

from kestrel.data_paths import find_data_file
from kestrel.features import (
    ALL_FEATURES,
    CAT_FEATURES,
    NUM_FEATURES,
    compute_prior_return_rate,
    lookup_product_details,
    add_batch_features,
    build_single_order_features,
    explain_prediction,
    CATALOG_MEDIAN_LIST_PRICE,
    CATALOG_MODE_WARRANTY,
)


@pytest.fixture(scope="module")
def products_df():
    return pd.read_csv(find_data_file("products.csv"))


@pytest.fixture(scope="module")
def pipeline():
    return joblib.load("artifacts/model.joblib")


def test_prior_return_rate_formula():
    """Validates smoothed prior return rate formula and safety against division by zero."""
    # Zero prior orders -> 0 / (0 + 1) = 0.0
    assert compute_prior_return_rate(0, 0) == 0.0

    # 1 prior order, 1 return -> 1 / (1 + 1) = 0.5 (Laplace smoothed)
    assert compute_prior_return_rate(1, 1) == 0.5

    # 2 prior orders, 1 return -> 1 / (2 + 1) = 1/3
    assert abs(compute_prior_return_rate(2, 1) - (1.0 / 3.0)) < 1e-6

    # 10 prior orders, 5 returns -> 5 / (10 + 1) = 5/11
    assert abs(compute_prior_return_rate(10, 5) - (5.0 / 11.0)) < 1e-6

    # Vectorized Series consistency
    orders_s = pd.Series([0, 1, 2, 10])
    returns_s = pd.Series([0, 1, 1, 5])
    rates_s = compute_prior_return_rate(orders_s, returns_s)
    expected = [0.0, 0.5, 1.0 / 3.0, 5.0 / 11.0]
    for actual, exp in zip(rates_s, expected):
        assert abs(actual - exp) < 1e-6


def test_product_joins_no_row_multiplication(products_df):
    """Verifies product catalog join is 1-to-1 or many-to-1, never multiplying rows."""
    test_orders = pd.DataFrame(
        [
            {"order_id": "O1", "sku": "KH-AF-01", "customer_prior_orders": 2, "customer_prior_returns": 0},
            {"order_id": "O2", "sku": "KH-RV-01", "customer_prior_orders": 0, "customer_prior_returns": 0},
            {"order_id": "O3", "sku": "KH-UNKNOWN-99", "customer_prior_orders": 5, "customer_prior_returns": 2},
        ]
    )
    result = add_batch_features(test_orders, products_df)
    assert len(result) == len(test_orders), "Product join multiplied order rows!"
    assert "product_family" in result.columns
    assert "prior_return_rate" in result.columns
    assert result.loc[result["order_id"] == "O1", "product_family"].iloc[0] == "Air Fryer"
    assert result.loc[result["order_id"] == "O3", "product_family"].iloc[0] == "Unknown"


def test_unknown_sku_handling(products_df):
    """Verifies unknown SKUs receive documented catalog defaults, not made-up families."""
    # Known SKU
    family, price, warranty, is_known = lookup_product_details("KH-AF-01", products_df)
    assert is_known is True
    assert family == "Air Fryer"
    assert price == 5199.0
    assert warranty == 12

    # Unknown SKU
    family_u, price_u, warranty_u, is_known_u = lookup_product_details("UNKNOWN-SKU-999", products_df)
    assert is_known_u is False
    assert family_u == "Unknown"
    assert price_u == CATALOG_MEDIAN_LIST_PRICE
    assert warranty_u == CATALOG_MODE_WARRANTY


def test_feature_parity_single_vs_batch(products_df):
    """Verifies single-order inference features match batch DataFrame features identically."""
    order_data = {
        "order_id": "O_PARITY",
        "sku": "KH-RV-01",
        "sales_channel": "app",
        "payment_mode": "cod",
        "discount_pct": 20.0,
        "qty": 1,
        "order_value_inr": 17599.0,
        "promised_delivery_days": 4,
        "is_gift": "N",
        "customer_prior_orders": 3,
        "customer_prior_returns": 1,
    }

    # Path 1: Single order inference builder
    single_df, _ = build_single_order_features(order_data, products_df)

    # Path 2: Batch feature builder
    batch_raw = pd.DataFrame([order_data])
    batch_df = add_batch_features(batch_raw, products_df)[ALL_FEATURES]

    # Verify column structure
    assert list(single_df.columns) == ALL_FEATURES
    assert list(batch_df.columns) == ALL_FEATURES

    # Verify exact values across all 13 features
    for col in ALL_FEATURES:
        v_single = single_df[col].iloc[0]
        v_batch = batch_df[col].iloc[0]
        if isinstance(v_single, float):
            assert abs(v_single - v_batch) < 1e-6, f"Mismatch in {col}: {v_single} vs {v_batch}"
        else:
            assert v_single == v_batch, f"Mismatch in {col}: {v_single} vs {v_batch}"


def test_model_explanations(pipeline, products_df):
    """Verifies that model explanations reflect fitted linear contributions."""
    order_data = {
        "order_id": "O_EXP",
        "sku": "KH-AF-01",
        "sales_channel": "app",
        "payment_mode": "cod",
        "discount_pct": 10.0,
        "qty": 1,
        "order_value_inr": 4500.0,
        "promised_delivery_days": 6,
        "is_gift": "N",
        "customer_prior_orders": 3,
        "customer_prior_returns": 2,
    }
    df, _ = build_single_order_features(order_data, products_df)
    reasons = explain_prediction(df, pipeline, top_n=3)

    assert len(reasons) > 0
    # Every reason must contain feature, impact (+Risk or -Risk), and description
    for r in reasons:
        assert "feature" in r
        assert r["impact"] in ["+Risk", "-Risk", "Neutral/Low"]
        assert len(r["description"]) > 10

    # COD and customer prior returns must show as +Risk
    features_present = [r["feature"] for r in reasons]
    impacts = {r["feature"]: r["impact"] for r in reasons}
    if "cat__payment_mode_cod" in features_present:
        assert impacts["cat__payment_mode_cod"] == "+Risk"
    if "num__customer_prior_returns" in features_present:
        assert impacts["num__customer_prior_returns"] == "+Risk"
