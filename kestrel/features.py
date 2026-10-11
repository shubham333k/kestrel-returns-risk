"""
kestrel/features.py — Centralised Feature Engineering & Explanations
=====================================================================
Ensures 100% parity between training, batch predictions, and API inference.

Key Design Decisions:
1. Prior Return Rate:
   Formula: prior_returns / (prior_orders + 1.0)
   Rationale: Laplace / additive prior smoothing. When prior_orders == 0, returns 0.0.
   Prevents single-order customers (1 order, 1 return) from receiving an extreme 1.0
   score compared to multi-order customers, while naturally avoiding division-by-zero.

2. Product Lookups & Unknown SKUs:
   Known SKUs: looked up from products catalog.
   Unknown SKUs: mapped to documented training-derived baselines:
     - product_family = "Unknown" (handled via OneHotEncoder handle_unknown='ignore')
     - list_price_inr = 4723.0 (catalog median list price)
     - warranty_months = 12 (catalog mode warranty)
   No made-up specific product families (e.g. Air Fryer) are invented.

3. Model-Faithful Linear Explanations:
   For Logistic Regression, computes exact feature contributions:
     contribution_i = z_i * w_i
   where z_i is the standardized/one-hot value and w_i is the fitted coefficient.
   Contributions are sorted by magnitude and labeled as +Risk or -Risk.

4. Excluded Leakage Columns:
   - last_service_event_type (POST-OUTCOME: REVERSE_PICKUP = 100% return)
   - pickup_scheduled_at (POST-OUTCOME: reverse logistics timestamp)
   - source (used only for deduplication)
   - delivery_note (free-text PII, 26% missing)
   - delivery_pincode (PII-adjacent, default 000000, not geocoded)
   - customer_id (raw ID; aggregated prior stats used instead)
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd

# ── Feature Definitions ───────────────────────────────────────────────────────
CAT_FEATURES = ["sales_channel", "payment_mode", "is_gift", "product_family"]
NUM_FEATURES = [
    "discount_pct",
    "qty",
    "order_value_inr",
    "promised_delivery_days",
    "customer_prior_orders",
    "customer_prior_returns",
    "prior_return_rate",
    "list_price_inr",
    "warranty_months",
]
ALL_FEATURES = CAT_FEATURES + NUM_FEATURES

LEAK_COLS = [
    "last_service_event_type",  # POST-OUTCOME leak
    "pickup_scheduled_at",      # POST-OUTCOME leak
    "source",                   # dedup marker only
    "delivery_note",            # free text / PII risk
    "delivery_pincode",         # PII-adjacent, un-geocoded
    "customer_id",              # raw identifier
    "returned",                 # target label
    "order_id",                 # transaction ID
    "order_placed_at",          # temporal ordering only
]

# Catalog fallback defaults for uncatalogued / unknown SKUs
CATALOG_MEDIAN_LIST_PRICE = 4723.0
CATALOG_MODE_WARRANTY = 12


def compute_prior_return_rate(
    prior_orders: Union[int, float, pd.Series, np.ndarray],
    prior_returns: Union[int, float, pd.Series, np.ndarray],
) -> Union[float, pd.Series, np.ndarray]:
    """Computes the smoothed prior return rate: prior_returns / (prior_orders + 1.0).

    Safe against division-by-zero, shrinks small-sample extremes towards 0.
    """
    if isinstance(prior_orders, pd.Series) or isinstance(prior_orders, np.ndarray):
        return prior_returns / (prior_orders + 1.0)
    orders = float(prior_orders)
    returns = float(prior_returns)
    return returns / (orders + 1.0)


def lookup_product_details(
    sku: str,
    products_df: pd.DataFrame,
) -> Tuple[str, float, int, bool]:
    """Looks up SKU in product catalog.

    Returns:
        (product_family, list_price_inr, warranty_months, is_known_sku)
    """
    if products_df is not None and not products_df.empty:
        sku_row = products_df[products_df["sku"] == sku]
        if len(sku_row) > 0:
            row = sku_row.iloc[0]
            return (
                str(row["family"]),
                float(row["list_price_inr"]),
                int(row["warranty_months"]),
                True,
            )

    # Documented training-derived fallback for uncatalogued SKUs
    return (
        "Unknown",
        CATALOG_MEDIAN_LIST_PRICE,
        CATALOG_MODE_WARRANTY,
        False,
    )


def add_batch_features(
    orders_df: pd.DataFrame,
    products_df: pd.DataFrame,
) -> pd.DataFrame:
    """Adds product features and smoothed prior return rate to an orders DataFrame.

    Used by train.py for both training and batch test scoring.
    """
    df = orders_df.copy()
    # Merge product catalog
    df = df.merge(
        products_df[["sku", "family", "list_price_inr", "warranty_months"]].rename(
            columns={"family": "product_family"}
        ),
        on="sku",
        how="left",
    )
    # Fill any uncatalogued SKU with documented defaults
    df["product_family"] = df["product_family"].fillna("Unknown")
    df["list_price_inr"] = df["list_price_inr"].fillna(CATALOG_MEDIAN_LIST_PRICE)
    df["warranty_months"] = df["warranty_months"].fillna(CATALOG_MODE_WARRANTY)

    # Smoothed prior return rate
    df["prior_return_rate"] = compute_prior_return_rate(
        df["customer_prior_orders"],
        df["customer_prior_returns"],
    )
    return df


def build_single_order_features(
    order_data: Dict[str, Any],
    products_df: pd.DataFrame,
) -> Tuple[pd.DataFrame, bool]:
    """Builds a single-row DataFrame matching ALL_FEATURES for real-time inference.

    Returns:
        (features_df, is_known_sku)
    """
    sku = str(order_data.get("sku", ""))
    family, list_price, warranty, is_known = lookup_product_details(sku, products_df)

    prior_orders = int(order_data.get("customer_prior_orders", 0))
    prior_returns = int(order_data.get("customer_prior_returns", 0))
    prior_rate = compute_prior_return_rate(prior_orders, prior_returns)

    row = {
        "sales_channel": str(order_data.get("sales_channel", "app")),
        "payment_mode": str(order_data.get("payment_mode", "cod")),
        "is_gift": str(order_data.get("is_gift", "N")),
        "product_family": family,
        "discount_pct": float(order_data.get("discount_pct", 0.0)),
        "qty": int(order_data.get("qty", 1)),
        "order_value_inr": float(order_data.get("order_value_inr", 0.0)),
        "promised_delivery_days": int(order_data.get("promised_delivery_days", 3)),
        "customer_prior_orders": prior_orders,
        "customer_prior_returns": prior_returns,
        "prior_return_rate": float(prior_rate),
        "list_price_inr": float(list_price),
        "warranty_months": int(warranty),
    }

    df = pd.DataFrame([row], columns=ALL_FEATURES)
    return df, is_known


def explain_prediction(
    features_df: pd.DataFrame,
    pipeline: Any,
    top_n: int = 3,
) -> List[Dict[str, str]]:
    """Computes exact linear contributions (z_i * w_i) from Logistic Regression pipeline.

    If pipeline does not support linear breakdown, returns clear general risk indicators.
    """
    reasons: List[Dict[str, str]] = []

    try:
        if (
            hasattr(pipeline, "named_steps")
            and "pre" in pipeline.named_steps
            and "clf" in pipeline.named_steps
        ):
            pre = pipeline.named_steps["pre"]
            clf = pipeline.named_steps["clf"]

            if hasattr(clf, "coef_") and hasattr(pre, "get_feature_names_out"):
                feature_names = list(pre.get_feature_names_out())
                coefs = clf.coef_[0]

                # Transform single row
                x_trans = pre.transform(features_df)
                if hasattr(x_trans, "toarray"):
                    x_trans = x_trans.toarray()
                x_row = x_trans[0]

                # Compute contribution per transformed column
                raw_contribs = []
                for name, val, coef in zip(feature_names, x_row, coefs):
                    contrib = float(val * coef)
                    if abs(contrib) > 0.005:
                        raw_contribs.append((name, val, coef, contrib))

                # Sort by absolute contribution magnitude
                raw_contribs.sort(key=lambda item: abs(item[3]), reverse=True)

                readable_names = {
                    "cat__payment_mode_cod": "Payment Mode: COD",
                    "cat__payment_mode_prepaid_upi": "Payment Mode: Prepaid UPI",
                    "cat__payment_mode_prepaid_card": "Payment Mode: Prepaid Card",
                    "cat__payment_mode_emi": "Payment Mode: EMI",
                    "cat__sales_channel_marketplace": "Sales Channel: Marketplace",
                    "cat__sales_channel_partner_outlet": "Sales Channel: Partner Outlet",
                    "cat__sales_channel_app": "Sales Channel: Mobile App",
                    "cat__sales_channel_web": "Sales Channel: Website",
                    "cat__is_gift_Y": "Gift Order: Yes",
                    "cat__is_gift_N": "Gift Order: No",
                    "cat__product_family_Robot Vacuum": "Product: Robot Vacuum",
                    "cat__product_family_Water Purifier": "Product: Water Purifier",
                    "cat__product_family_Induction Cooktop": "Product: Induction Cooktop",
                    "cat__product_family_Air Fryer": "Product: Air Fryer",
                    "cat__product_family_Ceiling Fan": "Product: Ceiling Fan",
                    "cat__product_family_Mixer Grinder": "Product: Mixer Grinder",
                    "cat__product_family_Room Heater": "Product: Room Heater",
                    "num__customer_prior_returns": "Customer Prior Returns",
                    "num__customer_prior_orders": "Customer Prior Orders",
                    "num__promised_delivery_days": "Promised Delivery Days",
                    "num__discount_pct": "Discount Percentage",
                    "num__order_value_inr": "Order Value (INR)",
                    "num__warranty_months": "Warranty Duration",
                    "num__prior_return_rate": "Prior Return Rate",
                    "num__list_price_inr": "Catalog List Price",
                    "num__qty": "Order Quantity",
                }

                for col_name, val, coef, contrib in raw_contribs[:top_n]:
                    impact_label = "+Risk" if contrib > 0 else "-Risk"
                    feature_title = readable_names.get(col_name, col_name)

                    if contrib > 0:
                        desc = f"{feature_title} increases return risk (+{contrib:.2f} log-odds impact in model)."
                    else:
                        desc = f"{feature_title} reduces return risk ({contrib:.2f} log-odds impact in model)."

                    reasons.append(
                        {
                            "feature": col_name,
                            "impact": impact_label,
                            "description": desc,
                        }
                    )
    except Exception:
        # Fall back to general risk indicators if inspection fails
        pass

    if not reasons:
        row = features_df.iloc[0]
        if row.get("payment_mode") == "cod":
            reasons.append(
                {
                    "feature": "payment_mode",
                    "impact": "+Risk",
                    "description": "General Risk Indicator: COD orders historically have elevated return rates.",
                }
            )
        if row.get("customer_prior_returns", 0) > 0:
            reasons.append(
                {
                    "feature": "customer_prior_returns",
                    "impact": "+Risk",
                    "description": f"General Risk Indicator: Customer has {int(row['customer_prior_returns'])} previous return(s).",
                }
            )
        if not reasons:
            reasons.append(
                {
                    "feature": "order_profile",
                    "impact": "Neutral/Low",
                    "description": "Order features are consistent with standard baseline return risk.",
                }
            )

    return reasons
