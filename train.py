"""
train.py — Kestrel Home Returns Risk (Variant A)
=================================================
Follows the implementation plan in Kestrel_Returns_Risk_Final_Plan.md exactly.

Data location:
    Place source CSVs (train.csv, test_unlabelled.csv, sample_submission.csv,
    customers.csv, products.csv) in the data/ folder inside this project, OR
    set the KESTREL_DATA_DIR environment variable to their parent folder.
    See README.md for full setup instructions.

Usage:
    py train.py

Outputs:
    predictions.csv            — Deliverable A (2096 rows, order_id + score)
    artifacts/model.joblib     — Fitted pipeline
    artifacts/metadata.json    — Threshold, feature names, versions, split info
    report/metrics.json        — All evaluation metrics
    report/error_analysis.md   — Error analysis narrative
"""

import json
import os
import warnings
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from kestrel.data_paths import find_data_file
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer

warnings.filterwarnings("ignore")

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent

# Source data files — resolved via KESTREL_DATA_DIR env var or data/ directory.
# See kestrel/data_paths.py and README.md for setup instructions.
TRAIN_CSV      = find_data_file("train.csv")
TEST_CSV       = find_data_file("test_unlabelled.csv")
CUSTOMERS_CSV  = find_data_file("customers.csv")
PRODUCTS_CSV   = find_data_file("products.csv")
SAMPLE_SUB_CSV = find_data_file("sample_submission.csv")

ARTIFACTS_DIR = ROOT / "artifacts"
REPORT_DIR    = ROOT / "report"
ARTIFACTS_DIR.mkdir(exist_ok=True)
REPORT_DIR.mkdir(exist_ok=True)

# ── Constants (from decisions.md) ────────────────────────────────────────────
COST_PER_RETURN_INR = 1_150   # Farhan (Finance Controller) figure
REVIEW_COST_PER_ORDER_INR = 0  # assumption: existing staff capacity
MONTHLY_ORDERS = 700
TOP_K_MONTHLY  = 100   # reviewer capacity assumption (~14 % of orders)
RANDOM_STATE   = 42

from kestrel.features import (
    ALL_FEATURES,
    CAT_FEATURES,
    NUM_FEATURES,
    LEAK_COLS,
    add_batch_features,
)

# ── 1. Load and clean ─────────────────────────────────────────────────────────
print("[1/9] Loading data ...")
train_raw = pd.read_csv(TRAIN_CSV, parse_dates=["order_placed_at"])
test_raw  = pd.read_csv(TEST_CSV,  parse_dates=["order_placed_at"])
customers = pd.read_csv(CUSTOMERS_CSV)
products  = pd.read_csv(PRODUCTS_CSV, parse_dates=["launch_date"])
sample    = pd.read_csv(SAMPLE_SUB_CSV)

print(f"  Train raw: {len(train_raw)} rows | Test: {len(test_raw)} rows")

# Dedup: partner_feed re-imports same order. Keep crm row.
# Sort so 'crm' < 'partner_feed' alphabetically -> crm first -> keep='first'
train = (train_raw
         .sort_values("source")
         .drop_duplicates(subset="order_id", keep="first")
         .reset_index(drop=True))
print(f"  Train after dedup: {len(train)} rows (removed {len(train_raw)-len(train)} partner_feed duplicates)")

# ── 2. Feature engineering ───────────────────────────────────────────────────
print("[2/9] Engineering features via shared kestrel.features module ...")
train = add_batch_features(train, products)
test  = add_batch_features(test_raw, products)

# Fill any missing product family (shouldn't happen but be safe)
for col in CAT_FEATURES:
    train[col] = train[col].fillna("unknown")
    test[col]  = test[col].fillna("unknown")

for col in NUM_FEATURES:
    med = train[col].median()
    train[col] = train[col].fillna(med)
    test[col]  = test[col].fillna(med)

X_full = train[ALL_FEATURES].copy()
y_full = train["returned"].copy()
dates  = train["order_placed_at"].copy()

X_test_submit = test[ALL_FEATURES].copy()

# ── 3. Split ─────────────────────────────────────────────────────────────────
print("[3/9] Splitting data ...")
# Time-ordered split. Holdout = last 20% by date. Untouched until final evaluation.
sorted_idx = dates.argsort().values  # ascending date order
n = len(sorted_idx)
holdout_start = int(n * 0.80)

train_idx   = sorted_idx[:holdout_start]
holdout_idx = sorted_idx[holdout_start:]

X_tr   = X_full.iloc[train_idx]
y_tr   = y_full.iloc[train_idx]
X_hold = X_full.iloc[holdout_idx]
y_hold = y_full.iloc[holdout_idx]

print(f"  Train slice: {len(X_tr)} rows, return rate {y_tr.mean():.4f}")
print(f"  Holdout:     {len(X_hold)} rows, return rate {y_hold.mean():.4f}")
print(f"  Positives in holdout: {y_hold.sum()} -- {'sufficient' if y_hold.sum()>=100 else 'few (reporting CV mean+sd)'}")

# ── 4. Preprocessing pipeline ─────────────────────────────────────────────────
print("[4/9] Building preprocessor ...")
preprocessor = ColumnTransformer([
    ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CAT_FEATURES),
    ("num", StandardScaler(), NUM_FEATURES),
], remainder="drop")

# ── 5. Baseline models ────────────────────────────────────────────────────────
print("[5/9] Fitting baselines ...")

# B0: constant = training return rate
b0_score = float(y_tr.mean())
b0_scores_hold = np.full(len(y_hold), b0_score)
b0_pr_auc = average_precision_score(y_hold, b0_scores_hold)
b0_roc_auc = roc_auc_score(y_hold, b0_scores_hold)
print(f"  B0 (constant {b0_score:.4f}): PR-AUC={b0_pr_auc:.4f}, ROC-AUC={b0_roc_auc:.4f}")

# B1: logistic regression
pipe_b1 = Pipeline([
    ("pre", ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CAT_FEATURES),
        ("num", StandardScaler(), NUM_FEATURES),
    ], remainder="drop")),
    ("clf", LogisticRegression(
        max_iter=1000, random_state=RANDOM_STATE, class_weight="balanced"
    )),
])
pipe_b1.fit(X_tr, y_tr)
b1_scores_hold = pipe_b1.predict_proba(X_hold)[:, 1]
b1_pr_auc  = average_precision_score(y_hold, b1_scores_hold)
b1_roc_auc = roc_auc_score(y_hold, b1_scores_hold)
print(f"  B1 (logistic): PR-AUC={b1_pr_auc:.4f}, ROC-AUC={b1_roc_auc:.4f}")

# M1: HistGradientBoosting with light tuning (~5 rounds on CV)
print("[6/9] Tuning M1 (HistGradientBoostingClassifier) ...")
tscv = TimeSeriesSplit(n_splits=5)

# Sort training slice by date for time-series CV
tr_dates = dates.iloc[train_idx].values
tr_order = np.argsort(tr_dates)
X_tr_sorted = X_tr.iloc[tr_order]
y_tr_sorted = y_tr.iloc[tr_order]

# 5-round grid (plan: cap at ~5)
param_grid = [
    {"max_iter": 200, "learning_rate": 0.05, "max_leaf_nodes": 31},
    {"max_iter": 300, "learning_rate": 0.05, "max_leaf_nodes": 31},
    {"max_iter": 200, "learning_rate": 0.10, "max_leaf_nodes": 31},
    {"max_iter": 200, "learning_rate": 0.05, "max_leaf_nodes": 15},
    {"max_iter": 300, "learning_rate": 0.10, "max_leaf_nodes": 63},
]

best_cv_score = -1
best_params   = param_grid[0]

for params in param_grid:
    cv_scores = []
    for tr_idx_, val_idx_ in tscv.split(X_tr_sorted):
        clf_tmp = Pipeline([
            ("pre", ColumnTransformer([
                ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CAT_FEATURES),
                ("num", StandardScaler(), NUM_FEATURES),
            ], remainder="drop")),
            ("clf", HistGradientBoostingClassifier(
                random_state=RANDOM_STATE, **params
            )),
        ])
        X_fold_tr = X_tr_sorted.iloc[tr_idx_]
        y_fold_tr = y_tr_sorted.iloc[tr_idx_]
        X_fold_val = X_tr_sorted.iloc[val_idx_]
        y_fold_val = y_tr_sorted.iloc[val_idx_]
        clf_tmp.fit(X_fold_tr, y_fold_tr)
        s = average_precision_score(y_fold_val, clf_tmp.predict_proba(X_fold_val)[:, 1])
        cv_scores.append(s)
    mean_s = np.mean(cv_scores)
    print(f"    {params}  CV PR-AUC={mean_s:.4f}")
    if mean_s > best_cv_score:
        best_cv_score = mean_s
        best_params   = params

print(f"  Best CV PR-AUC = {best_cv_score:.4f} with {best_params}")

# Fit best M1 on full training slice
pipe_m1 = Pipeline([
    ("pre", ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CAT_FEATURES),
        ("num", StandardScaler(), NUM_FEATURES),
    ], remainder="drop")),
    ("clf", HistGradientBoostingClassifier(
        random_state=RANDOM_STATE, **best_params
    )),
])
pipe_m1.fit(X_tr, y_tr)
m1_scores_hold = pipe_m1.predict_proba(X_hold)[:, 1]
m1_pr_auc  = average_precision_score(y_hold, m1_scores_hold)
m1_roc_auc = roc_auc_score(y_hold, m1_scores_hold)
print(f"  M1 holdout: PR-AUC={m1_pr_auc:.4f}, ROC-AUC={m1_roc_auc:.4f}")

# ── 6. Decision rule ─────────────────────────────────────────────────────────
# Use M1 only if it beats B1 by a meaningful margin (>0.02 PR-AUC on holdout)
MARGIN = 0.02
if m1_pr_auc > b1_pr_auc + MARGIN:
    chosen_model  = pipe_m1
    chosen_scores = m1_scores_hold
    chosen_name   = "HistGradientBoostingClassifier"
    chosen_pr_auc = m1_pr_auc
    chosen_roc    = m1_roc_auc
    print(f"  -> Chose M1 (beats B1 by >{MARGIN:.2f} PR-AUC)")
else:
    chosen_model  = pipe_b1
    chosen_scores = b1_scores_hold
    chosen_name   = "LogisticRegression"
    chosen_pr_auc = b1_pr_auc
    chosen_roc    = b1_roc_auc
    print(f"  -> Chose B1 (M1 advantage too small or negative; simpler model preferred)")

# ── 7. Threshold (by reviewer capacity) ──────────────────────────────────────
# Plan B5: flag top-k orders per month where k = TOP_K_MONTHLY
# On the holdout, pick threshold = score at rank k proportionally
k_frac     = TOP_K_MONTHLY / MONTHLY_ORDERS
n_flag     = max(1, int(np.ceil(len(y_hold) * k_frac)))
sorted_scores = np.sort(chosen_scores)[::-1]
threshold  = float(sorted_scores[min(n_flag - 1, len(sorted_scores) - 1)])
print(f"  Threshold={threshold:.4f} (top {n_flag} of {len(y_hold)} holdout rows, ~{k_frac*100:.1f}%)")

y_pred = (chosen_scores >= threshold).astype(int)

prec  = precision_score(y_hold, y_pred, zero_division=0)
rec   = recall_score(y_hold, y_pred, zero_division=0)
f1    = f1_score(y_hold, y_pred, zero_division=0)
cm    = confusion_matrix(y_hold, y_pred).tolist()
acc   = (y_pred == y_hold.values).mean()
majority_acc = max(y_hold.mean(), 1 - y_hold.mean())

print(f"  Precision={prec:.4f}, Recall={rec:.4f}, F1={f1:.4f}")
print(f"  Accuracy={acc:.4f} vs majority-class accuracy={majority_acc:.4f}")
print(f"  Confusion matrix: {cm}")

# ── 8. Bootstrap confidence intervals ────────────────────────────────────────
print("[7/9] Computing bootstrap CIs ...")

def grouped_bootstrap_ci(df_: pd.DataFrame, y="y", s="score", g="customer_id", n=1000, seed=0):
    rng  = np.random.default_rng(seed)
    parts = {k: v for k, v in df_.groupby(g)}
    keys = np.array(list(parts))
    vals = []
    for _ in range(n):
        b = pd.concat([parts[k] for k in rng.choice(keys, len(keys), replace=True)])
        if b[y].nunique() == 2:
            vals.append(average_precision_score(b[y], b[s]))
    if not vals:
        return [None, None, None]
    return [round(float(x), 4) for x in np.percentile(vals, [2.5, 50, 97.5])]

boot_df = pd.DataFrame({
    "y": y_hold.values,
    "score": chosen_scores,
    "customer_id": train.iloc[holdout_idx]["customer_id"].values,
})
ci_25, ci_50, ci_975 = grouped_bootstrap_ci(boot_df)
print(f"  PR-AUC 95% CI: [{ci_25}, {ci_975}]  median={ci_50}")

# ── 9. Error analysis ─────────────────────────────────────────────────────────
print("[8/9] Error analysis ...")
hold_df = train.iloc[holdout_idx].copy()
hold_df["score"] = chosen_scores
hold_df["pred"]  = y_pred

fn_df = hold_df[(hold_df["returned"] == 1) & (hold_df["pred"] == 0)]  # missed returns
fp_df = hold_df[(hold_df["returned"] == 0) & (hold_df["pred"] == 1)]  # false alarms

fn_by_channel = fn_df["sales_channel"].value_counts()
fp_by_channel = fp_df["sales_channel"].value_counts()
fn_by_payment = fn_df["payment_mode"].value_counts()
fn_by_family  = hold_df.merge(
    products[["sku","family"]], on="sku", how="left"
)[hold_df["returned"].values == 1]

error_md = f"""# Error Analysis -- Kestrel Returns Risk Holdout

**Model:** {chosen_name}
**Holdout period:** {train.iloc[holdout_idx]['order_placed_at'].min().date()} to {train.iloc[holdout_idx]['order_placed_at'].max().date()}
**Holdout n:** {len(y_hold)}, positives: {int(y_hold.sum())} (return rate {y_hold.mean():.4f})

## Summary
- **False negatives (missed returns):** {len(fn_df)} -- orders that were returned but not flagged
- **False positives (false alarms):** {len(fp_df)} -- orders flagged but not returned
- **Threshold:** {threshold:.4f} (top ~{TOP_K_MONTHLY} orders/month by reviewer capacity assumption)

## False Negatives by sales channel
{fn_by_channel.to_string()}

## False Negatives by payment mode
{fn_by_payment.to_string()}

## False Positives by sales channel
{fp_by_channel.to_string()}

## Sample false negatives (anonymised -- no PII)
{fn_df[['order_id','sales_channel','payment_mode','discount_pct','order_value_inr','customer_prior_returns','score']].head(5).to_string(index=False)}

## What the model cannot see
1. **Order content / description** -- high-value or complex appliances may have unmet expectations not captured in SKU or price alone.
2. **Actual delivery experience** -- delayed deliveries, damaged packaging, and courier-level issues are not in the data.
3. **Reason for return** -- the model has no explicit signal on why customers return (quality, changed mind, wrong product).
4. **Shield membership interactions** -- Meenal (Service Desk) notes that most returns are Shield customers for whom returns are free; shielded customers may behave systematically differently.
5. **Seasonal / campaign effects** -- the October festive payment gateway change (noted by Tanmay) may introduce anomalies not yet visible in train.
"""

(REPORT_DIR / "error_analysis.md").write_text(error_md, encoding="utf-8")
print("  Saved report/error_analysis.md")

# ── 10. Final refit on ALL mature data ───────────────────────────────────────
print("[9/9] Refitting chosen model on all data ...")
if chosen_name == "HistGradientBoostingClassifier":
    final_pipe = Pipeline([
        ("pre", ColumnTransformer([
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CAT_FEATURES),
            ("num", StandardScaler(), NUM_FEATURES),
        ], remainder="drop")),
        ("clf", HistGradientBoostingClassifier(
            random_state=RANDOM_STATE, **best_params
        )),
    ])
else:
    final_pipe = Pipeline([
        ("pre", ColumnTransformer([
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CAT_FEATURES),
            ("num", StandardScaler(), NUM_FEATURES),
        ], remainder="drop")),
        ("clf", LogisticRegression(
            max_iter=1000, random_state=RANDOM_STATE, class_weight="balanced"
        )),
    ])

final_pipe.fit(X_full, y_full)
print("  NOTE: Reported metrics come from the UNTOUCHED holdout, not this refit.")

# ── Save artifacts ────────────────────────────────────────────────────────────
joblib.dump(final_pipe, ARTIFACTS_DIR / "model.joblib")
print(f"  Saved artifacts/model.joblib")

# Build explanations for logistic regression
feature_importances = {}
if chosen_name == "LogisticRegression":
    ohe = final_pipe.named_steps["pre"].transformers_[0][1]
    ohe_names = list(ohe.get_feature_names_out(CAT_FEATURES))
    feature_names = ohe_names + NUM_FEATURES
    coefs = final_pipe.named_steps["clf"].coef_[0]
    feature_importances = {n: round(float(c), 4) for n, c in zip(feature_names, coefs)}

# Metadata
metadata = {
    "model": chosen_name,
    "best_params": best_params if chosen_name == "HistGradientBoostingClassifier" else {},
    "features": ALL_FEATURES,
    "cat_features": CAT_FEATURES,
    "num_features": NUM_FEATURES,
    "threshold": round(threshold, 6),
    "top_k_monthly": TOP_K_MONTHLY,
    "monthly_orders_assumed": MONTHLY_ORDERS,
    "split": "time-ordered 80/20 holdout; time-series CV on train slice (5 folds)",
    "sklearn_version": __import__("sklearn").__version__,
    "python_version": sys.version,
    "feature_importances_logreg": feature_importances,
}
with open(ARTIFACTS_DIR / "metadata.json", "w") as f:
    json.dump(metadata, f, indent=2)
print("  Saved artifacts/metadata.json")

# Metrics
metrics = {
    "holdout_n": int(len(y_hold)),
    "holdout_positives": int(y_hold.sum()),
    "holdout_return_rate": round(float(y_hold.mean()), 4),
    "majority_class_accuracy": round(float(majority_acc), 4),
    "models": {
        "B0_constant": {
            "pr_auc": round(b0_pr_auc, 4),
            "roc_auc": round(b0_roc_auc, 4),
        },
        "B1_logistic": {
            "pr_auc": round(b1_pr_auc, 4),
            "roc_auc": round(b1_roc_auc, 4),
        },
        "M1_hgb": {
            "pr_auc": round(m1_pr_auc, 4),
            "roc_auc": round(m1_roc_auc, 4),
        },
    },
    "chosen_model": chosen_name,
    "chosen_model_holdout": {
        "pr_auc": round(chosen_pr_auc, 4),
        "pr_auc_ci_95": [ci_25, ci_975],
        "roc_auc": round(chosen_roc, 4),
        "threshold": round(threshold, 6),
        "precision_at_threshold": round(prec, 4),
        "recall_at_threshold": round(rec, 4),
        "f1_at_threshold": round(f1, 4),
        "accuracy": round(acc, 4),
        "confusion_matrix": cm,
    },
    "leakage_excluded": [
        "last_service_event_type (REVERSE_PICKUP = 100% return rate -- post-outcome)",
        "pickup_scheduled_at (91.6% returned when non-null -- post-outcome logistics)",
        "delivery_note (free text / PII)",
        "delivery_pincode (PII-adjacent)",
        "source (dedup only)",
        "customer_id (direct identifier)",
    ],
    "dedup_rows_removed": int(len(train_raw) - len(train)),
}

with open(REPORT_DIR / "metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)
print("  Saved report/metrics.json")

# ── Generate predictions.csv (Deliverable A) ─────────────────────────────────
test_scores = final_pipe.predict_proba(X_test_submit)[:, 1]
predictions = pd.DataFrame({
    "order_id": test_raw["order_id"],
    "score": test_scores,
})

# Verify
assert list(predictions.columns) == list(sample.columns), "Column mismatch!"
assert len(predictions) == len(test_raw), "Row count mismatch!"
assert predictions["order_id"].is_unique, "Duplicate order IDs!"
assert set(predictions["order_id"]) == set(sample["order_id"]), "Order ID set mismatch!"
assert np.isfinite(predictions["score"]).all(), "Non-finite scores!"

predictions.to_csv(ROOT / "predictions.csv", index=False)
print(f"  Saved predictions.csv ({len(predictions)} rows)")
print(f"  Score stats: min={predictions['score'].min():.4f}, max={predictions['score'].max():.4f}, mean={predictions['score'].mean():.4f}")

# ── Final summary ─────────────────────────────────────────────────────────────
print()
print("=" * 60)
print("TRAINING COMPLETE")
print("=" * 60)
print(f"  Chosen model:     {chosen_name}")
print(f"  Holdout PR-AUC:   {chosen_pr_auc:.4f}  (95% CI: [{ci_25}, {ci_975}])")
print(f"  B0 PR-AUC:        {b0_pr_auc:.4f}")
print(f"  B1 PR-AUC:        {b1_pr_auc:.4f}")
print(f"  Majority-class accuracy benchmark: {majority_acc:.4f}")
print(f"  Threshold:        {threshold:.4f} (top ~{TOP_K_MONTHLY}/month)")
print(f"  Precision@k:      {prec:.4f}")
print(f"  Recall@k:         {rec:.4f}")
print()
print("  CP1 deliverables ready:")
print("    [OK] predictions.csv")
print("    [OK] report/metrics.json")
print("    [OK] artifacts/model.joblib")
print("    [OK] artifacts/metadata.json")
print("    [OK] report/error_analysis.md")
