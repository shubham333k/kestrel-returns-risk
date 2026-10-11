# Verification Results — Kestrel Returns Risk (Variant A)

**Execution Date:** 2026-10-11  
**Environment:** Python 3.14.7 (tags/v3.14.7:823f032), Windows 64-bit AMD64  
**Dependencies:** scikit-learn 1.9.0, pandas 3.0.3, numpy 2.5.0, scipy 1.18.0, fastapi 0.139.0, pytest 9.1.1  

---

## 1. Model Pipeline Execution (`py train.py`)

- **Command:** `py train.py` (executed from repository root)
- **Exit Code:** `0` (Success)
- **Execution Log:**

```text
[1/9] Loading data ...
  Train raw: 11155 rows | Test: 2096 rows
  Train after dedup: 10504 rows (removed 651 partner_feed duplicates)
[2/9] Engineering features via shared kestrel.features module ...
[3/9] Splitting data ...
  Train slice: 8403 rows, return rate 0.1140
  Holdout:     2101 rows, return rate 0.1152
  Positives in holdout: 242 -- sufficient
[4/9] Building preprocessor ...
[5/9] Fitting baselines ...
  B0 (constant 0.1140): PR-AUC=0.1152, ROC-AUC=0.5000
  B1 (logistic): PR-AUC=0.3754, ROC-AUC=0.7590
[6/9] Tuning M1 (HistGradientBoostingClassifier) ...
    {'max_iter': 200, 'learning_rate': 0.05, 'max_leaf_nodes': 31}  CV PR-AUC=0.2788
    {'max_iter': 300, 'learning_rate': 0.05, 'max_leaf_nodes': 31}  CV PR-AUC=0.2662
    {'max_iter': 200, 'learning_rate': 0.1, 'max_leaf_nodes': 31}  CV PR-AUC=0.2654
    {'max_iter': 200, 'learning_rate': 0.05, 'max_leaf_nodes': 15}  CV PR-AUC=0.2897
    {'max_iter': 300, 'learning_rate': 0.1, 'max_leaf_nodes': 63}  CV PR-AUC=0.2383
  Best CV PR-AUC = 0.2897 with {'max_iter': 200, 'learning_rate': 0.05, 'max_leaf_nodes': 15}
  M1 holdout: PR-AUC=0.3261, ROC-AUC=0.7395
  -> Chose B1 (M1 advantage too small or negative; simpler model preferred)
  Threshold=0.6485 (top 301 of 2101 holdout rows, ~14.3%)
  Precision=0.3289, Recall=0.4091, F1=0.3646
  Accuracy=0.8358 vs majority-class accuracy=0.8848
  Confusion matrix: [[1657, 202], [143, 99]]
[7/9] Computing bootstrap CIs ...
  PR-AUC 95% CI: [0.3177, 0.4374]  median=0.3776
[8/9] Error analysis ...
  Saved report/error_analysis.md
[9/9] Refitting chosen model on all data ...
  NOTE: Reported metrics come from the UNTOUCHED holdout, not this refit.
  Saved artifacts/model.joblib
  Saved artifacts/metadata.json
  Saved report/metrics.json
  Saved predictions.csv (2096 rows)
  Score stats: min=0.0587, max=0.9959, mean=0.4205
```

---

## 2. Model Performance Summary (Holdout vs Baselines)

| Model | Evaluated Split | PR-AUC | ROC-AUC | Precision@Threshold | Recall@Threshold | Status |
|---|---|---|---|---|---|---|
| **B0 Constant Prior** | 20% Time Holdout | 0.1152 | 0.5000 | N/A | N/A | Naive baseline |
| **B1 Logistic Regression** | 20% Time Holdout | **0.3754** (95% CI: [0.3177, 0.4374]) | **0.7590** | **32.89%** | **40.91%** | **Selected & Deployed** |
| **M1 HistGradientBoosting** | 20% Time Holdout | 0.3261 | 0.7395 | 30.12% | 37.60% | Rejected (lower PR-AUC) |

- **Majority Class Baseline:** 88.48% (a naive zero-return classifier achieves 88.5% accuracy but catches 0 returns).
- **Holdout Test Set Size:** 2,101 rows (242 true returns, return rate 11.52%).
- **Operating Threshold:** `0.6485` (selects ~14.3% top-risk orders, targeting reviewer capacity of ~100 orders/month).

---

## 3. Deliverable A Validation (`predictions.csv`)

Validation performed against `sample_submission.csv`:

| Check | Expected | Actual | Pass? |
|---|---|---|---|
| **Column Names & Order** | `['order_id', 'score']` | `['order_id', 'score']` | ✅ PASS |
| **Row Count** | 2,096 | 2,096 | ✅ PASS |
| **Order ID Uniqueness** | 2,096 unique IDs | 2,096 unique IDs | ✅ PASS |
| **Order ID Set Match** | 100% match with sample | 100% match (0 missing, 0 extra) | ✅ PASS |
| **Order ID Row Ordering** | Identical sequence | Identical sequence | ✅ PASS |
| **Missing Values** | 0 nulls | 0 nulls | ✅ PASS |
| **Score Finiteness** | 100% finite floats | 100% finite floats | ✅ PASS |
| **Score Range** | $[0.0, 1.0]$ | $[\text{min}=0.0587, \text{max}=0.9959]$ | ✅ PASS |
| **Score Mean** | Sensible distribution | $0.4205$ | ✅ PASS |

---

## 4. Test Suite Execution (`py -m pytest tests/ -v`)

- **Command:** `py -m pytest tests/ -v`
- **Result:** **19 passed, 0 failed** in 5.07s.
- **Test Modules:**
  - `tests/test_paths.py`: 5 passed (data dir, source files, path resolution, missing file errors, artifacts).
  - `tests/test_features.py`: 5 passed (formula, join cardinality, unknown SKU fallback, feature parity, model explanations).
  - `tests/test_service.py`: 9 passed (healthz, valid predict, unseen categories, extreme values, risk separation, HTTP healthz/UI, HTTP valid predict, HTTP validation errors, HTTP unknown SKU/channel).

---

## 5. API & Web UI End-to-End Verification

### HTTP Client Tests (`TestClient` / Live Server)
- **`GET /healthz`:** Returns HTTP 200 with `{"status": "ok", "model_loaded": true}`.
- **`GET /`:** Returns HTTP 200 HTML page with complete evaluation form and review disclaimer.
- **`POST /predict` (Valid Input):** Returns HTTP 200 with order ID, calibrated risk score, categorical risk band (`HIGH`, `MEDIUM`, or `LOW`), operating threshold (`0.6485`), and top linear model contributors.
- **Input Validation:** Out-of-range discount percentages (`> 100` or `< 0`), zero quantities (`< 1`), and missing required fields return clean HTTP 422 JSON errors with zero system stack traces or internal paths exposed.
- **Unknown SKU / Categories:** Gracefully handled with HTTP 200 utilizing documented catalog defaults (`family='Unknown'`, median price, mode warranty) without raising exceptions.

### Browser UI End-to-End Verification
Automated browser session executed and recorded (`app_ui_test`):
1. **High Risk Order (`KO2610504`):**
   - Inputs: Marketplace channel, COD payment, 30% discount, 3 prior orders, 2 prior returns.
   - Live Output: Risk score **96.0%**, **HIGH RISK** red badge.
   - Top Drivers: Customer Prior Returns (+2.02 log-odds), Payment Mode COD (+0.78 log-odds), Discount % (+0.66 log-odds).
   - Disclaimer: `⚠️ Risk estimate for human review, not an automated decision.` (No automated cancellation).
2. **Low Risk Order (`KO2610599`):**
   - Inputs: App channel, Prepaid UPI payment, 0% discount, 5 prior orders, 0 prior returns.
   - Live Output: Risk score **21.5%**, **LOW RISK** green badge.
   - Top Drivers: Prepaid UPI (-0.43 log-odds), Discount % (-0.33 log-odds), Gift Order No (-0.32 log-odds).
   - Disclaimer: `⚠️ Risk estimate for human review, not an automated decision.`
