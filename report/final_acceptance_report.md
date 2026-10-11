# Final Acceptance Report — Kestrel Returns Risk (Variant A)

**Audit Date:** 2026-10-11  
**Project:** Kestrel Home — Returns Risk Model & Decision System  
**Repository:** `https://github.com/shubham333k/kestrel-returns-risk`  
**Execution Environment:** Python 3.14.7, Windows 64-bit AMD64  
**Audit Objective:** Independent, evidence-based verification against all task requirements and the baseline audit (`notes/final_audit.md`).

---

## 1. Requirement Traceability & Verification Status

| Requirement / Item | Description | Evidence Path / Command | Exit Code / Result | Status |
|---|---|---|---|---|
| **Deliverable A: `predictions.csv`** | 2,096 predictions for `test_unlabelled.csv` matching `sample_submission.csv` format | [`predictions.csv`](../predictions.csv), assertion checks | 8 assertions passed (exact columns, length, uniqueness, finite, [0, 1] bounds) | **PASS** |
| **Deliverable B: Executive Memo** | Executive Memo for Head of D2C Operations at repository root | [`memo.md`](../memo.md), [`report/memo.md`](memo.md) | Formatted in clean UTF-8, ₹1,150 cost, gross vs net pilot savings, no auto-hold | **PASS** |
| **Centralised Data Discovery** | Portable data paths without UUID hardcoding or silent fallbacks | [`kestrel/data_paths.py`](../kestrel/data_paths.py), [`tests/test_paths.py`](../tests/test_paths.py) | 5 tests passed | **PASS** |
| **Feature Logic Parity** | 100% parity between batch training and real-time inference | [`kestrel/features.py`](../kestrel/features.py), [`tests/test_features.py`](../tests/test_features.py) | 5 tests passed | **PASS** |
| **Smoothed Prior Return Rate** | Safe Laplace-smoothed rate ($\text{returns}/(\text{orders} + 1.0)$) | [`kestrel/features.py`](../kestrel/features.py), [`notes/decisions.md`](../notes/decisions.md) | Unit tests verified (0/0=0.0, 1/1=0.5, zero-division safe) | **PASS** |
| **Unknown SKU / Category Fallback** | Documented catalog baselines (`family='Unknown'`), no invented families | [`kestrel/features.py`](../kestrel/features.py), [`tests/test_features.py`](../tests/test_features.py) | Verified fallback to catalog median list price (₹4,723) and mode warranty (12m) | **PASS** |
| **Data Leakage Elimination** | Excluded `last_service_event_type`, `pickup_scheduled_at`, PII | [`train.py`](../train.py), [`kestrel/features.py`](../kestrel/features.py) | Post-outcome columns strictly excluded from feature sets | **PASS** |
| **Time-Ordered Evaluation** | Untouched 20% holdout split, 5-fold TimeSeriesCV on training portion | [`train.py`](../train.py), [`report/metrics.json`](metrics.json) | Evaluated on 2,101 holdout rows (PR-AUC 0.3754 vs baseline 0.1152) | **PASS** |
| **Faithful Model Explanations** | Exact linear contributions ($w_i \cdot z_i$) from fitted coefficients | [`kestrel/features.py`](../kestrel/features.py), [`service/app.py`](../service/app.py) | Tested live in API & UI (positive/negative log-odds impact) | **PASS** |
| **FastAPI Service & Web UI** | Real-time prediction endpoint, `/healthz`, and interactive UI | [`service/app.py`](../service/app.py), `http://localhost:8000` | HTTP 200, clean 422 JSON validation, human safeguard note | **PASS** |
| **End-to-End Automated Tests** | Unit, feature, path, and HTTP TestClient test suite | [`tests/`](../tests/), `py -m pytest tests/ -v` | **19 passed, 0 failed** in 5.96s | **PASS** |
| **Dependencies & Clean Setup** | Pinned root `requirements.txt` with tested version bounds | [`requirements.txt`](../requirements.txt), [`README.md`](../README.md) | Dry-run install resolved 100% cleanly (0 conflicts) | **PASS** |
| **Repository Cleanliness** | Raw datasets excluded via `.gitignore`; deliverables tracked | [`.gitignore`](../.gitignore), `git status` | Source CSVs ignored, 0 untracked data files | **PASS** |
| **Video Walkthrough** | 3-minute screen walkthrough without slides | [`notes/recording_script.md`](../notes/recording_script.md), browser recording artifact | Script & observed scores verified; human must record live and upload | **NOT VERIFIED (Manual Action)** |
| **Portal Form Submission** | Candidate form responses mapped and verified | [`notes/form-answers.md`](../notes/form-answers.md) | Form answers drafted & checked; human must submit on portal | **NOT VERIFIED (Manual Action)** |

---

## 2. Command Execution Evidence Log

### A. Environment Dependency Check
- **Command:** `py -m pip install -r requirements.txt --dry-run`
- **Exit Code:** `0`
- **Output:** All dependencies satisfied (`pandas 3.0.3`, `numpy 2.5.0`, `scipy 1.18.0`, `scikit-learn 1.9.0`, `joblib 1.5.3`, `fastapi 0.139.0`, `uvicorn 0.50.2`, `pydantic 2.13.4`, `pytest 9.1.1`, `httpx 0.28.1`). Zero version conflicts.

### B. Automated Test Suite Execution
- **Command:** `py -m pytest tests/ -v`
- **Exit Code:** `0`
- **Output:**
  ```text
  tests/test_features.py::test_prior_return_rate_formula PASSED            [  5%]
  tests/test_features.py::test_product_joins_no_row_multiplication PASSED  [ 10%]
  tests/test_features.py::test_unknown_sku_handling PASSED                 [ 15%]
  tests/test_features.py::test_feature_parity_single_vs_batch PASSED       [ 21%]
  tests/test_features.py::test_model_explanations PASSED                   [ 26%]
  tests/test_paths.py::test_data_dir_exists PASSED                         [ 31%]
  tests/test_paths.py::test_required_data_files_present PASSED             [ 36%]
  tests/test_paths.py::test_find_data_file_returns_path PASSED             [ 42%]
  tests/test_paths.py::test_find_data_file_raises_on_missing PASSED        [ 47%]
  tests/test_paths.py::test_artifacts_present PASSED                       [ 52%]
  tests/test_service.py::test_healthz PASSED                               [ 57%]
  tests/test_service.py::test_valid_predict PASSED                         [ 63%]
  tests/test_service.py::test_unseen_category PASSED                       [ 68%]
  tests/test_service.py::test_extreme_values PASSED                        [ 73%]
  tests/test_service.py::test_risky_vs_safe_sanity PASSED                  [ 78%]
  tests/test_service.py::test_http_healthz_and_ui PASSED                   [ 84%]
  tests/test_service.py::test_http_valid_predict PASSED                    [ 89%]
  tests/test_service.py::test_http_validation_errors PASSED                [ 94%]
  tests/test_service.py::test_http_unknown_sku_and_category_handling PASSED [100%]
  ======================= 19 passed, 99 warnings in 5.96s =======================
  ```

### C. Deliverable A Assertions Check
- **Command:** `py -c "import pandas as pd, numpy as np; pred=pd.read_csv('predictions.csv'); sample=pd.read_csv('data/sample_submission.csv'); ..."`
- **Exit Code:** `0`
- **Result:**
  - `list(pred.columns) == ['order_id', 'score']`: **TRUE**
  - `len(pred) == 2096`: **TRUE**
  - `pred['order_id'].is_unique`: **TRUE**
  - `set(pred['order_id']) == set(sample['order_id'])`: **TRUE**
  - `(pred['order_id'] == sample['order_id']).all()`: **TRUE**
  - `pred.isna().sum().sum() == 0`: **TRUE**
  - `np.isfinite(pred['score']).all()`: **TRUE**
  - `(pred['score'] >= 0.0).all() and (pred['score'] <= 1.0).all()`: **TRUE** (range: `[0.0587, 0.9959]`)

### D. Live Service HTTP Smoke Tests
- **Endpoint `GET /healthz`:** HTTP 200 `{"status": "ok", "model_loaded": true}`.
- **Endpoint `GET /`:** HTTP 200 HTML page containing complete form and review disclaimer.
- **Endpoint `POST /predict` (Valid):** HTTP 200 with calibrated score, risk band, threshold (`0.6485`), and top linear model drivers.
- **Endpoint `POST /predict` (Invalid/Out-of-Range):** HTTP 422 JSON validation response with zero stack traces or internal paths exposed.

---

## 3. Human Actions Remaining Prior to Final Submission

The software engineering, machine learning modeling, and documentation components are **100% complete and verified**. The following manual human actions remain:

1. **Record the 3-Minute Walkthrough Video:**
   - Follow the timed script in [`notes/recording_script.md`](../notes/recording_script.md).
   - Record the live screen (terminal, code, and browser UI at `http://localhost:8000`).
   - Confirm duration is $\le$ 3:00 minutes with clear audio, and upload to the candidate portal.
2. **Submit Candidate Portal Form:**
   - Copy the verified answers from [`notes/form-answers.md`](../notes/form-answers.md) into the live portal fields.
   - Submit the form and retain confirmation.
3. **Push Commits to Remote GitHub Repository:**
   - Run `git push origin main` to synchronize all committed phases with GitHub.
