# Kestrel Home — Returns Risk Model & Decision System (Variant A)

An end-to-end Machine Learning pipeline, web service, and decision support UI for predicting e-commerce Direct-to-Consumer (D2C) return risk prior to warehouse dispatch.

---

## 📌 Problem Overview & Business Context

Kestrel Home faces significant D2C product returns, costing an estimated **₹1,150 per return** in reverse logistics, restocking, quality checks, and customer support (per Finance Controller Farhan Sheikh).

This project builds a pre-dispatch risk assessment system to flag high-risk orders for **human review (courtesy customer confirmation calls)** prior to warehouse dispatch:
- **Human-in-the-loop safeguard:** High-risk scores never trigger automated cancellations, holds, or blocks. Every flagged order is queued for team phone confirmation.
- **Shield customer protection:** High-LTV Shield loyalty members are explicitly protected and never subjected to order holds.
- **Operational capacity:** Designed for a review queue of ~100 orders per month (~14% of monthly volume), matching operational staffing.
- **Zero external API costs:** 100% open-source local execution with no paid external API dependencies.

---

## ⚙️ Environment & Python Compatibility

- **Supported Python Versions:** Python 3.10 – 3.14
- **Tested & Verified On:** Python 3.14.7 (Windows 64-bit AMD64)
- **Dependencies:** Managed via [`requirements.txt`](requirements.txt) (free, open-source libraries: `pandas`, `numpy`, `scipy`, `scikit-learn`, `joblib`, `fastapi`, `uvicorn`, `pydantic`, `pytest`, `httpx`).

---

## 🚀 Setup & Installation Guide

### 1. Clone Repository & Navigate
```bash
git clone https://github.com/shubham333k/kestrel-returns-risk.git
cd kestrel-returns-risk
```

### 2. Create and Activate Virtual Environment

**Windows PowerShell:**
```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
```
*(If PowerShell script execution is restricted, run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` first).*

**macOS / Linux:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

---

## 📂 Data Placement & Configuration

Source datasets contain customer transaction histories and are excluded from Git for privacy and compliance.

### Option A: Local `data/` Directory (Recommended)
Place the task CSV files directly into the `data/` folder:
```
kestrel-project/
└── data/
    ├── train.csv (or *-train.csv)
    ├── test_unlabelled.csv (or *-test_unlabelled.csv)
    ├── products.csv (or *-products.csv)
    └── sample_submission.csv (or *-sample_submission.csv)
```

### Option B: Custom Environment Variable
If your data is stored in an external directory, set `KESTREL_DATA_DIR`:

**Windows PowerShell:**
```powershell
$env:KESTREL_DATA_DIR = "C:\path\to\your\data"
```

**macOS / Linux:**
```bash
export KESTREL_DATA_DIR="/path/to/your/data"
```

> **Note:** If any required file is missing, `kestrel.data_paths` raises an actionable `FileNotFoundError` detailing the expected filename and search locations. No fake records or silent empty tables are substituted.

---

## 🧪 Testing & Execution

### 1. Run Complete Test Suite
Verify environment, path resolution, feature parity, model explanations, API contracts, and HTTP endpoints (19 tests):
```bash
pytest tests/ -v
```

### 2. Train Model & Generate Deliverables
Train the model, evaluate baselines, and generate `predictions.csv`:
```bash
python train.py
```
**Artifacts Generated:**
- `predictions.csv` — Deliverable A (2,096 test predictions with `order_id` and `return_risk_score`)
- `memo.md` & `report/memo.md` — Deliverable: Executive Memo for Head of D2C Operations
- `artifacts/model.joblib` — Trained production pipeline (Logistic Regression)
- `artifacts/metadata.json` — Model metadata, threshold (`0.6485`), and training parameters
- `report/metrics.json` — Out-of-sample evaluation metrics
- `report/error_analysis.md` — Holdout error patterns and failure analysis
- `report/verification_results.md` — Execution logs and test evidence

### 3. Launch Web Service & Decision UI
Start the FastAPI server:
```bash
uvicorn service.app:app --reload --port 8000
```
- **Interactive Decision Dashboard:** Open [http://localhost:8000](http://localhost:8000) in your browser.
- **Interactive API Documentation (Swagger UI):** Open [http://localhost:8000/docs](http://localhost:8000/docs).
- **Service Health Check:** [http://localhost:8000/healthz](http://localhost:8000/healthz).

---

## 📊 Example API Request & Response

### Request (`POST /predict`)
```json
{
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
  "customer_prior_returns": 1
}
```

### Response
```json
{
  "order_id": "KO2610504",
  "return_risk_score": 0.7879,
  "risk_band": "HIGH",
  "threshold": 0.6485,
  "reasons": [
    {
      "feature": "num__customer_prior_returns",
      "impact": "+Risk",
      "description": "Customer Prior Returns increases return risk (+0.91 log-odds impact in model)."
    },
    {
      "feature": "cat__payment_mode_cod",
      "impact": "+Risk",
      "description": "Payment Mode: COD increases return risk (+0.78 log-odds impact in model)."
    },
    {
      "feature": "cat__is_gift_N",
      "impact": "-Risk",
      "description": "Gift Order: No reduces return risk (-0.32 log-odds impact in model)."
    }
  ],
  "note": "Risk estimate for human review, not an automated decision."
}
```

---

## 📈 Model Performance Summary

| Model | Evaluated Split | PR-AUC | ROC-AUC | Notes |
|---|---|---|---|---|
| **B0 Constant Prior** | 20% Time Holdout | 0.1152 | 0.5000 | Naive baseline prior |
| **B1 Logistic Regression (Deployed)** | 20% Time Holdout | **0.3754** (95% CI: [0.3177, 0.4374]) | **0.7590** | **Selected** (simpler, explainable, stable) |
| **M1 HistGradientBoosting** | 20% Time Holdout | 0.3261 | 0.7395 | Tested with 5-fold TimeSeries CV |

- **Majority-Class Benchmark:** 88.48% (Predicting "no return" for all orders yields ~88.5% accuracy but detects 0 returns).
- **Operational Threshold:** `0.6485` (Selected to flag top ~14% risky orders matching reviewer capacity of ~100 orders/month).
- **Holdout Precision at Threshold:** 32.89% (~33 returns caught per month).
- **Holdout Recall at Threshold:** 40.91% (~41% of all returns identified).
- **Max Gross Theoretical Potential:** ₹37,823 / month (₹4.54 Lakhs / year).
- **Realistic Net Pilot Savings:** ~₹9,456 / month (~₹1.13 Lakhs / year) assuming 25% prevention rate. Zero demonstrated savings prior to pilot.

---

## 📁 Repository Structure & Deliverables

```
kestrel-project/
├── memo.md                 # REQUIRED DELIVERABLE: Executive Memo for Head of D2C Operations
├── predictions.csv         # REQUIRED DELIVERABLE: Deliverable A (2,096 test predictions)
├── requirements.txt        # Pinned runtime and testing dependencies
├── train.py                # Model training, CV evaluation, & predictions generator
├── kestrel/
│   ├── __init__.py
│   ├── data_paths.py       # Centralised data discovery & path resolution
│   └── features.py         # Shared feature building & linear explanation logic
├── service/
│   └── app.py              # FastAPI app & browser review UI
├── tests/
│   ├── test_paths.py       # Path resolution & environment tests (5 tests)
│   ├── test_features.py    # Feature parity & explanation tests (5 tests)
│   └── test_service.py     # Unit and HTTP TestClient API tests (9 tests)
├── artifacts/
│   ├── model.joblib        # Fitted scikit-learn pipeline
│   └── metadata.json       # Operating thresholds, feature list, and metadata
├── report/
│   ├── memo.md             # Executive Memo (copy)
│   ├── metrics.json        # Out-of-sample holdout metrics
│   ├── error_analysis.md   # Error breakdown & limitations
│   └── verification_results.md # Actual execution logs and test traces
└── notes/
    ├── final_audit.md      # Comprehensive Phase 1 audit & traceability
    ├── decisions.md        # Feature timing & data extraction sheet
    ├── log.md              # Engineering time tracking log
    ├── ai-log.md           # AI tooling usage & audit trail
    ├── form-answers.md     # Pre-mapped task submission answers
    └── recording_script.md # 3-minute video recording outline
```

---

## 📋 Deliverables Submitted via Task Portal

The following items are prepared in the repository and ready for portal submission:
1. **Walkthrough Video:** 3-minute walkthrough script prepared in [`notes/recording_script.md`](notes/recording_script.md). Recorded browser demo video saved in repository artifacts.
2. **Task Portal Form Responses:** Pre-mapped answers verified against the live schema in [`notes/form-answers.md`](notes/form-answers.md).
