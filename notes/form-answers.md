# Task Portal Submission Form Answers — Kestrel Home Returns Risk

These answers have been verified against all final reproduced pipeline metrics, project artifacts, and task brief requirements. They are ready to be reviewed and submitted through the candidate portal.

---

### 1. Business outcome + money
- **Target Business Outcome:** Implement a pre-dispatch risk assessment system to flag high-risk D2C orders for proactive customer confirmation phone calls by the service desk, curbing avoidable returns before warehouse dispatch.
- **Financial Baseline:** Fully-loaded cost of **₹1,150 per return** (established by Finance Controller Farhan Sheikh, covering reverse logistics, inspection, repackaging, and customer support), contrasted with the direct courier-only operational estimate of ₹600.
- **Target Volume & Capacity:** ~700 orders/month historically; service desk review capacity is assumed at **~100 orders/month** (~14.3% of volume).
- **Maximum Gross Potential Value:** Model precision at threshold (0.6485) is **32.89%** (identifying ~33 true returns/month). At ₹1,150/return, maximum gross theoretical potential savings is **₹37,823/month** (~₹4.54 Lakhs/year).
- **Realistic Net Pilot Savings:** At an expected 25% return prevention rate through courtesy phone confirmation, estimated net pilot savings are **~₹9,456/month** (~₹1.13 Lakhs/year).
- **Demonstrated Savings:** **₹0.00 to date** — true economic savings only materialize once interventions are executed and measured in production.
- **Break-Even Analysis:** $e^* = R / (P \times C) = 13.2\%$ if calls cost ₹50 each via outsourced agents; $0\%$ incremental cost using internal service desk staff.

---

### 2. Expected hidden-test score, metric, how estimated
- **Primary Metric:** Precision-Recall Area Under Curve (PR-AUC) — selected because returns are a class-imbalanced minority (~11.5%).
- **Expected Hidden-Test Estimate:** **0.3754 (95% Bootstrap CI: [0.3177, 0.4374])**.
- **Estimation Method:** Evaluated on an untouched, time-ordered holdout test slice (most recent 20% of mature training data by date, $n=2,101$).
- **Drift Risk & Generalization Note:** This estimate is derived from our historical local holdout. Performance on the hidden test set (covering Jul–Sep 2026) may experience slight variance due to seasonal shifts and payment gateway updates noted in the briefing; the 95% bootstrap interval reflects sampling uncertainty.

---

### 3. Sample size, evaluation method, error rate, failure cases
- **Sample Size:** 11,155 raw training orders; 10,504 orders after deduplicating 651 re-imported `partner_feed` duplicates. Test unlabelled set has 2,096 orders.
- **Evaluation Method:** Time-ordered 80/20 train/holdout split ($n_{\text{train}}=8,403$, $n_{\text{holdout}}=2,101$). 5-fold TimeSeriesSplit cross-validation performed strictly on the training portion.
- **Holdout Return Prevalence:** 242 positive returns out of 2,101 holdout orders (11.52% baseline).
- **Error Breakdown at Operating Threshold (0.6485):**
  - **False Negatives (missed returns):** 143 orders (orders returned that were not flagged in top 100/mo).
  - **False Positives (false alarms):** 202 orders (orders flagged in top 100/mo that would not have been returned).
  - **Overall Accuracy:** 83.58% (vs. 88.48% majority-class benchmark).
- **Known Failure Cases:** The model cannot predict returns caused by post-dispatch issues: transit delays, courier damage, or subjective product expectation mismatches.

---

### 4. How you narrowed or challenged the request
- **Challenged the 95%+ Accuracy Goal:** Proved that predicting "no return" for all orders yields 88.48% accuracy but catches zero returns. Replaced accuracy with PR-AUC and precision at reviewer capacity.
- **Challenged Automated Order Cancellations / Holds:** Proved that automated holds would cancel ~67 legitimate orders for every ~33 returns caught. Replaced automated holds with human courtesy confirmation calls.
- **Protected High-LTV Shield Loyalty Customers:** Excluded Shield loyalty members from any order hold workflows per Meenal Joshi's warning, preserving long-term brand goodwill.

---

### 5. Known bugs, shortcuts, limitations
- **No Geocoding for Pincodes:** Delivery pincodes are excluded because walk-in partner orders default to dummy `000000`, which could introduce spurious regional bias without full GIS geocoding.
- **No Natural Language Processing on Delivery Notes:** Free-text delivery notes are omitted due to customer PII risks and 26% missingness.
- **Static Monthly Threshold:** The current model uses a static score threshold (0.6485) calibrated for ~100 reviews/month rather than a dynamic daily queue throttle.

---

### 6. Excluded features and why
- `last_service_event_type`: Excluded as a severe post-outcome leak (`REVERSE_PICKUP` = 100% return rate).
- `pickup_scheduled_at`: Excluded as post-outcome logistics scheduling (91.6% return rate when populated).
- `delivery_note`: Excluded due to free-text customer PII risks and high missing rate.
- `delivery_pincode`: Excluded due to unvalidated PII-adjacent nature and dummy `000000` defaults.
- `source`: Used strictly for deduplication (`crm` vs `partner_feed`), omitted from model training.
- `customer_id`: Raw ID excluded to avoid memorization; aggregated point-in-time features (`customer_prior_orders`, `customer_prior_returns`, and smoothed `prior_return_rate`) used instead.

---

### 7. Unrequested additions
- **Centralised Architecture:** `kestrel.data_paths` for portable data discovery and `kestrel.features` for 100% training-serving feature parity.
- **Faithful Linear Model Explanations:** Real-time calculation of exact feature contributions ($w_i \cdot z_i$) from Logistic Regression coefficients.
- **Bootstrap 95% Confidence Intervals:** Clustered bootstrap by customer ID on holdout PR-AUC.
- **19-Test Automated Suite:** Unit, integration, feature parity, and HTTP `TestClient` tests in pytest.
- **Interactive Decision Dashboard:** Dark-mode FastAPI browser UI with clear human-review safeguards.

---

### 8. AI tools, help, time saved, discarded
- **Tools Used:** Antigravity (Claude Sonnet 4.6 & Gemini 3.7).
- **Time Saved:** ~4.5 hours on data auditing, feature parity scaffolding, FastAPI lifespan structure, and test suite design.
- **Discarded Suggestions:** Discarded generic 95% accuracy prompts, complex XGBoost native C++ dependencies (in favor of pure scikit-learn), and unvalidated feature lookups.

---

### 9. 3 things a teammate must know
1. **Deduplication Rule:** Partner outlet orders are re-imported via `partner_feed`. `train.py` deduplicates on `order_id`, prioritizing `crm` records.
2. **Leakage Safeguard:** Never re-introduce `last_service_event_type` or `pickup_scheduled_at` into pre-dispatch scoring — both are post-outcome records.
3. **Serving Parity:** Feature construction logic is centralized in `kestrel.features`. Retraining via `py train.py` automatically updates `artifacts/model.joblib` and `artifacts/metadata.json` consumed by `service/app.py`.

---

### 10. Honest hours
- **Total Hours Spent:** **13.5 actual hours** (17.0 planned hours).

---

### 11. Cost per prediction, monthly at ~700
- **Third-Party Inference Cost:** **₹0.00 / prediction** (₹0.00 / month at ~700 orders). Embedded local scikit-learn pipeline in FastAPI.
- **Hosting & Infrastructure:** Integrated into existing internal service desk deployment with zero paid external cloud APIs.

---

## ✋ Candidate Portal Submission Checklist (Human Actions)

- [ ] Confirm `predictions.csv` has 2,096 rows and matches `sample_submission.csv`.
- [ ] Confirm `memo.md` is at the repository root and contains verified financial figures.
- [ ] Record the 3-minute screen walkthrough following [`notes/recording_script.md`](recording_script.md).
- [ ] Upload walkthrough video to portal or provide valid public video link.
- [ ] Copy and paste the verified answers above into the candidate portal form fields.
- [ ] Submit form and retain portal confirmation receipt.
