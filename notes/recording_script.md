# 3-Minute Walkthrough Video Runbook & Script — Kestrel Returns Risk

**Target Duration:** Strictly $\le$ 3:00 minutes (180 seconds)  
**Format:** Live screen recording of terminal, code, and browser UI (Strictly **No Slide Decks**)  
**Tool Recommendation:** Windows Snipping Tool (`Win + Shift + R`), OBS Studio, or Loom  
**Target Video Deliverable:** `walkthrough.mp4` (or portal upload link)  

---

## 📋 Pre-Recording Setup Checklist

- [ ] **Clean Environment:** Terminal open to `kestrel-project` directory.
- [ ] **Background API Server Running:** `uvicorn service.app:app --reload --port 8000` active.
- [ ] **Browser Window Ready:** Open to `http://localhost:8000` (clean dark mode UI).
- [ ] **Privacy Check:** Ensure no local paths (`C:\Users\...`), personal email addresses, API tokens, or customer names appear on screen.
- [ ] **Timer Active:** Set phone or stopwatch to 2 minutes 50 seconds to ensure recording finishes comfortably under the hard 3:00 limit.

---

## ⏱️ Timed Script & Visual Cue Schedule

### Segment 1: The Problem & Why PR-AUC Over Accuracy (0:00 – 0:35)
- **Visual:** Terminal or editor showing [`memo.md`](../memo.md) and [`report/metrics.json`](../report/metrics.json).
- **Spoken Voiceover:**
  > *"Hi Ritu and team. When this project began, the brief requested a 95% accuracy model. However, our empirical audit showed that in historical D2C data, returns represent only 11.5% of orders.*
  > *A naive dummy model predicting 'never return' achieves 88.5% accuracy, yet catches zero returns and delivers zero rupees in savings.*
  > *In class-imbalanced problems, accuracy is misleading. We adopted Precision-Recall Area Under Curve (PR-AUC) as our primary metric to measure how effectively we identify the highest-risk orders."*

---

### Segment 2: Data Integrity & Leakage Decisions (0:35 – 1:10)
- **Visual:** Editor showing [`notes/decisions.md`](decisions.md) column timing table.
- **Spoken Voiceover:**
  > *"Our data audit revealed critical data leakage in the raw export:*
  > *`last_service_event_type` had a 100% correlation with returns because `REVERSE_PICKUP` is logged after the customer requests a return. Similarly, `pickup_scheduled_at` had a 91.6% return correlation.*
  > *We strictly stripped both post-outcome columns. We also deduplicated 651 re-imported partner outlet orders and excluded free-text delivery notes and unvalidated pincodes to eliminate PII risks.*
  > *Every feature used in our model is strictly known before the warehouse dispatches the parcel."*

---

### Segment 3: Model Comparison & Holdout Results (1:10 – 1:45)
- **Visual:** Editor showing [`report/verification_results.md`](../report/verification_results.md) comparison table.
- **Spoken Voiceover:**
  > *"We evaluated three models on an untouched, time-ordered 20% holdout split:*
  > *The B0 naive baseline prior scored a PR-AUC of 0.1152.*
  > *HistGradientBoosting achieved 0.3261.*
  > *Our deployed Logistic Regression model achieved a PR-AUC of 0.3754 with a 95% bootstrap confidence interval of [0.3177, 0.4374] — beating the baseline by more than 3.2 times.*
  > *We selected Logistic Regression because it yields higher holdout precision and provides additive, mathematically faithful feature contributions for service desk agents."*

---

### Segment 4: Live Decision Dashboard Demo (1:45 – 2:30)
- **Visual:** Switch to browser at `http://localhost:8000`. Fill out form and submit live.
- **Action A (High-Risk Order):**
  - Order ID: `KO2610504`
  - SKU: `KH-RV-01 (Robot Vacuum Lite)`
  - Sales Channel: `marketplace` | Payment Mode: `cod`
  - Discount %: `30` | Prior Orders: `3` | Prior Returns: `2`
  - Click **"Evaluate Risk Score"**.
- **Spoken Voiceover:**
  > *"Here is the live FastAPI dashboard. For an order on Marketplace with Cash on Delivery, high discount, and 2 previous returns, the model outputs an observed risk score of 96.0% — placing it in the HIGH RISK band.*
  > *Notice the exact linear model drivers displayed below: Customer Prior Returns adds +2.02 log-odds, and Cash on Delivery adds +0.78.*
  > *Crucially, notice the safeguard: 'Risk estimate for human review, not an automated decision.' We never cancel this order automatically."*

- **Action B (Low-Risk Order):**
  - Change Order ID: `KO2610599`
  - SKU: `KH-AF-01`
  - Sales Channel: `app` | Payment Mode: `prepaid_upi`
  - Discount %: `0` | Prior Orders: `5` | Prior Returns: `0`
  - Click **"Evaluate Risk Score"**.
- **Spoken Voiceover:**
  > *"Now let's evaluate a repeat customer on our App using Prepaid UPI with zero returns. The score immediately drops to 21.5% — LOW RISK, with protective factors clearly highlighted."*

---

### Segment 5: Financial Realism & Operational Pilot (2:30 – 2:55)
- **Visual:** Switch back to [`memo.md`](../memo.md) Section 3.
- **Spoken Voiceover:**
  > *"Finally, on financials: Finance Controller Farhan Sheikh established a fully-loaded return cost of ₹1,150.*
  > *Calibrating our threshold to our team review capacity of ~100 orders per month catches ~33 returns, representing a gross theoretical potential of ₹37,823 per month.*
  > *However, demonstrated savings to date are ₹0.00 until we run the pilot.*
  > *Our memo recommends a 4-week pilot where the desk conducts courtesy confirmation calls. Shield members are protected from order holds to safeguard lifetime customer value."*

---

### Segment 6: Wrap-Up & Test Suite (2:55 – 3:00)
- **Visual:** Quick glimpse of terminal showing `19 passed in 5.07s`.
- **Spoken Voiceover:**
  > *"All 19 automated unit and HTTP integration tests pass cleanly. Thank you."*

---

## 📹 Post-Recording Verification Checklist

1. **Duration Check:** Confirm the video length is $\le$ 3:00 (e.g. 2:45 to 2:58).
2. **Audio Quality:** Play back to ensure voice is clear, audible, and synchronized with screen actions.
3. **Resolution:** Confirm text and numbers are readable (minimum 1080p recommended).
4. **Naming:** Save file as `walkthrough.mp4` or upload to the video hosting platform requested by the task portal.
