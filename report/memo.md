# Executive Memo: Returns Risk Model & Operational Pilot Proposal

**To:** Ritu Deshpande, Head of D2C Operations  
**From:** Data & Analytics Team  
**Date:** October 11, 2026  
**Subject:** Pre-Dispatch Returns Risk Model — Technical Evaluation, Financial Impact, & Operational Pilot Proposal  

---

### 1. Executive Summary & Core Recommendation

We recommend initiating a controlled, **human-in-the-loop operational pilot** of pre-dispatch returns risk scoring for Kestrel Home D2C orders:

1. **Human-Review Safeguard:** The model will rank outgoing orders and flag the top **~100 highest-risk orders per month** (~14% of monthly volume) for proactive phone confirmation by the service desk.
2. **Zero Automated Cancellations or Holds:** High-risk scores will **never** trigger automatic order cancellations, holds, or blocks. Every flagged order will undergo human review to protect genuine customer goodwill.
3. **Shield Customer Protection:** In line with Meenal Joshi’s operational guidance, Shield loyalty members (who purchase 3+ appliances annually and receive free returns as a membership perk) must **never have their orders held**. They will receive friendly, white-glove delivery confirmation calls rather than restrictive friction.
4. **Zero Third-Party API Costs:** The scoring model runs 100% locally via open-source scikit-learn inside our lightweight FastAPI service, incurring **₹0.00 in per-prediction or external cloud API fees**.

---

### 2. Empirical Findings & Challenges to Initial Assumptions

#### A. The "95% Accuracy" Goal vs. Class-Imbalanced Reality
- In our cleaned historical dataset (10,504 deduplicated orders), the overall return rate is **11.45%** (and **11.52%** on the holdout test period).
- A naive baseline that blindly predicts *"No order will ever be returned"* achieves **88.48% accuracy**, yet identifies **zero returns** and generates **₹0.00 in business savings**.
- Because returns represent an imbalanced minority class (~11.5%), raw accuracy is fundamentally misleading. We evaluated models using **Precision-Recall Area Under Curve (PR-AUC)**:
  - **B0 Naive Prior Baseline:** PR-AUC = `0.1152`, ROC-AUC = `0.5000`
  - **B1 Logistic Regression (Deployed):** PR-AUC = `0.3754` (95% Bootstrap CI: `[0.3177, 0.4374]`), ROC-AUC = `0.7590`
  - **M1 HistGradientBoosting:** PR-AUC = `0.3261`, ROC-AUC = `0.7395` (tested via 5-fold TimeSeriesCV)
- **Model Choice:** Logistic Regression was selected because it outperforms gradient boosting on holdout PR-AUC by +0.05, runs in sub-millisecond inference time, and provides transparent, mathematically faithful feature contributions ($w_i \cdot z_i$) for service desk agents.

#### B. Pre-Dispatch Integrity & Data Leakage Elimination
Our initial data audit identified two critical post-outcome columns that artificially inflated return prediction:
- `last_service_event_type`: Contained `REVERSE_PICKUP` (100% return rate).
- `pickup_scheduled_at`: Post-outcome reverse logistics scheduling (91.6% return rate when populated).

Both fields—along with un-geocoded pincodes (`delivery_pincode`) and free-text notes (`delivery_note`)—were strictly excluded. All model inputs are verified to be strictly known **prior to warehouse dispatch**.

---

### 3. Financial Analysis: Potential vs. Demonstrated Value

#### Cost Accounting: ₹1,150 Fully-Loaded vs. ₹600 Direct Courier Cost
In evaluating the financial impact, we adopt Finance Controller Farhan Sheikh's fully-loaded operational figure of **₹1,150 per return**, which accounts for reverse logistics freight, warehouse inspection, repackaging, restock handling, and customer support. For comparison, the earlier ₹600 operational estimate covers only direct courier charges and understates true P&L leakage.

| Operating Parameter | Value | Basis / Source |
|---|---|---|
| Monthly Order Volume ($N$) | ~700 orders | Historical monthly average (Apr 2025 – Sep 2026) |
| Baseline Return Rate ($r$) | 11.52% (~80 returns/mo) | Holdout test period |
| Fully-Loaded Cost per Return ($C$) | **₹1,150** | Finance Controller (Farhan Sheikh) |
| Operational Review Capacity ($k$) | **100 orders / month** (~14.3%) | Service desk staffing assumption |
| Operating Score Threshold | **0.6485** | Calibrated to select top ~14% risky orders |
| Model Precision at Threshold ($P$) | **32.89%** | Holdout test evaluation |
| Model Recall at Threshold | **40.91%** | Holdout test evaluation |
| True Returns Identified ($k \times P$) | **~33 returns / month** | ~41% of all monthly returns |
| **Maximum Gross Potential Savings** | **₹37,823 / month** | **₹4.54 Lakhs / year** (gross upper bound) |

#### Important Distinction: Gross Potential vs. Realistic Net Savings
1. **Demonstrated Savings to Date:** **₹0.00**  
   No model generates cost savings until an operational intervention is deployed, executed, and measured in the field.
2. **Maximum Gross Potential Savings (₹37,823/month):**  
   Represents the theoretical maximum if 100% of the 33 identified returns could be prevented with zero customer contact friction and zero administrative cost.
3. **Realistic Net Pilot Savings:**  
   In practice, confirmation calls achieve a contact rate (typically 60–75%) and a resolution rate where customer uncertainty is resolved (e.g. 20–30% returns averted through address correction, delivery date rescheduling, or buyer remorse resolution).  
   - At a realistic **25% prevention rate** on identified returns:  
     $$\text{Estimated Net Savings} = 32.89 \times 0.25 \times ₹1,150 = \mathbf{₹9,456\text{ / month}} \quad (\approx \mathbf{₹1.13\text{ Lakhs / year}})$$

#### Break-Even Intervention Analysis ($e^*$)
If dedicated review calls incur a marginal cost $R$ (e.g. ₹50 per phone call via outsourced agents):
$$e^* = \frac{R}{P \times C} = \frac{50}{0.3289 \times 1,150} = \mathbf{13.2\%}$$
If more than **13.2% of calls** successfully prevent an impending return, the confirmation program is net profitable. Using existing internal service desk bandwidth ($R \approx ₹0$), the program is immediately cash-flow positive.

---

### 4. Operational Risk Management & Customer Goodwill

| Risk Vector | Magnitude | Mitigation Strategy |
|---|---|---|
| **False Positives (False Alarms)** | ~67 of 100 flagged orders/mo are non-returns | **Human-in-the-loop framing:** Calls must be framed as courtesy customer care (*"Confirming your installation address and product specifications before dispatch"*), never as fraud interrogation or cancellation threats. |
| **Customer Cancellation Friction** | High if orders are delayed | Orders must not be delayed beyond 24 hours. If unreachable after two attempts, orders proceed to normal dispatch. |
| **Shield Loyalty Member Attrition** | Critical brand risk | Shield members must be flagged with a loyalty VIP tag. They are never subjected to order holds. |
| **Missed Returns (False Negatives)** | ~47 returns/mo unflagged | Expected under 100-order capacity constraints. Re-calibrated monthly as new post-dispatch outcome data is collected. |

---

### 5. Implementation Roadmap

1. **Week 1 — Service Deployment:**  
   Deploy the containerized FastAPI risk assessment service and interactive decision dashboard on internal infrastructure.
2. **Weeks 2–5 — 4-Week Operational Pilot:**  
   Route the top 100 scored orders per month to Meenal Joshi's service desk team. Conduct friendly pre-dispatch confirmation calls. Maintain a randomized 20% holdout control group of uncontacted high-risk orders.
3. **Week 6 — Executive Review & ROI Measurement:**  
   Compare actual return rates between contacted orders and the control group to establish demonstrated empirical savings ($e$) and determine full-scale rollout parameters.
