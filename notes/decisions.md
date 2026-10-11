# Decisions — Kestrel Returns Risk (Variant A)

## Extraction sheet (Plan §1)

| # | Item | Finding |
|---|---|---|
| 1 | **Target column** | `returned` (1 = returned). Recorded after dispatch, confirmed from README and email. |
| 2 | **Return window** | Not explicitly stated. From data: train covers Apr 2025–Jun 2026; test Jul–Sep 2026. Service event `REVERSE_PICKUP` appears in train data as a post-outcome marker. |
| 3 | **predictions.csv format** | `order_id`, `score` (from sample_submission.csv). Higher score = riskier. 2096 rows. |
| 4 | **Pre-dispatch vs post-outcome columns** | See timing table below. |
| 5 | **Exclusions** | Duplicate partner_feed re-imports: keep `crm` row, drop `partner_feed` duplicate (651 rows). |
| 6 | **Prohibited/sensitive features** | `delivery_pincode` (PII-adjacent, 000000 = walk-in default). `delivery_note` (free text, PII risk). `customer_id` directly. Excluded per policy caution. |
| 7 | **Money** | Email: Ritu says ₹600/return. Farhan (Finance Controller) says ₹1,150 all-in. We use ₹1,150 (Finance Controller's figure, more authoritative). Hold/cancel cost: unquantified — noted as assumption. |
| 8 | **Intervention / reviewer capacity** | Pilot: confirmation calls (Meenal). No stated capacity number — **assumption: review 10–15% of orders/month** (top-k). State as assumption. |
| 9 | **95% claim** | Made by Ritu to the board, on no stated data/metric. Majority-class accuracy on deduped train ≈ 88.6%. So predicting "never return" gives ~88.6% accuracy — not 95%. The 95% claim is not supported by the data baseline. |
| 10 | **Conflicts** | Ritu (₹600) vs Farhan (₹1,150): use Farhan's figure. Ritu wants "flag and hold"; Meenal warns about Shield customers. Memo recommends human-reviewed pilot, not auto-hold. |
| 11 | **Volume** | Train: 10,504 deduped orders over ~15 months ≈ 700/month. Test: 2,096 orders over 3 months ≈ 699/month. Confirmed. |
| 12 | **Service fields** | API accepts raw order fields (same columns as test_unlabelled.csv minus post-outcome columns). No ID lookups needed — `customer_prior_orders` and `customer_prior_returns` are pre-computed in the export. |

## Column timing table

| Column | Timing | Use? | Notes |
|---|---|---|---|
| order_id | pre-dispatch | ID only | |
| order_placed_at | pre-dispatch | ✅ (for split only) | |
| customer_id | pre-dispatch | ❌ direct | Use aggregated history instead |
| sku | pre-dispatch | ✅ | Join to products for family, price band |
| sales_channel | pre-dispatch | ✅ | |
| payment_mode | pre-dispatch | ✅ | COD has highest return rate |
| discount_pct | pre-dispatch | ✅ | Higher discount → higher return |
| qty | pre-dispatch | ✅ | |
| order_value_inr | pre-dispatch | ✅ | |
| promised_delivery_days | pre-dispatch | ✅ | |
| delivery_pincode | pre-dispatch | ❌ | PII-adjacent; 000000 walk-in default; not usable without geocoding |
| is_gift | pre-dispatch | ✅ | |
| customer_prior_orders | pre-dispatch | ✅ | Already point-in-time per README |
| customer_prior_returns | pre-dispatch | ✅ | Already point-in-time per README |
| delivery_note | pre-dispatch | ❌ | Free text, PII risk, 26% missing |
| last_service_event_type | **POST-OUTCOME** | ❌ LEAK | REVERSE_PICKUP → 100% returned; DEMO_DONE/INSTALL_DONE → 0% returned |
| pickup_scheduled_at | **POST-OUTCOME** | ❌ LEAK | 91.6% returned when non-null; reverse logistics scheduling |
| source | pre-dispatch | ❌ | Used only for dedup; not a true pre-dispatch signal |
| returned | label | TARGET | |

## Key decisions

- **Dedup:** keep `crm` row when duplicate exists (`partner_feed` re-import).
- **Leakage exclusions:** drop `last_service_event_type`, `pickup_scheduled_at`, `source`, `delivery_note`, `delivery_pincode`, `customer_id` from features.
- **Prior return rate feature:** `customer_prior_returns / (customer_prior_orders + 1.0)` (Laplace-smoothed point-in-time rate). Implemented centrally in `kestrel.features.compute_prior_return_rate`. Rationale: Prevents single-order customers (1 order, 1 return) from receiving an extreme 1.0 rate compared to repeat customers, while guaranteeing division-by-zero safety.
- **Unknown SKU handling:** Documented catalog fallback in `kestrel.features.lookup_product_details`: uncatalogued SKUs default to `product_family='Unknown'` (handled by `OneHotEncoder(handle_unknown='ignore')`), median list price (`₹4,723`), and mode warranty (`12 months`). Never hardcodes a specific family like Air Fryer.
- **Model explanations:** Faithful linear model contributions ($z_i \cdot w_i$) from fitted Logistic Regression coefficients in `kestrel.features.explain_prediction`, replacing hardcoded heuristic rules.
- **Models:** B0 (constant), B1 (logistic regression), M1 (HistGradientBoostingClassifier).
- **Primary metric:** PR-AUC (imbalanced, minority = returns).
- **Threshold:** Based on assumption of reviewing top ~100 orders/month (≈14% of 700).
- **Cost figure used:** ₹1,150 (Farhan, Finance Controller).
