# Phase 1 Final Audit — Kestrel Returns Risk (Variant A)

**Audit date:** 2026-10-11  
**Auditor:** Antigravity (Phase 1 — read-only, no files modified)

---

## 1. Source Files Confirmed Present

| File (actual name on disk) | Size | Role |
|---|---|---|
| `9b5e6f77-...-train.csv` | 1,423,846 B | Labelled training set |
| `45c3e718-...-test_unlabelled.csv` | 257,508 B | Unlabelled test set (submission target) |
| `7c584dce-...-sample_submission.csv` | 29,359 B | Submission format reference |
| `449d665c-...-customers.csv` | 295,343 B | Customer dimension |
| `c9e8afbe-...-products.csv` | 1,482 B | Product dimension |
| `a9b059d8-...-README.txt` | 1,980 B | Task data dictionary |
| `1c499891-...-email-thread.txt` | 3,042 B | Business context email |
| `2ea778e2-...-ops-policy.pdf` | 3,476 B | Operations policy |

All files are in the **parent directory** (`Kestrel-Returns-Risk/`), one level above the project root. No raw data files are committed to Git.

---

## 2. Data Schema & Join Audit

| Check | Result |
|---|---|
| Train shape | 11,155 rows × 19 columns |
| Test shape | 2,096 rows × 18 columns (no `returned` column) |
| Sample submission shape | 2,096 rows × 2 columns (`order_id`, `score`) |
| Sample submission IDs == Test IDs | **YES** — exact match |
| Train duplicate `order_id` count | **651** (partner_feed re-imports — handled in train.py) |
| Test duplicate `order_id` count | **0** |
| Train/test `order_id` overlap | **0** |
| Train `customer_id` not in customers.csv | **0** (all 6,131 unique train customers are in customers.csv) |
| Test `customer_id` not in customers.csv | **0** |
| Train SKUs not in products.csv | **None** (all 21 SKUs present) |
| Test SKUs not in products.csv | **None** |
| Product join row multiplication | **NO** — 11,155 before = 11,155 after join |
| Train missing: `delivery_note` | 2,752 rows (~24.7%) |
| Train missing: `pickup_scheduled_at` | 9,855 rows (~88.4%) |
| Test missing: `delivery_note` | 538 rows (~25.7%) |
| Test missing: `pickup_scheduled_at` | **2,096 rows (100%)** — all null in test |
| Return rate (raw train) | 11.36% (1,267 / 11,155) |
| Return rate (deduped train) | ~11.4% (after removing 651 partner_feed duplicates) |
| Holdout return rate | 11.52% (242 / 2,101) per metrics.json |

---

## 3. Requirement Traceability Table

| # | Requirement (Source) | Current Implementation | Status | Evidence Needed | Next Action |
|---|---|---|---|---|---|
| R1 | Deliver `predictions.csv` at repo root with `order_id`, `score` columns, 2096 rows, no missing values, finite scores | `predictions.csv` exists at root; sample_submission has 2096 rows | **needs fix** | Re-run train.py after Phase 2 path fixes and validate output | Phase 5: re-run and validate |
| R2 | Score = return risk, higher = riskier (README.txt) | Scores are probabilities from logistic regression; higher = riskier ✓ | **verified** | `predictions.csv` score distribution: min 0.059, max 0.996 | — |
| R3 | Target label: `returned` (1 = returned) | Correct — used as `y_full` in train.py | **verified** | train.py line 176: `y_full = train["returned"]` | — |
| R4 | Exclude post-outcome / leaked columns | `last_service_event_type`, `pickup_scheduled_at`, `delivery_note`, `delivery_pincode`, `source`, `customer_id` excluded | **verified** | LEAK_COLS list in train.py | — |
| R5 | `service/app.py` — FastAPI service with `POST /predict`, `GET /healthz`, `GET /` | Implemented using `lifespan` handler | **not yet tested** | Run app locally and hit all endpoints | Phase 6 |
| R6 | API response includes: order_id, score, band, threshold, reasons, note | Response model `OrderRiskResponse` includes all fields | **not yet tested** | Live test run | Phase 6 |
| R7 | Human-review disclaimer in API response | `"note": "Risk estimate for human review, not an automated decision."` | **verified** (code review) | Live test | Phase 6 |
| R8 | Risk explanations consistent with model | **NEEDS FIX** — `service/app.py` uses hard-coded rule-based heuristics, not LR coefficients | **needs fix** | Replace with coefficient-based contributions | Phase 4 |
| R9 | Memo ≤1 page, addressed to Ritu Deshpande | Memo exists at `report/memo.md` only | **needs fix** | Task brief requires memo at **repo root** | Phase 7/8 |
| R10 | `requirements.txt` at repo root | **MISSING** | **needs fix** | File does not exist | Phase 3 |
| R11 | `README.md` at repo root | Exists ✓ | **verified** | — | — |
| R12 | No auto-cancel based on score alone | Memo and UI both include human-review guardrail | **verified** | Both memo and UI disclaimer confirmed | — |
| R13 | 95% accuracy claim challenged with evidence | Memo section 2.1 does this correctly — majority class = 88.48%, catching 0 returns | **needs fix** | Line "Achieving 95% accuracy by simply thresholding can be trivially done" is contradictory — thresholding cannot reach 95% on this data with an 88.5% majority class without more tuning; rephrase to be factually precise | Phase 7 |
| R14 | Error analysis documented | `report/error_analysis.md` exists | **not yet tested** | Re-run train.py to regenerate | Phase 5 |
| R15 | Bootstrap 95% CI for PR-AUC | Computed and stored in `report/metrics.json` | **verified** | `pr_auc_ci_95: [0.3177, 0.4374]` | — |
| R16 | Clean-clone reproducibility | **NEEDS FIX** — data paths use `ROOT.parent` with UUID-named glob; service/app.py hardcodes UUID product filename | **needs fix** | See Path Issues section | Phase 2 |
| R17 | Tests pass in clean environment | 5 tests claimed to pass | **not yet tested** | Run `py -m pytest tests/` | Phase 5/6 |
| R18 | Video recording ≤3 minutes, no slides | Script exists at `notes/recording_script.md` | **NOT VERIFIED** | Actual recording must be made by developer | Phase 9 |
| R19 | Task portal form submitted | `notes/form-answers.md` is a draft only | **NOT VERIFIED** | Must be completed manually | Phase 10 |
| R20 | Dedup partner_feed re-imports | `train.py` sorts by source ('crm' < 'partner_feed'), deduplicates keeping crm | **verified** | 651 duplicates removed per metrics.json | — |

---

## 4. Identified Conflicts & Issues

### Issue 1: `prior_return_rate` formula inconsistency
- **`train.py` (line 132–134):** `customer_prior_returns / (customer_prior_orders + 1)`
- **`notes/decisions.md`:** states `customer_prior_returns / max(customer_prior_orders, 1)`
- **Impact:** These differ when `customer_prior_orders = 0`: train.py gives `returns / 1`; decisions.md gives `returns / 1` — **same result for orders=0**. But for `orders=1`: train.py gives `returns / 2`; decisions.md gives `returns / 1`. They diverge for customers with 1 prior order.
- **Resolution needed:** The README.txt does not specify the formula. The email thread says the `customer_prior_returns` field is point-in-time. The add-1 smoothing in `train.py` is defensible (Laplace smoothing avoids 100% rate for 1-of-1 returners). **Fix decisions.md to match train.py** — the code is the ground truth. Document why.

### Issue 2: Machine-specific UUID path in `service/app.py`
- **Line 26:** `PRODUCTS_PATH = BASE_DIR.parent / "c9e8afbe-9307-46ff-a798-89bf9b8e3a9c-products.csv"`
- **Impact:** This is a machine-specific UUID filename. Will fail on any other machine after a clean clone.
- **Fix:** Use a documented `data/` directory convention (or `KESTREL_DATA_DIR` env var), look for `products.csv` by suffix pattern.

### Issue 3: `service/app.py` silently falls back to empty products table
- **Lines 44–46:** When the product file is not found, it silently substitutes an empty DataFrame instead of raising an error.
- **Impact:** API then falls back to using `req.order_value_inr` as `list_price`, `Air Fryer` as family, `12` as warranty — invented values that can change model scores silently.
- **Fix:** Raise a clear `FileNotFoundError` on startup if products file is missing.

### Issue 4: Same UUID path in `tests/test_service.py`
- **Line 29:** `products_path = base_dir.parent / "c9e8afbe-9307-46ff-a798-89bf9b8e3a9c-products.csv"`
- Same problem as service/app.py — test silently uses empty products if path missing.

### Issue 5: Memo has garbled encoding characters
- `report/memo.md` line 6: Subject line shows `?` in place of em-dash due to encoding issue.
- Line 13: `Meenal?s` (smart quote encoding failure).
- **Fix:** Re-write memo with UTF-8 encoding ensuring no `\u20b9` (₹) causes cp1252 encode errors.

### Issue 6: `requirements.txt` missing
- No `requirements.txt` exists at repo root. This blocks clean-clone reproducibility.

### Issue 7: Risk explanations are rule-based heuristics, not model coefficients
- `service/app.py` `generate_reasons()` uses `if payment_mode == "cod"` logic.
- The plan requires LR coefficient-based contributions (coefficient × standardised value).
- This means the explanations do not reflect the actual model's scoring for a given order.
- **Fix:** Use `model.coef_` and the transformer's `transform()` to compute per-feature contributions.

### Issue 8: Memo is missing from repo root
- Task brief structure implies memo should be at root alongside `README.md` and `predictions.csv`.
- Currently only at `report/memo.md`.

### Issue 9: Memo line about 95% accuracy is imprecise
- "Achieving 95% accuracy by simply thresholding can be trivially done without any model" — this is misleading. With an 88.5% majority-class baseline, you would need extremely biased tuning to reach 95% accuracy; it is NOT trivially achievable. Should say: "the requested 95% accuracy target is not supported by the data. The majority-class baseline (predicting no returns ever) achieves only 88.5%."

---

## 5. Claims Consistency Check

| Claim | Source | Verified? |
|---|---|---|
| Holdout PR-AUC = 0.3754 | metrics.json / README / memo | Consistent ✓ |
| Bootstrap CI [0.3177, 0.4374] | metrics.json / memo | Consistent ✓ |
| Majority-class accuracy = 88.48% | metrics.json | Consistent ✓ |
| Deduped train = 10,504 rows | metrics.json | Consistent ✓ |
| Return rate = 11.5% | metrics.json / memo | Consistent ✓ |
| Threshold = 0.648508 | metrics.json / metadata.json | Consistent ✓ |
| Top-k = 100/month | metadata.json / memo | Labelled as assumption ✓ |
| Cost per return = ₹1,150 | memo / form-answers.md | Traced to email thread (Farhan) ✓ |
| Break-even e* = 13.2% | memo | **Needs verification** — formula: R/(P×C) = 50/(0.3289×1150) = 0.132 ✓ |
| Potential savings = ₹37,823/month | memo / form-answers.md | 100×0.3289×1150 = 37,823 ✓ but **not labelled as maximum gross potential** |
| Prior return rate formula | train.py vs decisions.md | **INCONSISTENT** — see Issue 1 |

---

## 6. Summary of Gaps Requiring Action

| Priority | Gap | Phase to Fix |
|---|---|---|
| HIGH | `requirements.txt` missing | Phase 3 |
| HIGH | Machine-specific UUID paths in service/app.py and tests | Phase 2 |
| HIGH | Silent empty-product fallback (should error) | Phase 2/4 |
| HIGH | Risk explanations are heuristic, not LR coefficient-based | Phase 4 |
| MEDIUM | `prior_return_rate` formula: decisions.md must match train.py | Phase 4 |
| MEDIUM | Memo missing from repo root | Phase 7/8 |
| MEDIUM | Memo encoding errors (₹ symbol) | Phase 7 |
| MEDIUM | Memo: imprecise statement about 95% accuracy | Phase 7 |
| MEDIUM | Memo: ₹37,823 not explicitly labelled as max gross potential | Phase 7 |
| LOW | Video recording not yet made | Phase 9 (manual) |
| LOW | Task form not yet submitted | Phase 10 (manual) |

**No original project files were modified during this audit.**
