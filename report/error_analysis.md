# Error Analysis -- Kestrel Returns Risk Holdout

**Model:** LogisticRegression
**Holdout period:** 2026-04-02 to 2026-06-30
**Holdout n:** 2101, positives: 242 (return rate 0.1152)

## Summary
- **False negatives (missed returns):** 143 -- orders that were returned but not flagged
- **False positives (false alarms):** 202 -- orders flagged but not returned
- **Threshold:** 0.6485 (top ~100 orders/month by reviewer capacity assumption)

## False Negatives by sales channel
sales_channel
app               59
marketplace       36
web               32
partner_outlet    16

## False Negatives by payment mode
payment_mode
cod             57
prepaid_upi     39
prepaid_card    26
emi             21

## False Positives by sales channel
sales_channel
marketplace       81
app               63
web               43
partner_outlet    15

## Sample false negatives (anonymised -- no PII)
 order_id sales_channel payment_mode  discount_pct  order_value_inr  customer_prior_returns    score
KO2608406           app          cod            31          3258.87                       0 0.519317
KO2608415           web          emi            24          2127.24                       0 0.380901
KO2608422   marketplace prepaid_card            20          2698.40                       0 0.377371
KO2608427           app prepaid_card            23          3636.71                       1 0.472342
KO2608433           web          cod             1          2474.01                       0 0.507936

## What the model cannot see
1. **Order content / description** -- high-value or complex appliances may have unmet expectations not captured in SKU or price alone.
2. **Actual delivery experience** -- delayed deliveries, damaged packaging, and courier-level issues are not in the data.
3. **Reason for return** -- the model has no explicit signal on why customers return (quality, changed mind, wrong product).
4. **Shield membership interactions** -- Meenal (Service Desk) notes that most returns are Shield customers for whom returns are free; shielded customers may behave systematically differently.
5. **Seasonal / campaign effects** -- the October festive payment gateway change (noted by Tanmay) may introduce anomalies not yet visible in train.
