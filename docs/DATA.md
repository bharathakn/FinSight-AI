# Data Reference — FinSight AI

## 1. Sources (`dataset/raw/`, `dataset/processed/custom_india/`) — read-only
| Source | Path | Shape | Notes |
|---|---|---|---|
| India synthetic v1.0 | `processed/custom_india/` (7 CSV + metadata) | 10,000 tx · 1,200 customers · 1,500 accounts · 1,400 devices · 2,000 beneficiaries | Aug 1–30 2026, IST timestamps, seed 42 |
| IEEE-CIS | `raw/ieee_cis/` | train 590,540 tx (+ identity), test ~506K | Anonymous V1–V339; heavy missingness |
| PaySim | `raw/paysim/PS_*.csv` | 6.36M rows, 11 cols | Fraud only in TRANSFER / CASH_OUT |
| OFAC SDN | `raw/sanctions/ofac/ofac_sdn.csv` | 19,321 entities | **Headerless**, `-0-` = null, aliases inside `remarks` ("a.k.a. '…'") |
| UN Consolidated | `raw/sanctions/un/*.xml` | 736 individuals + 275 entities | FIRST..FOURTH_NAME, *_ALIAS/ALIAS_NAME |
| Regulatory | `raw/regulatory/{rbi,fatf}/*.pdf` | — | Report framing only |

## 2. Lineage
```
custom_india (v1.0) ──inject.py (seed 42)──▶ custom_india_v1_1 (v1.1) + injection_log.csv
        │                                          │
        └──preprocessor.build_master──▶ curated/master_transactions_v1_{0,1}.parquet (57 cols)
                                                   │
                         graph/network_features ──▶ curated/master_features_v1_{0,1}.parquet (76 cols)
                                                   └▶ artifacts/graph/account_metrics_v1_{0,1}.parquet
OFAC + UN ──sanctions/parser──▶ artifacts/sanctions/entities.parquet (20,332 entities, 27,263 names)
```
Regenerate everything with `make pipeline`.

## 3. v1.1 injection summary (D-001, D-002)
| Group | Rows | What changed |
|---|---|---|
| ring (S9) | 200 | 10 hubs; members pay one hub within 48h (receiver, timestamp) |
| mule (S8) | 200 | X→M then M→Y in 2–30 min at 90–98% (receiver, timestamp, amount) |
| layer (S7) | 200 | 50 chains of 4 hops, 5–60 min apart, 2–5% decay |
| burst (S3) | 150 | 30 senders × 5 tx in 10 min (sender, timestamp) |
| struct (S6) | 150 | 30 senders × 5 tx of ₹45,000–49,999 in 24h |
| shared_dev (S11) | 150 | 15 devices × ~10 accounts (device_id, ip, is_new_device) |
| decoys (S1) | 420 | 10 merchant hubs, 60 benign forwards, 20 family devices, 10 salary bursts |
| camouflage | 471 | 50% of network-fraud rows get 4 behavioural cols from legit donors |

Labels are never modified. Validation: `reports/phase1b_validation.txt`.

## 4. Column roles in `master_features_*.parquet`
**Labels / evaluation only (LEAKAGE, never features):** `is_fraud` (primary target), `risk_label`, `scenario_id`, `alert_id`, `is_flagged_alert`

**Identifiers (joins/graph only):** `transaction_id`, `sender_account_id`, `receiver_account_id`, `sender_customer_id`, `receiver_customer_id`, `sender_vpa`, `receiver_vpa`, `sender_ifsc`, `receiver_ifsc`, `device_id`, `ip_address`

**Exp A — transaction:** `log_amount` (`amount_inr`), `channel`, `hour_of_day`, `day_of_week`, `account_balance_before`, `sender_account_age_days`, `sender_historical_30d_avg_tx_amt_inr`, `sender_historical_30d_tx_count`, `sender_historical_30d_avg_balance_inr`, `sender_account_type`, `receiver_account_type`, `receiver_account_age_days`

**Exp B — + behavioural:** `is_new_device`, `is_new_beneficiary`, `time_since_beneficiary_added_min`, `amount_ratio_vs_30d_avg`, `distance_from_home_km`, `tx_frequency_24h`, `time_since_last_tx_sec`, `device_type`, `device_is_registered`, `device_associated_accounts`, `device_age_days`, `sender_bene_count`, `sender_bene_added_7d`, `sender_bene_min_prior_tx`, `sender_kyc_status`, `sender_pep_flag`, `sender_customer_type`, `sender_declared_annual_income_inr`

**Exp C — + network (point-in-time):** `sender_tx_10m`, `sender_tx_1h`, `sender_tx_24h`, `sender_distinct_receivers_24h`, `receiver_tx_in_24h`, `receiver_distinct_senders_24h`, `receiver_distinct_senders_7d`, `sec_since_sender_last_inflow`, `forward_ratio`, `is_pass_through`, `chain_depth`, `device_distinct_accounts`, `ip_distinct_accounts`
**Exp C — + network (global, transductive):** `sender_pagerank`, `receiver_pagerank`, `sender_community_size`, `receiver_community_size`, `same_community`, `receiver_clustering`

**Avoid / context only:** `sender_current_balance_inr` (end-of-period snapshot, not point-in-time), `account_balance_after` (= before − amount), `day_of_month` (only one month of data), `tx_lat`/`tx_long`/`tx_city`/`tx_state`/home city/state (EDA and maps only), `log_amount` and `amount_inr` together (use one)

## 5. Audit facts worth citing
- 0 nulls (except `alert_id` by design), 0 duplicates, 100% FK integrity in v1.0
- `channel` ≡ `transaction_type`; `status` constant → dropped
- `amount_ratio_vs_30d_avg` = amount / sender 30d avg (99.1%); `distance_from_home_km` is **not** derived from lat/long (11.5% match)
- `time_since_beneficiary_added_min < 1 day` and `distance > 100 km` never occur on LEGITIMATE rows (generator artifact → high-confidence rules partly by construction)
- `account_type = MULE_SUSPECT` (40 accounts) is **not** a label proxy: 11.3% fraud among those senders vs 13.5% base (D-014)
- PASS_THROUGH flag: 99.4% fraud, 1.6% support (D-016); late-night hours reach 22–27% fraud
- An always-legitimate model scores 86.5% accuracy on `is_fraud`, so accuracy must never be the headline metric
- Sanctions threshold sweep on 1,200 customers: alerts 279 @70, 20 @80, 2 @90 (all false positives)
