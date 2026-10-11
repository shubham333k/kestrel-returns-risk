# Time Log — Kestrel Returns Risk Project

| Phase / Activity | Planned Hours | Actual Hours | Notes |
|---|---|---|---|
| 0–2: Read, extract, decisions | 2.0 | 1.5 | Read README, email thread, policy; mapped extraction sheet in `decisions.md` |
| 2–4: Data audit & leakage check | 2.0 | 1.5 | Checked shapes, missing values, deduped partner_feed (651 rows), audited `last_service_event_type` and `pickup_scheduled_at` leaks |
| 4–8: Feature engineering, split & modelling | 4.0 | 3.5 | Point-in-time features, 80/20 time-ordered split, trained B0, B1, M1, computed bootstrap CIs |
| 8–9: Deliverable A verification | 1.0 | 0.5 | Generated `predictions.csv` (2096 rows) & verified format against sample_submission |
| 9–12: Service, UI & pytest suite | 3.0 | 2.5 | FastAPI `app.py` with `lifespan`, interactive dark-mode HTML UI, pytest test suite (19 passing tests) |
| 12–14: Business value model & Executive Memo | 2.0 | 1.5 | Computed potential/demonstrated savings, break-even $e^*$, wrote `report/memo.md` |
| 14–15: Recording script & error analysis | 1.0 | 1.0 | Wrote `report/error_analysis.md` and 3-minute video script `notes/recording_script.md` |
| 15–17: Repo setup, form answers, final QA | 2.0 | 1.5 | Wrote `README.md`, `.gitignore`, `notes/form-answers.md`, ran clean-machine test |
| **Total** | **17.0** | **13.5** | Finished with margin ahead of deadline |
