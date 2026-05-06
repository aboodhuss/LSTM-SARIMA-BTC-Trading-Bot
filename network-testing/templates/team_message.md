Subject: Trading Bot Network Test Package and Required Outputs

Team,

I have prepared a complete execution package for the network testing phase.

Package location:
- `trading-bot-complete-<timestamp>.zip`

Project root:
- `trading-bot` (project root folder after extraction)

Please execute the plan in this order:

1. Read `network-testing/NETWORKING_TEST_README.md`.
2. Then review `network-testing/docs/NETWORK_TEST_GUIDE.md`.
3. Start the system and verify baseline operation.
4. Run profiles from `network-testing/profiles/network_profile_matrix.csv` one at a time.
5. For each profile, use:
   - `network-testing/scripts/run_profile_session.sh` on Linux, or
   - the Clumsy mapping in `network-testing/profiles/windows_clumsy_profiles.md`.
6. Confirm shaping cleanup and post-profile recovery after each profile block.

Required deliverables for each profile:

1. Raw session CSV (`*_raw.csv`).
2. Session summary JSON (`*_summary.json`).
3. Updated `summary_table.csv`.
4. Short notes describing anomalies and recovery behavior.
5. Baseline-vs-profile comparison for:
   - decision delay
   - processed/dropped candle continuity
   - trade count
   - total PnL
   - confidence behavior

If you hit setup issues, send:
- full command used
- terminal output
- profile ID
- run timestamp (UTC)
