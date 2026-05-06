# Networking Test README

This is the guide to send to the networking team.

## Goal

Run repeatable baseline vs shaped-network tests on the trading bot and return comparable metrics.

## What To Read First

1. `network-testing/docs/NETWORK_TEST_GUIDE.md`
2. `network-testing/profiles/network_profile_matrix.csv`

## Quick Folder Guide

- `network-testing/docs`: full runbook and interpretation rules.
- `network-testing/profiles`: approved profiles to run.
- `network-testing/scripts`: commands used for execution and capture.
- `network-testing/results`: where output files are saved.
- `network-testing/templates/team_message.md`: team message template.

## Standard Workflow (Linux)

1. Start the project stack:
   - `cd trading-bot`
   - `./start.sh`
2. Find NIC:
   - `ip -br a`
3. Run baseline:
   - `./network-testing/scripts/run_profile_session.sh <iface> baseline 180`
4. Run shaped profiles one at a time:
   - `./network-testing/scripts/run_profile_session.sh <iface> latency_300_jitter_50 90`
   - `./network-testing/scripts/run_profile_session.sh <iface> loss_3 90`
   - `./network-testing/scripts/run_profile_session.sh <iface> bandwidth_2mbit 90`
5. Run recovery baseline:
   - `./network-testing/scripts/run_profile_session.sh <iface> baseline_recovery 30`

## Windows Fallback

If Linux `tc` is unavailable, use:

- `network-testing/profiles/windows_clumsy_profiles.md`

Keep the same profile IDs and durations so results stay comparable.

## Required Outputs To Submit

For each profile:

1. `*_raw.csv`
2. `*_summary.json`
3. Updated `summary_table.csv`
4. Short notes on anomalies and recovery behavior

All outputs are written to `network-testing/results/` by default.

## Create Sendable Full Project Zip

From the project root (`trading-bot`) run:

- `./network-testing/scripts/package_project.sh`

This creates `trading-bot-complete-<timestamp>.zip` in the parent folder of the project root.

## Safety

- One profile at a time only.
- Clear shaping between runs.
- If data flow stalls, clear immediately and record timestamp.

Clear shaping command:

- `sudo ./network-testing/scripts/apply_tc_profile.sh <iface> clear`
