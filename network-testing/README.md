# Network Testing Package Index

Use this file as the folder map.

## Start Here

- `NETWORKING_TEST_README.md`: team-facing guide you can send directly.
- `docs/NETWORK_TEST_GUIDE.md`: full technical runbook mapped to sections 4.1 to 5.6.

## Folder Layout

- `docs/`: detailed execution methodology and interpretation rules.
- `profiles/`: profile matrix and Windows Clumsy equivalents.
- `scripts/`: profile application, metric capture, session runner, packaging.
- `templates/`: copy-paste team message.
- `results/`: generated output from profile runs.

## Primary Commands

From the project root (`trading-bot`):

1. Baseline run on Linux:
   - `./network-testing/scripts/run_profile_session.sh <iface> baseline 180`
2. Example shaped run:
   - `./network-testing/scripts/run_profile_session.sh <iface> latency_300_jitter_50 90`
3. Package the project:
   - `./network-testing/scripts/package_project.sh`
   - Output is `trading-bot-complete-<timestamp>.zip` in the parent folder of the project root

## Notes

- Replace `<iface>` with the network interface, for example `eth0` or `enp3s0`.
- `run_profile_session.sh` automatically clears shaping after each run.
- Local test mode without Linux shaping:
  - `./network-testing/scripts/run_profile_session.sh --no-shaping lo0 baseline_smoke 0.05 http://127.0.0.1:8000`
