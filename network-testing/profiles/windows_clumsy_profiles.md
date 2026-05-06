# Windows Clumsy Profile Mapping

Use this file when Linux `tc` is unavailable.

## Steps

1. Launch Clumsy as Administrator.
2. Filter traffic to the target process or destination as needed.
3. Enable only one impairment type at a time for each session.
4. Match duration to the profile matrix.
5. Disable all impairments between runs and verify recovery.

## Profile Equivalents

- `baseline`: all Clumsy impairments disabled.
- `latency_150`: Lag enabled, value `150 ms`.
- `latency_300_jitter_50`: Lag enabled around `300 ms`, add variability around `50 ms` if available in your Clumsy build.
- `latency_500_jitter_100`: Lag enabled around `500 ms`, variability around `100 ms`.
- `loss_1`: Drop enabled, `1%`.
- `loss_3`: Drop enabled, `3%`.
- `loss_5`: Drop enabled, `5%`.
- `bandwidth_5mbit`: Throttle enabled to approximately `5 Mbps`.
- `bandwidth_2mbit`: Throttle enabled to approximately `2 Mbps`.
- `bandwidth_1mbit`: Throttle enabled to approximately `1 Mbps`.

## Recording Rule

For every run, store:

1. Screenshot of active Clumsy settings.
2. Session output files from `capture_metrics.py`.
3. Short note of start and stop times.
