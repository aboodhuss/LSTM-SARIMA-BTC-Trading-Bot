"""
train_kaggle_sarima_profile.py – One-shot Kaggle SARIMA profile generator.

Run this script **once** when setting up the repo.  It downloads the BTC/USD
minute-level Kaggle dataset, fits a grid of SARIMA models, and saves the
best configuration to `backend/sarima_kaggle_profile.json`.

That profile is then loaded at runtime by sarima_preprocessor.py to
reconstruct missing candles before LSTM inference.

Kaggle datasets used
--------------------
  1. Minute-level BTC data  :  https://www.kaggle.com/datasets/swaptr/bitcoin-historical-data
  2. Daily BTC price data   :  https://www.kaggle.com/datasets/hasanyiitakbulut/bitcoin-btc-historical-price-data-2020-2026

Usage
-----
    cd backend
    python train_kaggle_sarima_profile.py

The output (sarima_kaggle_profile.json) should be committed to the repo so
other team members do not need to re-download or re-fit the profile.
"""

from __future__ import annotations

import json
from pathlib import Path

from sarima_preprocessor import (
    PROFILE_PATH,
    ensure_kaggle_dataset,
    load_kaggle_minute_frame,
    save_profile,
    select_best_profile,
)


def train_profile(output_path: Path = PROFILE_PATH) -> dict[str, object]:
    """
    Download the Kaggle BTC minute dataset, run the SARIMA grid search,
    and write the winning profile to disk.

    Returns
    -------
    dict
        The full profile dictionary (source metadata + model parameters).
    """
    # Step 1: Download (or retrieve cached) Kaggle dataset via kagglehub.
    dataset_dir = ensure_kaggle_dataset()

    # Step 2: Load and parse the minute-level CSV.
    minute_df = load_kaggle_minute_frame(dataset_dir)

    # Step 3: Run the SARIMA grid search and pick the best by AIC.
    profile = select_best_profile(minute_df)

    # Step 4: Save to JSON for the runtime pre-processor to load.
    save_profile(profile, output_path)
    return profile


if __name__ == "__main__":
    profile = train_profile()
    print(json.dumps(profile, indent=2))
