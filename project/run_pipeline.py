"""
run_pipeline.py
----------------
Single entry point that runs the entire MLOps pipeline end to end:

    load raw data (v1) -> preprocess (v2) -> feature engineering (v3)
    -> train + cross-validate all models -> evaluate -> log everything to MLflow

Usage:
    python run_pipeline.py                     # full pipeline, trains on v3
    python run_pipeline.py --dataset-version v2   # train on the cleaned (non feature-engineered) data instead
    python run_pipeline.py --skip-prep         # reuse existing data/v2, data/v3 files and just (re)train
"""

from __future__ import annotations

import argparse

from src import features, preprocessing, train


def main(dataset_version: str = "v3", seed: int = 42, skip_prep: bool = False) -> None:
    if not skip_prep:
        print("Step 1/3: preprocessing raw data -> data/v2 ...")
        preprocessing.run()

        print("Step 2/3: engineering features -> data/v3 ...")
        features.run()
    else:
        print("Skipping preprocessing/feature engineering (--skip-prep set).")

    print(f"Step 3/3: training & evaluating models on dataset version '{dataset_version}' ...")
    train.run(dataset_version=dataset_version, seed=seed)

    print("\nPipeline finished. Run `mlflow ui` and open http://localhost:5000 to inspect results.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the full Telco churn MLOps pipeline.")
    parser.add_argument("--dataset-version", default="v3", choices=["v2", "v3"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-prep", action="store_true", help="Skip preprocessing/feature engineering steps.")
    args = parser.parse_args()
    main(dataset_version=args.dataset_version, seed=args.seed, skip_prep=args.skip_prep)
