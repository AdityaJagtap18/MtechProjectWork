#!/usr/bin/env python3
"""Train one GraphSAGE run and save it (model + preprocessing + config +
training history + run metadata) without evaluating it -- for evaluation
in a separate step, see `scripts/evaluate_graphsage.py`. For the common
case of training-then-evaluating in one process, prefer
`scripts/run_graphsage_experiment.py`.

Usage:
    python scripts/train_graphsage.py --config configs/graphsage.yaml --seed 42
"""

from __future__ import annotations

import argparse
import os

import torch

from scm_dataset.modeling.config import load_config
from scm_dataset.modeling.experiment import build_run_metadata, new_run_dir, save_config, write_json
from scm_dataset.modeling.pipeline import prepare
from scm_dataset.modeling.train import class_balance_summary, train_graphsage


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/graphsage.yaml")
    parser.add_argument("--seed", type=int, default=None, help="Override the config's seed.")
    args = parser.parse_args()

    config = load_config(args.config)
    seed = args.seed if args.seed is not None else config.seed

    prepared = prepare(config)
    train_result = train_graphsage(prepared, seed=seed)

    run_dir = new_run_dir(config.experiment.output_dir, f"hetero_graphsage_seed{seed}_trainonly")
    save_config(run_dir, config)
    torch.save(train_result.model.state_dict(), os.path.join(run_dir, "model.pt"))
    prepared.preprocessor.save(os.path.join(run_dir, "preprocessing"))
    train_result.history.to_csv(os.path.join(run_dir, "training_history.csv"), index=False)
    metadata = build_run_metadata(
        config, prepared.benchmark, prepared.frames, seed,
        split_summary=class_balance_summary(prepared.examples),
        training_duration_seconds=train_result.training_duration_seconds,
        extra={"best_epoch": train_result.best_epoch, "stopped_early": train_result.stopped_early, "pos_weight": train_result.pos_weight},
    )
    write_json(os.path.join(run_dir, "run_metadata.json"), metadata)
    os.rmdir(os.path.join(run_dir, "plots"))  # unused by this train-only entry point

    print(f"Trained (best_epoch={train_result.best_epoch}, best_val_pr_auc={train_result.best_val_pr_auc:.4f}) -> {run_dir}")


if __name__ == "__main__":
    main()
