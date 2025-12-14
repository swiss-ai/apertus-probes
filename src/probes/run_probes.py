# run_probes.py

import argparse
from probes_core import ProbeConfig, run_probe_experiment


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a single probe experiment on cached activations."
    )

    # Required
    parser.add_argument(
        "--datasets",
        type=str,
        nargs="+",
        required=True,
        help="List of dataset names to mix (space-separated). Example: mmlu_professional ARC-Challenge",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        required=True,
        help="Model name used in the cache paths. Example: Apertus-8B-Instruct-2509",
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        required=True,
        help="Base directory where caches and results are stored.",
    )

    # Optional
    parser.add_argument(
        "--save-name",
        type=str,
        default="",
        help="Optional suffix for result filenames. If empty, uses '+'.join(datasets).",
    )
    parser.add_argument(
        "--error-type",
        type=str,
        default="SM",
        choices=["SM", "CE"],
        help="Error type: SM (1-softmax) or CE (cross-entropy). Default: SM",
    )
    parser.add_argument(
        "--token-pos",
        type=str,
        default="exact",
        choices=["exact", "last", "both"],
        help="Token position to use: 'exact' 'last' or 'both'. Default: exact",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=52,
        help="Random seed for train/test split and models. Default: 52",
    )
    parser.add_argument(
        "--nr-attempts",
        type=int,
        default=5,
        help="Number of random train/test splits (attempts) per layer. Default: 5",
    )
    parser.add_argument(
        "--max-trials",
        type=int,
        default=5,
        help="Max refits per attempt when all coefficients are zero. Default: 5",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=25,
        help="Max threads for parallel layer/model training. Default: 25",
    )
    parser.add_argument(
        "--alphas",
        type=float,
        nargs="+",
        default=[0.5, 0.25, 0.1, 0.05],
        help="L1 alphas for Lasso regression. Default: 0.5 0.25 0.1 0.05",
    )

    # Booleans: transform targets / normalize features
    parser.add_argument(
        "--no-transform-targets",
        action="store_true",
        help="Disable logit transform of regression targets.",
    )
    parser.add_argument(
        "--normalize-features",
        action="store_true",
        help="Enable StandardScaler feature normalization (default: disabled).",
    )
    parser.add_argument(
        "--use-logit",
        action="store_true",
        help="Use LogitRegression instead of standard Lasso for regression models. Default: Lasso",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    # When using LogitRegression or Lasso, don't transform targets (preprocessing)
    # LogitRegression does logit transform internally, Lasso doesn't need it
    # Always set transform_targets=False when using regression models (LogitRegression or Lasso)
    transform_targets = False
    
    config = ProbeConfig(
        selected_datasets=args.datasets,
        model_name=args.model_name,
        save_dir=args.save_dir,
        save_name=args.save_name,
        seed=args.seed,
        error_type=args.error_type,
        transform_targets=transform_targets,
        normalize_features=args.normalize_features,  # Default is False, use flag to enable
        nr_attempts=args.nr_attempts,
        max_trials=args.max_trials,
        max_workers=args.max_workers,
        alphas=tuple(args.alphas),
        token_pos=args.token_pos,
    )

    print("\n=== Running probe experiment ===")
    print(config)
    df = run_probe_experiment(config)
    print("Result rows:", len(df))


if __name__ == "__main__":
    main()
