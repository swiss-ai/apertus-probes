# run_cross_dataset_probes.py

import argparse
from probes_core import ProbeConfig, run_cross_dataset_probe_experiment


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train probes on one dataset and test on another dataset."
    )

    # Required
    parser.add_argument(
        "--train-dataset",
        type=str,
        required=True,
        help="Dataset name to train on. Example: mmlu_professional",
    )
    parser.add_argument(
        "--test-dataset",
        type=str,
        required=True,
        help="Dataset name to test on. Example: ARC-Challenge",
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
        help="Optional suffix for result filenames. If empty, uses '{train_dataset}_to_{test_dataset}'.",
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
        help="Random seed for models. Default: 52",
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
        "--no-normalize-features",
        action="store_true",
        help="Disable StandardScaler feature normalization.",
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
    use_logit_regression = args.use_logit
    # Always set transform_targets=False when using regression models (LogitRegression or Lasso)
    transform_targets = False
    
    config = ProbeConfig(
        selected_datasets=[args.train_dataset, args.test_dataset],  # For compatibility, not used in cross-dataset mode
        model_name=args.model_name,
        save_dir=args.save_dir,
        save_name=args.save_name if args.save_name else f"{args.train_dataset}_to_{args.test_dataset}",
        seed=args.seed,
        error_type=args.error_type,
        transform_targets=transform_targets,
        normalize_features=not args.no_normalize_features,
        nr_attempts=1,  # Not used in cross-dataset mode (single attempt)
        max_trials=args.max_trials,
        max_workers=args.max_workers,
        alphas=tuple(args.alphas),
        token_pos=args.token_pos,
        use_logit_regression=use_logit_regression,
    )

    print("\n=== Running cross-dataset probe experiment ===")
    print(f"Train dataset: {args.train_dataset}")
    print(f"Test dataset: {args.test_dataset}")
    print(config)
    df = run_cross_dataset_probe_experiment(config, args.train_dataset, args.test_dataset)
    print("Result rows:", len(df))


if __name__ == "__main__":
    main()

