"""Command-line interface."""
from __future__ import annotations
import argparse
import json
import logging
import sys
from .pipeline import predict, train


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ml-predict", description="Train, compare, evaluate, and run tabular ML models.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    training = subparsers.add_parser("train", help="Train and evaluate candidate models from a CSV file.")
    training.add_argument("--data", required=True, help="Path to input CSV dataset.")
    training.add_argument("--target", required=True, help="Name of the target column.")
    training.add_argument("--task", choices=["auto", "classification", "regression"], default="auto")
    training.add_argument("--output-dir", default="artifacts")
    training.add_argument("--test-size", type=float, default=0.2)
    training.add_argument("--cv", type=int, default=5)
    training.add_argument("--random-state", type=int, default=42)
    training.add_argument("--metric", help="Selection metric; defaults to weighted F1 or R².")
    inference = subparsers.add_parser("predict", help="Generate predictions from a saved model artifact.")
    inference.add_argument("--model", required=True, help="Path to artifacts/model.joblib.")
    inference.add_argument("--data", required=True, help="CSV containing feature columns.")
    inference.add_argument("--output", default="predictions.csv")
    return parser


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    args = _parser().parse_args()
    try:
        if args.command == "train":
            result = train(args.data, args.target, args.output_dir, args.task, args.test_size, args.cv,
                           args.random_state, args.metric)
            print(json.dumps({"selected_model": result["selected_model"], "task": result["task"],
                              "test_metrics": result["test_metrics"]}, indent=2))
        else:
            result = predict(args.model, args.data, args.output)
            print(f"Wrote {len(result)} predictions to {args.output}")
    except (FileNotFoundError, ValueError, RuntimeError) as error:
        logging.error("%s", error)
        sys.exit(2)


if __name__ == "__main__":
    main()
