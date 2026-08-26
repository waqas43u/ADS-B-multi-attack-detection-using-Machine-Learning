from __future__ import annotations

import pickle
from dataclasses import asdict
import json

import pandas as pd

from adsb_attack_detection.config import parse_args
from adsb_attack_detection.data_preprocessing import load_and_preprocess_data, split_and_scale
from adsb_attack_detection.evaluation import evaluate_model, safe_name, save_summary_table
from adsb_attack_detection.models import build_models
from adsb_attack_detection.plots import save_learning_curve


def main():
    config = parse_args()
    config.output_dir.mkdir(parents=True, exist_ok=True)

    with open(config.output_dir / "config.json", "w", encoding="utf-8") as file:
        json.dump({key: str(value) for key, value in asdict(config).items()}, file, indent=2)

    print(f"Loading dataset: {config.data_path}")
    X, y, class_labels = load_and_preprocess_data(config.data_path, config.shuffle_random_state)
    X_train, X_test, y_train, y_test, scaler = split_and_scale(X, y, config.test_size, config.random_state)

    rows = []
    models = build_models(config.random_state)
    for model_name, model in models.items():
        print(f"Training and evaluating {model_name}...")
        metrics = evaluate_model(model_name, model, X_train, X_test, y_train, y_test, class_labels, config.output_dir)
        rows.append(metrics)

        if config.save_models:
            with open(config.output_dir / f"{safe_name(model_name)}_model.pkl", "wb") as file:
                pickle.dump(model, file)

        if config.save_learning_curves:
            save_learning_curve(model_name, model, X_train, y_train, config.output_dir, config.random_state)

    if config.save_models:
        with open(config.output_dir / "scaler.pkl", "wb") as file:
            pickle.dump(scaler, file)

    table = save_summary_table(rows, config.output_dir)
    formatted = table.copy()
    for column in formatted.columns[1:]:
        formatted[column] = pd.to_numeric(formatted[column], errors="coerce").map(lambda value: f"{value:.4f}" if pd.notna(value) else "")

    print("\nModel comparison table:")
    print(formatted.to_string(index=False))
    print(f"\nSaved outputs to: {config.output_dir}")


if __name__ == "__main__":
    main()
