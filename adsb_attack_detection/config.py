from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class Config:
    data_path: Path = Path("Excel_ADS_B_Dataset_344000.csv")
    output_dir: Path = Path("results")
    test_size: float = 0.20
    random_state: int = 42
    shuffle_random_state: Optional[int] = None
    save_models: bool = True
    save_learning_curves: bool = True


def parse_args() -> Config:
    parser = argparse.ArgumentParser(description="Train KNN, SVM, and XGBoost for ADS-B attack detection.")
    parser.add_argument("--data", dest="data_path", type=Path, default=Path("Excel_ADS_B_Dataset_344000.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    parser.add_argument("--test-size", type=float, default=0.20)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--shuffle-random-state", type=int, default=None)
    parser.add_argument("--no-save-models", action="store_false", dest="save_models")
    parser.add_argument("--no-learning-curves", action="store_false", dest="save_learning_curves")
    args = parser.parse_args()
    return Config(
        data_path=args.data_path,
        output_dir=args.output_dir,
        test_size=args.test_size,
        random_state=args.random_state,
        shuffle_random_state=args.shuffle_random_state,
        save_models=args.save_models,
        save_learning_curves=args.save_learning_curves,
    )
