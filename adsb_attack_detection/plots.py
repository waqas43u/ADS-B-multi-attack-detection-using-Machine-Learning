from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import ShuffleSplit, learning_curve

from .evaluation import safe_name


def save_learning_curve(model_name: str, estimator, X, y, output_dir: Path, random_state: int):
    cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=random_state)
    train_sizes, train_scores, valid_scores = learning_curve(
        estimator,
        X,
        y,
        cv=cv,
        n_jobs=-1,
        train_sizes=np.linspace(0.1, 1.0, 5),
    )

    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    valid_mean = np.mean(valid_scores, axis=1)
    valid_std = np.std(valid_scores, axis=1)

    plt.figure(figsize=(8, 5))
    plt.title(f"Training vs Validation Curve for {model_name}")
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    plt.grid(True)
    plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1, color="r")
    plt.fill_between(train_sizes, valid_mean - valid_std, valid_mean + valid_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_mean, "o-", color="r", label="Training score")
    plt.plot(train_sizes, valid_mean, "o-", color="g", label="Validation score")
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(output_dir / f"{safe_name(model_name)}_learning_curve.png", dpi=300)
    plt.close()
