from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


CLASS_LABELS = [
    "Ghost injection",
    "No attack",
    "Aircraft standing still",
    "Aircraft displaying false information",
    "Jumping aircraft",
    "Transponder code alteration",
    "Trajectory modification",
    "Non-responsive aircraft",
    "Aircraft spoofing",
    "Message Delay",
]

DROP_COLUMNS = [
    "id",
    "airlineId",
    "flightNumber",
    "origin",
    "destination",
    "takeOffTime",
    "taxi_start",
]


def load_and_preprocess_data(data_path: Path, shuffle_random_state=None) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
    df = pd.read_csv(data_path)
    existing_drop_columns = [column for column in DROP_COLUMNS if column in df.columns]
    df = df.drop(existing_drop_columns, axis=1)
    df = df.drop_duplicates()
    df = df.dropna(how="any", axis=0)

    # Preserves the notebook behavior. Pass --shuffle-random-state for deterministic shuffling.
    df = df.sample(frac=1, random_state=shuffle_random_state)

    label_to_number = {label: i for i, label in enumerate(CLASS_LABELS)}
    df["attackType_mapped"] = df["attackType"].map(label_to_number)
    df = df.drop("attackType", axis=1)

    # Preserves the notebook feature selection exactly.
    X = df.iloc[:, 1:-1]
    y = df.iloc[:, -1]
    return X, y, CLASS_LABELS


def split_and_scale(X, y, test_size: float, random_state: int):
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
    )
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler
