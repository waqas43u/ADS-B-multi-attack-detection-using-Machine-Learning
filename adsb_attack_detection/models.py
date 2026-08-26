from __future__ import annotations

from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier


def build_models(random_state: int):
    return {
        "KNN": KNeighborsClassifier(n_neighbors=5, metric="minkowski", p=2),
        "SVM": SVC(kernel="rbf", C=1, gamma="scale", random_state=random_state),
        "XGBoost": XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=random_state),
    }
