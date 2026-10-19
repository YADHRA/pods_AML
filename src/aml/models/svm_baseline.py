"""Linear SVM-style baseline for the PODS-AML project."""

import inspect

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


SEEDS = (42, 43, 44)
MAX_ITER = 20

# Fixed six-configuration grid. Do not change after running the experiment.
CONFIGS = [
    {"alpha": alpha, "class_weight": weight}
    for alpha in (1e-5, 1e-4, 1e-3)
    for weight in (None, "balanced")
]


def build_preprocessor(features, categorical_features):
    """Fit preprocessing separately on training data only."""
    categorical = [
        col for col in categorical_features if col in features
    ]
    numeric = [col for col in features if col not in categorical]

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler(with_mean=False)),
    ])

    encoder_args = {
        "handle_unknown": "ignore",
        "dtype": np.float32,
    }

    # Support both newer and older scikit-learn versions.
    if "sparse_output" in inspect.signature(
        OneHotEncoder
    ).parameters:
        encoder_args["sparse_output"] = True
    else:
        encoder_args["sparse"] = True

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(**encoder_args)),
    ])

    transformers = []
    if numeric:
        transformers.append(
            ("numeric", numeric_pipeline, numeric)
        )
    if categorical:
        transformers.append(
            ("categorical", categorical_pipeline, categorical)
        )

    return ColumnTransformer(
        transformers,
        remainder="drop",
        sparse_threshold=1.0,
    )


def make_classifier(config, seed):
    """Create a linear hinge-loss classifier."""
    return SGDClassifier(
        loss="hinge",
        penalty="l2",
        alpha=config["alpha"],
        class_weight=config["class_weight"],
        max_iter=MAX_ITER,
        tol=None,
        shuffle=True,
        random_state=seed,
        average=True,
    )


def score(model, transformed_features):
    """Return decision scores; probabilities are not required."""
    return model.decision_function(transformed_features)