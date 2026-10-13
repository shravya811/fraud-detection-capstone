"""Builds the full model pipeline: feature engineering -> preprocessing -> (sampler) -> model."""
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier

from src import config
from src.features import FLAG_COLUMNS, LOG_SCALED_COLUMNS, FraudFeatureBuilder, signed_log1p

IMBALANCE_STRATEGIES = ["none", "class_weight", "undersample", "oversample", "smote"]

# Hyperparameter search spaces (small on purpose: the training set has millions of rows)
PARAM_GRIDS = {
    "logistic_regression": {"model__C": [0.1, 1.0, 10.0]},
    "decision_tree": {
        "model__max_depth": [6, 10, 14],
        "model__min_samples_leaf": [10, 50],
    },
    "random_forest": {
        "model__max_depth": [10, 16],
        "model__min_samples_leaf": [10, 50],
    },
    "gradient_boosting": {
        "model__learning_rate": [0.05, 0.1],
        "model__max_iter": [100, 200],
        "model__max_leaf_nodes": [31, 63],
    },
    "svm": {"model__C": [0.01, 0.1, 1.0]},
}
# How many random combinations to try per model
SEARCH_N_ITER = {
    "logistic_regression": 3,
    "decision_tree": 4,
    "random_forest": 3,
    "gradient_boosting": 4,
    "svm": 3,
}


def build_preprocessor():
    """Log-transform + scale money columns, scale the hour, one-hot encode the type."""
    money = Pipeline([
        ("log", FunctionTransformer(signed_log1p, feature_names_out="one-to-one")),
        ("scale", StandardScaler()),
    ])
    return ColumnTransformer(
        [
            ("money", money, LOG_SCALED_COLUMNS),
            ("hour", StandardScaler(), ["hour_of_day"]),
            ("flags", "passthrough", FLAG_COLUMNS),
            ("type", OneHotEncoder(handle_unknown="ignore", sparse_output=False), ["type"]),
        ],
        remainder="drop",
    )


def build_estimator(name, balanced):
    """One scikit-learn model. balanced=True turns on class weights."""
    weight = "balanced" if balanced else None
    seed = config.RANDOM_STATE
    if name == "logistic_regression":
        return LogisticRegression(max_iter=200, class_weight=weight, random_state=seed)
    if name == "decision_tree":
        return DecisionTreeClassifier(class_weight=weight, random_state=seed)
    if name == "random_forest":
        # Each tree sees a 20% bootstrap sample; across 100 trees the whole training set is used.
        return RandomForestClassifier(
            n_estimators=100, max_samples=0.2, n_jobs=-1, random_state=seed,
            class_weight="balanced_subsample" if balanced else None,
        )
    if name == "gradient_boosting":
        # HistGradientBoosting = scikit-learn's gradient boosting, built for millions of rows.
        return HistGradientBoostingClassifier(class_weight=weight, random_state=seed)
    if name == "svm":
        # Linear SVM. A kernel SVC cannot train on millions of rows.
        return LinearSVC(dual=False, class_weight=weight, random_state=seed)
    raise ValueError(f"Unknown model name: {name}")


def build_sampler(strategy):
    """Resampling methods from imbalanced-learn (applied to training data only)."""
    if strategy in ("none", "class_weight"):
        return None
    from imblearn.over_sampling import SMOTE, RandomOverSampler
    from imblearn.under_sampling import RandomUnderSampler

    seed = config.RANDOM_STATE
    if strategy == "undersample":
        return RandomUnderSampler(sampling_strategy=0.1, random_state=seed)  # 10 legit : 1 fraud
    if strategy == "oversample":
        return RandomOverSampler(sampling_strategy=0.05, random_state=seed)  # fraud = 5% of legit
    if strategy == "smote":
        return SMOTE(sampling_strategy=0.05, k_neighbors=5, random_state=seed)
    raise ValueError(f"Unknown imbalance strategy: {strategy}")


def build_pipeline(name, strategy="class_weight", estimator=None):
    """Full pipeline for one model. Resampling only happens inside fit(), never in predict()."""
    steps = [("features", FraudFeatureBuilder()), ("preprocess", build_preprocessor())]
    sampler = build_sampler(strategy)
    if estimator is None:
        estimator = build_estimator(name, strategy == "class_weight")
    model = estimator
    if sampler is None:
        return Pipeline(steps + [("model", model)])
    from imblearn.pipeline import Pipeline as ImbPipeline
    return ImbPipeline(steps + [("sampler", sampler), ("model", model)])
