"""Evaluation helpers shared by the training scripts and the MLflow scripts."""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
    RocCurveDisplay,
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def get_scores(model, X):
    """Fraud probability when the model has one, otherwise the SVM decision score."""
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    return model.decision_function(X)


def compute_metrics(y_true, y_pred, y_score):
    """All required metrics. pr_auc is average precision (area under the precision-recall curve)."""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_score),
        "pr_auc": average_precision_score(y_true, y_score),
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
    }


def plot_confusion_matrix(y_true, y_pred, title, ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(4.5, 4))
    else:
        fig = ax.figure
    ConfusionMatrixDisplay.from_predictions(
        y_true, y_pred, display_labels=["Legit", "Fraud"], values_format="d",
        cmap="Blues", ax=ax, colorbar=False,
    )
    ax.set_title(title)
    return fig


def plot_roc_curve(y_true, y_score, title, ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 4))
    else:
        fig = ax.figure
    RocCurveDisplay.from_predictions(y_true, y_score, ax=ax, name=title)
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4)
    ax.set_title("ROC curve")
    return fig


def plot_pr_curve(y_true, y_score, title, ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 4))
    else:
        fig = ax.figure
    PrecisionRecallDisplay.from_predictions(y_true, y_score, ax=ax, name=title)
    ax.set_title("Precision-Recall curve")
    return fig


def feature_importance_table(pipeline, X, y):
    """Importance of every model input. Uses the model's own importances when available."""
    names = list(pipeline.named_steps["preprocess"].get_feature_names_out())
    model = pipeline.named_steps["model"]
    if hasattr(model, "feature_importances_"):
        values, source = model.feature_importances_, "model feature_importances_"
    elif hasattr(model, "coef_"):
        values, source = np.abs(model.coef_[0]), "absolute coefficient"
    else:
        transformed = pipeline[:-1].transform(X)
        result = permutation_importance(
            model, transformed, y, scoring="average_precision",
            n_repeats=2, random_state=42, n_jobs=1,
        )
        values, source = result.importances_mean, "permutation importance (average precision)"
    table = pd.DataFrame({"feature": names, "importance": values})
    table["source"] = source
    return table.sort_values("importance", ascending=False).reset_index(drop=True)


def plot_feature_importance(table, title, top=15):
    top_rows = table.head(top).iloc[::-1]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.barh(top_rows["feature"], top_rows["importance"], color="#2a6f97")
    ax.set_xlabel(f"Importance ({table['source'].iloc[0]})")
    ax.set_title(title)
    fig.tight_layout()
    return fig
