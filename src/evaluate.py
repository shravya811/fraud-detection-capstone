"""Steps 9-10 - evaluate every tuned model on the untouched test set and pick the best.

Run from the project root:  python -m src.evaluate
Outputs (reports/): model_comparison.csv/.md, best_model.json, and PNG charts in figures/.
The best model is chosen by cross-validated PR-AUC from training, NOT by test results.
"""
import json

import joblib
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd

from src import config
from src.data_utils import class_counts, load_clean_data, split_xy, time_split
from src.metrics import (
    compute_metrics,
    feature_importance_table,
    get_scores,
    plot_confusion_matrix,
    plot_feature_importance,
    plot_pr_curve,
    plot_roc_curve,
)


def main():
    matplotlib.use("Agg")
    df = load_clean_data()
    _, test_df, cutoff = time_split(df, config.TEST_ROW_FRACTION)
    X_test, y_test = split_xy(test_df)
    print(f"Test rows: {len(test_df):,} (steps > {cutoff}) {class_counts(y_test)}")

    tuning = json.loads((config.REPORTS_DIR / "tuning_results.json").read_text())
    names = [n for n in config.MODEL_NAMES if n in tuning]
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    rows, scores_by_model = [], {}
    cm_fig, cm_axes = plt.subplots(1, len(names), figsize=(4.2 * len(names), 4))
    cm_axes = [cm_axes] if len(names) == 1 else list(cm_axes)
    for ax, name in zip(cm_axes, names):
        model = joblib.load(config.CANDIDATES_DIR / f"{name}.pkl")
        scores = get_scores(model, X_test)
        predictions = model.predict(X_test)
        scores_by_model[name] = scores
        result = compute_metrics(y_test, predictions, scores)
        result["model"] = name
        result["cv_pr_auc"] = tuning[name]["cv_average_precision"]
        rows.append(result)
        plot_confusion_matrix(y_test, predictions, name, ax=ax)

        importance = feature_importance_table(model, X_test, y_test)
        importance.to_csv(config.REPORTS_DIR / f"feature_importance_{name}.csv", index=False)
        fig = plot_feature_importance(importance, f"Feature importance - {name}")
        fig.savefig(config.FIGURES_DIR / f"feature_importance_{name}.png", dpi=120)
        plt.close(fig)
        print(f"{name}: recall={result['recall']:.4f} precision={result['precision']:.4f} "
              f"f1={result['f1']:.4f} pr_auc={result['pr_auc']:.4f} "
              f"roc_auc={result['roc_auc']:.4f}")

    cm_fig.tight_layout()
    cm_fig.savefig(config.FIGURES_DIR / "confusion_matrices.png", dpi=120)
    plt.close(cm_fig)

    for kind, plotter in [("roc_curves", plot_roc_curve), ("pr_curves", plot_pr_curve)]:
        fig, ax = plt.subplots(figsize=(7, 5.5))
        for name in names:
            plotter(y_test, scores_by_model[name], name, ax=ax)
        ax.set_title("ROC curves - all models" if kind == "roc_curves"
                     else "Precision-Recall curves - all models")
        fig.tight_layout()
        fig.savefig(config.FIGURES_DIR / f"{kind}.png", dpi=120)
        plt.close(fig)

    table = pd.DataFrame(rows).set_index("model")
    table = table[["accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc", "cv_pr_auc",
                   "true_positives", "false_negatives", "false_positives", "true_negatives"]]
    table.to_csv(config.REPORTS_DIR / "model_comparison.csv")
    (config.REPORTS_DIR / "model_comparison.md").write_text(table.round(4).to_markdown()
                                                            if _has_tabulate() else
                                                            table.round(4).to_string())

    best = table["cv_pr_auc"].idxmax()
    (config.REPORTS_DIR / "best_model.json").write_text(json.dumps({
        "best_model": best,
        "chosen_by": "highest cross-validated PR-AUC (average precision) on the training set",
        "test_metrics": {k: float(v) for k, v in table.loc[best].items()},
    }, indent=2))
    print("\n", table.round(4).to_string())
    print(f"\nBest model by cross-validated PR-AUC: {best}")
    print("Reminder: accuracy is misleading here (predicting 'legitimate' every time scores "
          f"{(y_test == 0).mean() * 100:.2f}%). Compare recall, precision, F1 and PR-AUC.")


def _has_tabulate():
    try:
        import tabulate  # noqa: F401
        return True
    except ImportError:
        return False


if __name__ == "__main__":
    main()
