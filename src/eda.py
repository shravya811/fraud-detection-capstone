"""Step 3 - exploratory data analysis. Saves PNG charts to reports/figures/.

Run from the project root:  python -m src.eda
"""
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from src import config
from src.data_utils import load_clean_data
from src.features import FraudFeatureBuilder

LEGIT_COLOR, FRAUD_COLOR = "#2a6f97", "#d62828"


def _save(fig, name):
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    path = config.FIGURES_DIR / name
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def plot_class_balance(df):
    counts = df[config.TARGET].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.bar(["Legitimate", "Fraud"], counts.values, color=[LEGIT_COLOR, FRAUD_COLOR])
    ax.set_yscale("log")
    ax.set_ylabel("Transactions (log scale)")
    ax.set_title(f"Class balance (fraud = {df[config.TARGET].mean() * 100:.3f}%)")
    for i, value in enumerate(counts.values):
        ax.text(i, value, f"{value:,}", ha="center", va="bottom")
    return _save(fig, "01_class_balance.png")


def plot_type_patterns(df):
    summary = df.groupby("type", observed=True)[config.TARGET].agg(["size", "sum"])
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    summary["size"].plot.bar(ax=axes[0], color=LEGIT_COLOR)
    axes[0].set_title("Transactions by type")
    axes[0].set_ylabel("Count")
    summary["sum"].plot.bar(ax=axes[1], color=FRAUD_COLOR)
    axes[1].set_title("Fraud transactions by type")
    axes[1].set_ylabel("Fraud count")
    return _save(fig, "02_type_patterns.png")


def plot_amount_distribution(df):
    fig, ax = plt.subplots(figsize=(7, 4))
    for label, color, name in [(0, LEGIT_COLOR, "Legitimate"), (1, FRAUD_COLOR, "Fraud")]:
        amounts = df.loc[df[config.TARGET] == label, "amount"]
        ax.hist(np.log10(amounts + 1), bins=60, alpha=0.6, density=True, color=color, label=name)
    ax.set_xlabel("log10(amount + 1)")
    ax.set_ylabel("Density")
    ax.set_title("Transaction amount: fraud vs legitimate")
    ax.legend()
    return _save(fig, "03_amount_distribution.png")


def plot_hourly_patterns(df):
    hour = (df["step"].astype("int64") - 1) % 24
    summary = df.groupby(hour)[config.TARGET].agg(["size", "mean"])
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].bar(summary.index, summary["size"], color=LEGIT_COLOR)
    axes[0].set_title("Transaction volume by hour of day")
    axes[0].set_xlabel("Hour of day ((step - 1) % 24)")
    axes[0].set_ylabel("Transactions")
    axes[1].bar(summary.index, summary["mean"] * 100, color=FRAUD_COLOR)
    axes[1].set_title("Fraud rate by hour of day")
    axes[1].set_xlabel("Hour of day ((step - 1) % 24)")
    axes[1].set_ylabel("Fraud rate (%)")
    return _save(fig, "04_hourly_patterns.png")


def plot_fraud_over_time(df):
    per_step = df.groupby("step", observed=True)[config.TARGET].sum()
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(per_step.index, per_step.values, color=FRAUD_COLOR, linewidth=1)
    ax.set_xlabel("Step (hour since start)")
    ax.set_ylabel("Fraud transactions")
    ax.set_title("Fraud transactions over time")
    return _save(fig, "05_fraud_over_time.png")


def _box_stats(values, label):
    """Exact box-plot numbers (whiskers = 1st/99th percentile), computed on every row."""
    q1, median, q3 = np.quantile(values, [0.25, 0.5, 0.75])
    low, high = np.quantile(values, [0.01, 0.99])
    return {"label": label, "q1": q1, "med": median, "q3": q3, "whislo": low,
            "whishi": high, "fliers": []}


def plot_feature_vs_target(df):
    """Engineered features by class, plus the correlation matrix (returned for plotting)."""
    features = FraudFeatureBuilder().transform(df[config.RAW_FEATURES])
    y = df[config.TARGET].to_numpy()
    numeric = features.drop(columns=["type"]).astype("float32")
    del features
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for ax, col in zip(axes, ["errorBalanceOrig", "errorBalanceDest", "amount_to_origin_balance"]):
        signed = (np.sign(numeric[col]) * np.log1p(numeric[col].abs())).to_numpy()
        stats = [_box_stats(signed[y == 0], "Legit"), _box_stats(signed[y == 1], "Fraud")]
        boxes = ax.bxp(stats, patch_artist=True, showfliers=False)
        for patch, color in zip(boxes["boxes"], [LEGIT_COLOR, FRAUD_COLOR]):
            patch.set_facecolor(color)
        ax.set_title(col)
        ax.set_ylabel("signed log1p (box = 25-75%, whiskers = 1-99%)")
    path = _save(fig, "06_engineered_features_vs_target.png")
    numeric[config.TARGET] = y.astype("float32")
    return path, numeric.corr()


def plot_correlation(corr):
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax,
                annot_kws={"size": 7})
    ax.set_title("Correlation matrix (features and isFraud)")
    return _save(fig, "07_correlation_heatmap.png")


def run_all_plots():
    df = load_clean_data()
    paths = [
        plot_class_balance(df),
        plot_type_patterns(df),
        plot_amount_distribution(df),
        plot_hourly_patterns(df),
        plot_fraud_over_time(df),
    ]
    path, corr = plot_feature_vs_target(df)
    paths.append(path)
    paths.append(plot_correlation(corr))
    return paths


if __name__ == "__main__":
    matplotlib.use("Agg")
    for p in run_all_plots():
        print("Saved", p)
