import matplotlib
matplotlib.use("Agg")  # save plots without needing a display
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.feature_selection import mutual_info_classif
from .config import IMG_DIR, OUT_DIR, TARGET, ORDER
from .preprocess import load_data, validate, split_data, get_X_y


def run():
    df = load_data()
    validate(df)
    train, _ = split_data(df)          # EDA on TRAIN only, never peek at test
    print(train.head())
    print(train.describe().T)

    # 1. Class balance
    plt.figure(figsize=(5, 4))
    sns.countplot(data=train, x=TARGET, order=ORDER)
    plt.title("Performance Band Distribution")
    plt.savefig(IMG_DIR / "01_class_balance.png", dpi=150, bbox_inches="tight")
    plt.close()

    # 2. Histograms of key features
    feats = ["on_time_delivery_rate", "code_review_score", "manager_score",
             "peer_feedback_score", "training_hours", "bug_count",
             "unplanned_absences", "story_points_completed"]
    train[feats].hist(figsize=(14, 8), bins=25)
    plt.suptitle("Key Feature Distributions")
    plt.savefig(IMG_DIR / "02_histograms.png", dpi=150, bbox_inches="tight")
    plt.close()

    # 3. Correlation heatmap
    plt.figure(figsize=(12, 9))
    sns.heatmap(train.select_dtypes("number").corr(), cmap="coolwarm", center=0)
    plt.title("Correlation Heatmap")
    plt.savefig(IMG_DIR / "03_correlation.png", dpi=150, bbox_inches="tight")
    plt.close()

    # 4. Do top signals track the bands?
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    for a, col in zip(ax, ["on_time_delivery_rate", "manager_score", "training_hours"]):
        sns.boxplot(data=train, x=TARGET, y=col, order=ORDER, ax=a)
        a.set_title(f"{col} by band")
    plt.savefig(IMG_DIR / "04_boxplots.png", dpi=150, bbox_inches="tight")
    plt.close()

    # 5. Department vs band
    pd.crosstab(train["department"], train[TARGET], normalize="index")[ORDER] \
        .plot(kind="bar", stacked=True, figsize=(8, 4))
    plt.title("Band share by department")
    plt.savefig(IMG_DIR / "05_department_band.png", dpi=150, bbox_inches="tight")
    plt.close()

    # 6. Mutual information ranking
    X, y = get_X_y(train)
    num = X.select_dtypes("number")
    mi = mutual_info_classif(num.fillna(num.median()), y, random_state=13)
    mi = pd.Series(mi, index=num.columns).sort_values(ascending=False)
    mi.to_csv(OUT_DIR / "mutual_information.csv")
    print("\n[eda] Top 10 features by mutual information:\n", mi.head(10))
    print(f"[eda] plots saved in {IMG_DIR}")
