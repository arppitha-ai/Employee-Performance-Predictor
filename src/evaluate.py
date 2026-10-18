import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import (accuracy_score, classification_report,
                             ConfusionMatrixDisplay, confusion_matrix, f1_score)
from .config import MODEL_PATH, IMG_DIR, OUT_DIR, ORDER, SEED
from .preprocess import load_data, split_data, get_X_y


def run():
    model = joblib.load(MODEL_PATH)
    _, test = split_data(load_data())
    X_test, y_test = get_X_y(test)
    pred = model.predict(X_test)

    acc = accuracy_score(y_test, pred)
    f1 = f1_score(y_test, pred, average="macro")
    report = classification_report(y_test, pred, labels=ORDER, digits=3)
    print(f"[eval] Accuracy: {acc:.3f} | Macro-F1: {f1:.3f}\n{report}")
    (OUT_DIR / "classification_report.txt").write_text(
        f"Accuracy: {acc:.3f}\nMacro-F1: {f1:.3f}\n\n{report}")

    # Confusion matrix
    cm = confusion_matrix(y_test, pred, labels=ORDER)
    ConfusionMatrixDisplay(cm, display_labels=ORDER).plot(cmap="Blues")
    plt.title("Confusion Matrix (hold-out test)")
    plt.savefig(IMG_DIR / "06_confusion_matrix.png", dpi=150, bbox_inches="tight")
    plt.close()

    # Permutation importance on the whole pipeline (raw input columns)
    r = permutation_importance(model, X_test, y_test, n_repeats=5,
                               scoring="f1_macro", random_state=SEED, n_jobs=-1)
    imp = pd.Series(r.importances_mean, index=X_test.columns).sort_values(ascending=False)
    imp.to_csv(OUT_DIR / "feature_importance.csv")
    print("[eval] Top 15 drivers:\n", imp.head(15))
    imp.head(15)[::-1].plot(kind="barh", figsize=(8, 6))
    plt.title("Top 15 Performance Drivers (permutation importance)")
    plt.xlabel("Drop in macro-F1 when feature is shuffled")
    plt.savefig(IMG_DIR / "07_feature_importance.png", dpi=150, bbox_inches="tight")
    plt.close()
