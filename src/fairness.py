import joblib
import pandas as pd
from sklearn.metrics import f1_score
from .config import MODEL_PATH, OUT_DIR, TARGET
from .preprocess import load_data, split_data, get_X_y


def group_report(test, pred, group_col, positive="High"):
    d = test.assign(pred=pred)
    rows = []
    for g, sub in d.groupby(group_col):
        rows.append({group_col: g, "n": len(sub),
                     "high_selection_rate": (sub["pred"] == positive).mean(),
                     "accuracy": (sub["pred"] == sub[TARGET]).mean(),
                     "macro_f1": f1_score(sub[TARGET], sub["pred"], average="macro")})
    r = pd.DataFrame(rows)
    # Four-fifths rule: ratio to the most-selected group, flag below 0.8
    r["disparate_impact"] = r["high_selection_rate"] / r["high_selection_rate"].max()
    r["flag"] = r["disparate_impact"] < 0.8
    return r.round(3)


def run():
    model = joblib.load(MODEL_PATH)
    _, test = split_data(load_data())
    X_test, _ = get_X_y(test)          # gender is NOT a model input
    pred = model.predict(X_test)

    out = []
    for col in ["gender", "department", "job_level"]:
        rep = group_report(test, pred, col)
        print(f"\n[fairness] by {col}\n{rep.to_string()}")
        rep.insert(0, "group_col", col)
        out.append(rep.rename(columns={col: "group"}))
    pd.concat(out).to_csv(OUT_DIR / "fairness_report.csv", index=False)
