from datetime import date, timedelta
import joblib
import pandas as pd
from .config import MODEL_PATH, OUT_DIR
from .preprocess import load_data, split_data, get_X_y


def recommend(row):
    """Return list of (driver, action) pairs for one employee (a pandas row)."""
    t = []
    if row["on_time_delivery_rate"] < 0.75:
        t.append(("Low on-time delivery", "Sprint-planning & estimation workshop"))
    if row["bug_count"] >= 6 and row["code_review_score"] < 3.3:
        t.append(("High bugs + low review score", "Pair programming + code-quality course"))
    if row["training_hours"] < 25:
        t.append(("Low training hours", "Assign targeted certification (AWS CCP / Azure DP-900)"))
    if row["unplanned_absences"] >= 4:
        t.append(("High unplanned absences", "Manager 1:1 on workload and wellbeing"))
    if row["peer_feedback_score"] < 3.0:
        t.append(("Weak peer feedback", "Collaboration / communication coaching"))
    if row["manager_score"] < 3.0:
        t.append(("Low mid-year manager score", "Goal-setting and coaching check-in"))
    return t


def run():
    model = joblib.load(MODEL_PATH)
    _, test = split_data(load_data())
    X_test, _ = get_X_y(test)
    proba = model.predict_proba(X_test)
    low_i = list(model.classes_).index("Low")

    scored = test.assign(predicted_band=model.predict(X_test), p_low=proba[:, low_i])
    top = scored.sort_values("p_low", ascending=False).head(20)

    plans = []
    for _, r in top.iterrows():
        t = recommend(r)
        plans.append({
            "employee_id": r["employee_id"], "department": r["department"],
            "predicted_band": r["predicted_band"], "p_low": round(r["p_low"], 2),
            "key_drivers": "; ".join(d for d, _ in t[:3]) or "No rule triggered, manual review",
            "actions": "; ".join(a for _, a in t[:2]) or "Manager discussion",
            "review_date": date.today() + timedelta(days=30),
        })
    out = pd.DataFrame(plans)
    out.to_csv(OUT_DIR / "employee_action_plans.csv", index=False)
    print(out.head(5).to_string(index=False))
