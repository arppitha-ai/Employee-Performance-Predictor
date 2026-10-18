import joblib
import streamlit as st
from src.config import MODEL_PATH
from src.preprocess import load_data, get_X_y
from src.recommend import recommend

st.set_page_config(page_title="Employee Performance Predictor", layout="wide")
st.title("📊 Employee Performance Predictor")
st.caption("Synthetic data. A decision-support tool for HR, not an automated decision-maker.")


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


model = load_model()
df = load_data()
X, _ = get_X_y(df)
df["predicted_band"] = model.predict(X)
df["p_low"] = model.predict_proba(X)[:, list(model.classes_).index("Low")]

dept = st.sidebar.multiselect("Department", sorted(df["department"].unique()),
                              default=sorted(df["department"].unique()))
view = df[df["department"].isin(dept)]

c1, c2, c3 = st.columns(3)
c1.metric("Employees", len(view))
c2.metric("Predicted High", int((view["predicted_band"] == "High").sum()))
c3.metric("Predicted Low (at risk)", int((view["predicted_band"] == "Low").sum()))

st.subheader("Predicted band distribution")
st.bar_chart(view["predicted_band"].value_counts())

st.subheader("Employee lookup")
emp = st.selectbox("Employee ID", view["employee_id"])
row = view[view["employee_id"] == emp].iloc[0]
st.write(f"**Predicted band:** {row['predicted_band']}  |  **Risk of Low:** {row['p_low']:.0%}")
recs = recommend(row)
if recs:
    st.write("**Suggested drivers and actions**")
    for d, a in recs[:3]:
        st.write(f"- {d} → {a}")
else:
    st.write("No rule-based risk drivers triggered.")
