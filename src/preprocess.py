import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, RobustScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from .config import DATA_PATH, TARGET, SEED

# Columns that must never be model inputs
NON_FEATURES = ["employee_id", "gender", TARGET]

RULES = {
    "experience_years": (0, 45),
    "on_time_delivery_rate": (0, 1),
    "manager_score": (0, 5),
    "peer_feedback_score": (0, 5),
    "avg_task_delay_days": (-10, 60),
}


def load_data():
    df = pd.read_csv(DATA_PATH)
    assert df["employee_id"].is_unique, "Duplicate employee_id found!"
    return df


def validate(df):
    print("[validate] duplicates:", df.duplicated().sum())
    for col, (lo, hi) in RULES.items():
        bad = df[(df[col] < lo) | (df[col] > hi)]
        print(f"[validate] {col}: {len(bad)} out of range")
    miss = df.isna().sum()
    print("[validate] missing values:\n", miss[miss > 0])


def split_data(df):
    # One row per employee, so a row split is also an employee split
    return train_test_split(df, test_size=0.2, random_state=SEED, stratify=df[TARGET])


def get_X_y(df):
    X = df.drop(columns=[c for c in NON_FEATURES if c in df.columns])
    return X, df[TARGET]


def build_preprocessor(X):
    # exclude="number" works on both old pandas (object) and pandas 3 (str)
    cat_cols = X.select_dtypes(exclude="number").columns.tolist()
    num_cols = X.select_dtypes(include="number").columns.tolist()
    num_pipe = Pipeline([("imputer", SimpleImputer(strategy="median")),
                         ("scaler", RobustScaler())])
    cat_pipe = Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                         ("ohe", OneHotEncoder(handle_unknown="ignore"))])
    return ColumnTransformer([("num", num_pipe, num_cols),
                              ("cat", cat_pipe, cat_cols)])
