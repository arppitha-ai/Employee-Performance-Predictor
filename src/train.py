import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from .config import MODEL_PATH, SEED
from .preprocess import load_data, split_data, get_X_y, build_preprocessor


def run():
    df = load_data()
    train, _ = split_data(df)
    X, y = get_X_y(train)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

    # Baseline: simple, interpretable
    log = Pipeline([("pre", build_preprocessor(X)),
                    ("clf", LogisticRegression(max_iter=1000, class_weight="balanced"))])
    base = cross_val_score(log, X, y, cv=cv, scoring="f1_macro")
    print(f"[train] Baseline LogReg CV macro-F1: {base.mean():.3f} (+/- {base.std():.3f})")

    # Tuned Random Forest
    rf = Pipeline([("pre", build_preprocessor(X)),
                   ("clf", RandomForestClassifier(class_weight="balanced", random_state=SEED))])
    grid = {"clf__n_estimators": [200, 400],
            "clf__max_depth": [8, 12, None],
            "clf__min_samples_leaf": [1, 3, 5]}
    gs = GridSearchCV(rf, grid, scoring="f1_macro", cv=cv, n_jobs=-1, verbose=1)
    gs.fit(X, y)
    print(f"[train] RF best CV macro-F1: {gs.best_score_:.3f}")
    print("[train] best params:", gs.best_params_)

    # Keep whichever is better
    if base.mean() > gs.best_score_:
        best = log.fit(X, y)
        print("[train] Logistic Regression won, saving it")
    else:
        best = gs.best_estimator_
        print("[train] Random Forest won, saving it")
    joblib.dump(best, MODEL_PATH)
    print(f"[train] model saved to {MODEL_PATH}")
