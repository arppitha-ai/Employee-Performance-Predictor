import numpy as np
import pandas as pd
from .config import DATA_PATH, TARGET, SEED


def z(s):
    """Standardise a column (mean 0, std 1)."""
    s = pd.Series(s)
    return (s - s.mean()) / s.std()


def generate_employee_data(n=2000, seed=SEED):
    rng = np.random.default_rng(seed)

    # ---- Demographics & role context ----
    experience = np.clip(rng.gamma(2.5, 3.0, n), 0, 35).round(1)
    age = np.clip(22 + experience + rng.normal(0, 2, n), 21, 60).round()
    gender = rng.choice(["Male", "Female"], n, p=[0.55, 0.45])
    education = rng.choice(["Bachelors", "Masters", "PhD"], n, p=[0.65, 0.30, 0.05])
    department = rng.choice(["Engineering", "QA", "Data", "HR", "Sales", "Support"],
                            n, p=[0.35, 0.12, 0.15, 0.08, 0.18, 0.12])
    job_level = np.select([experience < 3, experience < 7, experience < 12],
                          ["Junior", "Mid", "Senior"], default="Lead")
    manager_tenure = rng.uniform(0.5, 10, n).round(1)

    # ---- Engagement ----
    training_hours = np.clip(rng.normal(40, 18, n), 0, 120).round(1)
    certifications = rng.poisson(1.2, n)
    hackathons = rng.poisson(0.6, n)

    # ---- Work signals ----
    projects = np.clip(rng.poisson(3.5, n), 1, 10)
    on_time = np.clip(rng.normal(0.82, 0.10, n), 0.3, 1.0).round(3)
    task_delay = np.clip(rng.normal(5 - 20 * (on_time - 0.82), 3, n), -3, 30).round(1)

    # ---- Quality ----
    code_review = np.clip(rng.normal(3.6, 0.6, n) + 0.004 * training_hours, 1, 5).round(2)
    bug_count = rng.poisson(np.clip(10 - 1.5 * code_review, 0.5, None))
    qa_defect_density = (bug_count / projects).round(2)

    # ---- Productivity ----
    story_points = np.clip(rng.normal(50 + 5 * projects, 12, n), 5, None).round()
    billable_ratio = np.clip(rng.normal(0.75, 0.10, n), 0.3, 1.0).round(3)

    # ---- Attendance ----
    sick_days = rng.poisson(4, n)
    unplanned_abs = rng.poisson(2, n)
    login_hours = np.clip(rng.normal(8.2, 0.8, n), 5, 11).round(2)

    # ---- Feedback ----
    peer = np.clip(3.6 + 2 * (on_time - 0.82) + rng.normal(0, 0.5, n), 1, 5).round(2)
    manager_score = np.clip(3.5 + 0.5 * (code_review - 3.6) + rng.normal(0, 0.6, n), 1, 5).round(2)
    kudos = rng.poisson(np.clip(2 + (peer - 3.5) * 2, 0.2, None))

    # ---- HR signals ----
    promotions = rng.choice([0, 1, 2], n, p=[0.70, 0.25, 0.05])
    salary_band = rng.integers(1, 6, n)

    # ---- THE SIMULATION RULE: how "true" performance is created ----
    # gender/age/department are deliberately NOT in this formula.
    score = (0.9 * z(on_time) + 0.6 * z(code_review) - 0.5 * z(bug_count)
             + 0.7 * z(manager_score) + 0.5 * z(peer) + 0.3 * z(training_hours)
             + 0.2 * z(certifications) + 0.3 * z(story_points)
             - 0.3 * z(unplanned_abs) - 0.2 * z(task_delay) + 0.15 * z(kudos)
             + rng.normal(0, 0.9, n))

    # 25% Low, 50% Medium, 25% High
    band = pd.qcut(score, [0, 0.25, 0.75, 1.0], labels=["Low", "Medium", "High"])

    df = pd.DataFrame({
        "employee_id": [f"EMP{i:04d}" for i in range(1, n + 1)],
        "age": age, "gender": gender, "education": education,
        "experience_years": experience, "department": department,
        "job_level": job_level, "manager_tenure": manager_tenure,
        "projects_count": projects, "avg_task_delay_days": task_delay,
        "on_time_delivery_rate": on_time, "bug_count": bug_count,
        "code_review_score": code_review, "qa_defect_density": qa_defect_density,
        "story_points_completed": story_points, "billable_hours_ratio": billable_ratio,
        "training_hours": training_hours, "certifications_count": certifications,
        "hackathons_participated": hackathons, "sick_days": sick_days,
        "unplanned_absences": unplanned_abs, "avg_login_hours": login_hours,
        "peer_feedback_score": peer, "manager_score": manager_score,
        "kudos_count": kudos, "promotions_in_2y": promotions,
        "salary_percentile_band": salary_band, TARGET: band.astype(str),
    })

    # Real data has gaps: blank out ~3% of a few columns
    for col in ["peer_feedback_score", "code_review_score", "training_hours"]:
        df.loc[rng.random(n) < 0.03, col] = np.nan
    return df


def run():
    df = generate_employee_data()
    df.to_csv(DATA_PATH, index=False)
    print(f"[data] saved {df.shape} to {DATA_PATH}")
    print(df[TARGET].value_counts())
