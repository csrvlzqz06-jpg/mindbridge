import argparse
import numpy as np
import pandas as pd

FEATURES = [
    "avg_daily_login_time",
    "login_time_variance",
    "days_active_per_week",
    "assignment_completion_rate",
    "schedule_irregularity",
    "self_reported_stress",
]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_csv", type=str, default="data/synthetic_academic_data.csv")
    parser.add_argument("--weeks", type=int, default=8)  # time_steps
    parser.add_argument("--out_npz", type=str, default="data/windows_dataset.npz")
    args = parser.parse_args()

    df = pd.read_csv(args.in_csv)

    # Aseguramos que cada usuario tenga weeks filas
    users = df["user_id"].unique()

    X_list, y_list = [], []
    for uid in users:
        u = df[df["user_id"] == uid].sort_values("week")
        if len(u) != args.weeks:
            continue
        X = u[FEATURES].to_numpy(dtype=np.float32)
        y = int(u["risk_class"].iloc[0])
        X_list.append(X)
        y_list.append(y)

    X = np.stack(X_list, axis=0)  # (N, T, F)
    y = np.array(y_list, dtype=np.int64)

    # Normalización simple por feature (z-score global)
    mean = X.mean(axis=(0,1), keepdims=True)
    std  = X.std(axis=(0,1), keepdims=True) + 1e-6
    Xn = (X - mean) / std

    np.savez(args.out_npz, X=Xn, y=y, mean=mean, std=std, features=np.array(FEATURES))
    print(f"[OK] Saved windows dataset: {args.out_npz}")
    print("X shape:", Xn.shape, "y shape:", y.shape)

if __name__ == "__main__":
    main()
