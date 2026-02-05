import argparse
import numpy as np
import pandas as pd

def clamp01(x):
    return np.clip(x, 0.0, 1.0)

def generate_one_user_series(rng, weeks, profile):
    """
    Genera una serie temporal semanal (weeks) con features numéricas no invasivas.
    profile: 0=LOW, 1=MEDIUM, 2=HIGH
    """
    t = np.arange(weeks)

    # Base (comportamiento "normal")
    avg_daily_login_time = rng.normal(loc=55, scale=10, size=weeks)            # minutos/día
    login_time_variance  = rng.normal(loc=0.25, scale=0.08, size=weeks)        # variabilidad (0-1 aprox)
    days_active_per_week = rng.normal(loc=5.2, scale=0.7, size=weeks)          # 0-7
    completion_rate      = rng.normal(loc=0.82, scale=0.08, size=weeks)        # 0-1
    schedule_irregularity= rng.normal(loc=0.22, scale=0.08, size=weeks)        # 0-1
    late_submission_rate = rng.normal(loc=0.10, scale=0.05, size=weeks)        # 0-1 (proporción de entregas tardías)

    # Modulación por perfil (simula deterioro)
    if profile == 0:  # LOW: estable / mejora leve
        drift = rng.normal(loc=-0.01, scale=0.01, size=weeks)
        schedule_irregularity += drift
        late_submission_rate += rng.normal(loc=-0.005, scale=0.005, size=weeks)

    elif profile == 1:  # MEDIUM: estrés creciente gradual
        drift = 0.015 * t + rng.normal(loc=0.0, scale=0.03, size=weeks)
        schedule_irregularity += drift
        login_time_variance += 0.008 * t
        completion_rate -= 0.01 * t
        late_submission_rate += 0.02 * t + rng.normal(0, 0.01, size=weeks)

    else:  # HIGH: burnout / ruptura más brusca
        # Primera mitad relativamente normal, segunda mitad cae fuerte
        pivot = weeks // 2
        step = np.zeros(weeks)
        step[pivot:] = 1.0

        schedule_irregularity += 0.15 * t + 0.35 * step + rng.normal(0, 0.05, weeks)
        login_time_variance  += 0.02 * t + 0.20 * step + rng.normal(0, 0.03, weeks)
        completion_rate      -= 0.02 * t + 0.25 * step + rng.normal(0, 0.03, weeks)
        days_active_per_week -= 0.03 * t + 0.80 * step + rng.normal(0, 0.20, weeks)
        late_submission_rate += 0.04 * t + 0.30 * step + rng.normal(0, 0.03, weeks)

    # Saneamiento de rangos
    avg_daily_login_time = np.clip(avg_daily_login_time, 0, 240)
    login_time_variance  = clamp01(login_time_variance)
    days_active_per_week = np.clip(days_active_per_week, 0, 7)
    completion_rate      = clamp01(completion_rate)
    schedule_irregularity= clamp01(schedule_irregularity)
    late_submission_rate = clamp01(late_submission_rate)

    return pd.DataFrame({
        "week": t,
        "avg_daily_login_time": avg_daily_login_time,
        "login_time_variance": login_time_variance,
        "days_active_per_week": days_active_per_week,
        "assignment_completion_rate": completion_rate,
        "schedule_irregularity": schedule_irregularity,
        "late_submission_rate": late_submission_rate,
    })

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--users", type=int, default=1200)
    parser.add_argument("--weeks", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=str, default="data/synthetic_academic_data.csv")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)

    # Balanceado por clase
    # 0=LOW, 1=MEDIUM, 2=HIGH
    profiles = rng.choice([0, 1, 2], size=args.users, p=[0.45, 0.35, 0.20])

    rows = []
    for user_id, profile in enumerate(profiles):
        df = generate_one_user_series(rng, args.weeks, profile)
        df.insert(0, "user_id", user_id)
        df["risk_class"] = profile  # etiqueta supervisada
        rows.append(df)

    out_df = pd.concat(rows, ignore_index=True)

    # Mezcla filas para evitar orden por usuario
    out_df = out_df.sample(frac=1.0, random_state=args.seed).reset_index(drop=True)

    out_df.to_csv(args.out, index=False)
    print(f"[OK] Saved: {args.out}")
    print(out_df.head(10))

if __name__ == "__main__":
    main()
