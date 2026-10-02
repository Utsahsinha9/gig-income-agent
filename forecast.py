import numpy as np
import pandas as pd

# --- Step 1: load raw payout events, aggregate into weekly totals ---
df = pd.read_csv("synthetic_gig_income.csv", parse_dates=["date"])

def weekly_totals(persona_df):
    s = persona_df.set_index("date")["amount"].resample("W-SUN").sum()
    return s.fillna(0.0)

weekly = (
    df.groupby("persona")
    .apply(lambda g: weekly_totals(g))
    .reset_index()
    .rename(columns={"amount": "weekly_income", "date": "week"})
)

# --- Step 2: rolling-window quantile forecast, walk-forward (single week) ---
# This is the finalized method the project uses.
QUANTILES = [0.1, 0.5, 0.9]
WINDOW = 8
MIN_HISTORY = 4

def forecast_persona(persona_df):
    persona_df = persona_df.sort_values("week").reset_index(drop=True)
    income = persona_df["weekly_income"]

    results = []
    for i in range(len(persona_df)):
        history = income.iloc[max(0, i - WINDOW):i]

        if len(history) < MIN_HISTORY:
            floor = 0.0
            typical = income.iloc[:i].mean() if i > 0 else 0.0
            optimistic = income.iloc[:i].max() if i > 0 else 0.0
            confident = False
        else:
            floor, typical, optimistic = np.percentile(history, [q * 100 for q in QUANTILES])
            confident = True

        results.append({
            "week": persona_df["week"].iloc[i],
            "actual": income.iloc[i],
            "floor": floor,
            "typical": typical,
            "optimistic": optimistic,
            "confident": confident,
        })

    return pd.DataFrame(results)


# --- Parameterized version of the same method — used by ablation_floor.py
# to sweep the floor quantile. Confirmed via ablation that no floor_q in
# a reasonable range fixes the "honest zero floor" issue for volatile
# personas, since the zero-mass in the data itself exceeds the quantile
# range tested. Kept for reference / future ablation work. ---

def forecast_persona_param(persona_df, window=WINDOW, min_history=MIN_HISTORY,
                             floor_q=0.10, typical_q=0.50, optimistic_q=0.90):
    persona_df = persona_df.sort_values("week").reset_index(drop=True)
    income = persona_df["weekly_income"]

    results = []
    for i in range(len(persona_df)):
        history = income.iloc[max(0, i - window):i]

        if len(history) < min_history:
            floor = 0.0
            typical = income.iloc[:i].mean() if i > 0 else 0.0
            optimistic = income.iloc[:i].max() if i > 0 else 0.0
            confident = False
        else:
            floor, typical, optimistic = np.percentile(
                history, [floor_q * 100, typical_q * 100, optimistic_q * 100]
            )
            confident = True

        results.append({
            "week": persona_df["week"].iloc[i],
            "actual": income.iloc[i],
            "floor": floor,
            "typical": typical,
            "optimistic": optimistic,
            "confident": confident,
        })

    return pd.DataFrame(results)


# --- Multi-week horizon forecasting — NOT used in the final pipeline.
# Ablation showed this under-covers badly (as low as 29%) because
# summing overlapping rolling blocks makes consecutive "samples"
# highly correlated, shrinking the effective sample size feeding the
# quantile estimate. Kept here only as a documented dead end. ---

HORIZON_WEEKS = 3

def forecast_persona_horizon(persona_df, horizon=HORIZON_WEEKS, window=WINDOW, min_history=MIN_HISTORY):
    persona_df = persona_df.sort_values("week").reset_index(drop=True)
    income = persona_df["weekly_income"]

    block_sums = income.rolling(window=horizon).sum()

    results = []
    for i in range(len(persona_df)):
        actual = block_sums.iloc[i]
        if pd.isna(actual):
            continue

        history_end = i - horizon
        history = block_sums.iloc[max(0, history_end - window + 1): history_end + 1].dropna()

        if len(history) < min_history:
            past = block_sums.iloc[:i].dropna()
            floor = 0.0
            typical = past.mean() if len(past) else 0.0
            optimistic = past.max() if len(past) else 0.0
            confident = False
        else:
            floor, typical, optimistic = np.percentile(history, [10, 50, 90])
            confident = True

        results.append({
            "week": persona_df["week"].iloc[i],
            "actual": actual,
            "floor": floor,
            "typical": typical,
            "optimistic": optimistic,
            "confident": confident,
        })

    return pd.DataFrame(results)


# --- Naive baselines, for comparison ---
# Why these two specifically: "last-value" and "flat-mean" are the
# standard trivial baselines in forecasting — if our quantile method
# can't beat these, it isn't earning its complexity.

def naive_last_value_baseline(persona_df):
    """Predicts this week = last week's actual. No band, just a point forecast."""
    persona_df = persona_df.sort_values("week").reset_index(drop=True)
    income = persona_df["weekly_income"]

    predictions = income.shift(1)  # strictly uses only the prior week
    return pd.DataFrame({
        "week": persona_df["week"],
        "actual": income,
        "predicted": predictions,
    }).dropna()


def naive_mean_baseline(persona_df):
    """Predicts this week = mean of all PRIOR weeks (expanding window)."""
    persona_df = persona_df.sort_values("week").reset_index(drop=True)
    income = persona_df["weekly_income"]

    predictions = income.expanding().mean().shift(1)  # mean of weeks before this one
    return pd.DataFrame({
        "week": persona_df["week"],
        "actual": income,
        "predicted": predictions,
    }).dropna()


def mean_absolute_error(actual, predicted):
    return np.abs(actual - predicted).mean()


# --- Step 3: evaluation helpers ---
def pinball_loss(actual, predicted, tau):
    diff = actual - predicted
    return np.maximum(tau * diff, (tau - 1) * diff)


def _run_evaluation():
    all_forecasts = []
    for persona, group in weekly.groupby("persona"):
        fc = forecast_persona(group)
        fc["persona"] = persona
        all_forecasts.append(fc)

    all_forecasts = pd.concat(all_forecasts)

    for persona, fc in all_forecasts.groupby("persona"):
        confident_weeks = fc[fc["confident"]]
        if len(confident_weeks) == 0:
            print(f"\n{persona}: not enough history to evaluate yet")
            continue

        inside_band = (
            (confident_weeks["actual"] >= confident_weeks["floor"]) &
            (confident_weeks["actual"] <= confident_weeks["optimistic"])
        )
        coverage = inside_band.mean()

        avg_pinball = pinball_loss(
            confident_weeks["actual"], confident_weeks["typical"], 0.5
        ).mean()

        print(f"\n{persona}")
        print(f"  weeks evaluated: {len(confident_weeks)} (of {len(fc)} total, rest low-confidence)")
        print(f"  band coverage (target ~80%): {coverage:.1%}")
        print(f"  avg pinball loss (median forecast): {avg_pinball:.2f}")

    all_forecasts.to_csv("forecasts.csv", index=False)
    print("\nSaved detailed forecasts to forecasts.csv")


if __name__ == "__main__":
    _run_evaluation()