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


# --- Multi-week horizon forecasting ---
# Why: a single week's floor can hit ₹0 just from one unlucky
# zero-payout week (pure Poisson noise), producing a useless
# "never spend anything" recommendation. Real people naturally
# smooth irregular income across a few weeks rather than resetting
# every Monday — so we forecast a rolling HORIZON (e.g. the next
# 2 weeks combined) instead of a single isolated week.

HORIZON_WEEKS = 3

def forecast_persona_horizon(persona_df, horizon=HORIZON_WEEKS, window=WINDOW, min_history=MIN_HISTORY):
    persona_df = persona_df.sort_values("week").reset_index(drop=True)
    income = persona_df["weekly_income"]

    # trailing horizon-week sums: block_sums[i] = total income over
    # the `horizon` weeks ending at week i (NaN until enough weeks exist)
    block_sums = income.rolling(window=horizon).sum()

    results = []
    for i in range(len(persona_df)):
        actual = block_sums.iloc[i]
        if pd.isna(actual):
            continue  # not enough weeks yet to even form this block

        # history: past horizon-week block sums, using only blocks
        # that ended strictly BEFORE this one started — no lookahead
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