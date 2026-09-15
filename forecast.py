import numpy as np
import pandas as pd

# --- Step 1: load raw payout events, aggregate into weekly totals ---
# Why weekly: irregular per-event timing is the raw reality, but
# budgeting decisions happen on a weekly cadence ("what can I spend
# this week"), so that's the unit we forecast at.

df = pd.read_csv("synthetic_gig_income.csv", parse_dates=["date"])

def weekly_totals(persona_df):
    s = persona_df.set_index("date")["amount"].resample("W-SUN").sum()
    # resample() only creates rows for weeks that HAD at least one
    # event nearby; fillna(0) makes zero-income weeks explicit rather
    # than silently missing — a zero week is real signal, not a gap.
    return s.fillna(0.0)

weekly = (
    df.groupby("persona")
    .apply(lambda g: weekly_totals(g))
    .reset_index()
    .rename(columns={"amount": "weekly_income", "date": "week"})
)

# --- Step 2: rolling-window quantile forecast, walk-forward ---
# Why shift(1): the forecast for week N must only see weeks BEFORE N.
# Using week N's own data to "forecast" week N is lookahead bias —
# it would make the model look accurate while being useless in
# practice, since in reality you don't know this week's income yet.

QUANTILES = [0.1, 0.5, 0.9]   # floor, typical, optimistic
WINDOW = 8                     # trailing 8 weeks of history
MIN_HISTORY = 4                # below this, flag low-confidence instead of guessing

def forecast_persona(persona_df):
    persona_df = persona_df.sort_values("week").reset_index(drop=True)
    income = persona_df["weekly_income"]

    results = []
    for i in range(len(persona_df)):
        history = income.iloc[max(0, i - WINDOW):i]  # strictly before week i

        if len(history) < MIN_HISTORY:
            # Not enough history to trust a tight quantile estimate.
            # Widen the band artificially and mark it explicitly —
            # this IS the "honest exception" behavior, applied here
            # instead of hiding a shaky number behind false confidence.
            floor, typical, optimistic = 0.0, income.iloc[:i].mean() if i > 0 else 0.0, income.max() if i > 0 else 0.0
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

# --- Step 3: run it for every persona, evaluate honestly ---

def pinball_loss(actual, predicted, tau):
    diff = actual - predicted
    return np.maximum(tau * diff, (tau - 1) * diff)

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

    # Coverage: how often the actual landed inside [floor, optimistic]
    # An 80% band (10th-90th percentile) SHOULD contain ~80% of actuals
    # if the forecast is honest, not overconfident or too wide.
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