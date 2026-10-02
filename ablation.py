import numpy as np
import pandas as pd
import forecast

# --- Sweep configuration ---
# Why these ranges: horizon below 2 reproduces the original ₹0-floor
# problem; above 5 starts smoothing away almost all week-to-week
# signal, which stops being a "weekly budgeting" tool at all.
# Window below 6 gives quantiles too little data; above 12 starts
# reacting too slowly to a genuine shift in someone's income.

HORIZONS = [2, 3, 4, 5]
WINDOWS = [6, 8, 10, 12]


def evaluate_config(horizon, window):
    """
    Runs forecast_persona_horizon for every persona at one
    (horizon, window) setting, and returns pooled + per-persona metrics.
    """
    pooled_actual, pooled_typical, pooled_floor, pooled_optimistic = [], [], [], []
    low_confidence_pct = {}

    for persona, group in forecast.weekly.groupby("persona"):
        fc = forecast.forecast_persona_horizon(group, horizon=horizon, window=window)

        total_weeks = len(fc)
        confident_weeks = fc[fc["confident"]]
        low_confidence_pct[persona] = (
            1 - len(confident_weeks) / total_weeks if total_weeks else np.nan
        )

        pooled_actual.extend(confident_weeks["actual"])
        pooled_typical.extend(confident_weeks["typical"])
        pooled_floor.extend(confident_weeks["floor"])
        pooled_optimistic.extend(confident_weeks["optimistic"])

    pooled_actual = np.array(pooled_actual)
    pooled_typical = np.array(pooled_typical)
    pooled_floor = np.array(pooled_floor)
    pooled_optimistic = np.array(pooled_optimistic)

    if len(pooled_actual) == 0:
        return {
            "horizon": horizon, "window": window,
            "coverage": np.nan, "pinball_loss": np.nan,
            "n_confident_weeks": 0,
            **{f"low_conf_pct_{p}": v for p, v in low_confidence_pct.items()},
        }

    inside_band = (pooled_actual >= pooled_floor) & (pooled_actual <= pooled_optimistic)
    coverage = inside_band.mean()
    avg_pinball = forecast.pinball_loss(pooled_actual, pooled_typical, 0.5).mean()

    return {
        "horizon": horizon, "window": window,
        "coverage": coverage, "pinball_loss": avg_pinball,
        "n_confident_weeks": len(pooled_actual),
        **{f"low_conf_pct_{p}": v for p, v in low_confidence_pct.items()},
    }


if __name__ == "__main__":
    results = []
    for h in HORIZONS:
        for w in WINDOWS:
            results.append(evaluate_config(h, w))

    results_df = pd.DataFrame(results)
    pd.set_option("display.width", 160)
    pd.set_option("display.max_columns", None)
    print(results_df.to_string(index=False))

    results_df.to_csv("ablation_results.csv", index=False)
    print("\nSaved to ablation_results.csv")