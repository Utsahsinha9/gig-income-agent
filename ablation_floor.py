import numpy as np
import pandas as pd
import forecast

FLOOR_QS = [0.05, 0.10, 0.15, 0.20, 0.25]
WINDOWS = [6, 8, 10, 12]
OPTIMISTIC_Q = 0.90


def evaluate_config(floor_q, window):
    pooled_actual, pooled_typical, pooled_floor, pooled_optimistic = [], [], [], []
    low_confidence_pct = {}

    for persona, group in forecast.weekly.groupby("persona"):
        fc = forecast.forecast_persona_param(
            group, window=window, floor_q=floor_q, optimistic_q=OPTIMISTIC_Q
        )

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

    target_coverage = OPTIMISTIC_Q - floor_q

    if len(pooled_actual) == 0:
        return {
            "floor_q": floor_q, "window": window, "target_coverage": target_coverage,
            "coverage": np.nan, "pinball_loss": np.nan, "n_confident_weeks": 0,
            "pct_zero_floor": np.nan,
            **{f"low_conf_pct_{p}": v for p, v in low_confidence_pct.items()},
        }

    inside_band = (pooled_actual >= pooled_floor) & (pooled_actual <= pooled_optimistic)
    coverage = inside_band.mean()
    avg_pinball = forecast.pinball_loss(pooled_actual, pooled_typical, 0.5).mean()
    pct_zero_floor = (pooled_floor == 0).mean()

    return {
        "floor_q": floor_q, "window": window, "target_coverage": target_coverage,
        "coverage": coverage, "pinball_loss": avg_pinball,
        "n_confident_weeks": len(pooled_actual), "pct_zero_floor": pct_zero_floor,
        **{f"low_conf_pct_{p}": v for p, v in low_confidence_pct.items()},
    }


if __name__ == "__main__":
    results = []
    for fq in FLOOR_QS:
        for w in WINDOWS:
            results.append(evaluate_config(fq, w))

    results_df = pd.DataFrame(results)
    pd.set_option("display.width", 180)
    pd.set_option("display.max_columns", None)
    print(results_df.to_string(index=False))

    results_df.to_csv("ablation_floor_results.csv", index=False)
    print("\nSaved to ablation_floor_results.csv")