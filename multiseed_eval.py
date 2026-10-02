import numpy as np
import pandas as pd
import generate_data
import forecast

SEEDS = [42, 43, 44, 45, 46]


def evaluate_one_seed(seed):
    """
    Regenerates all personas under a given seed, runs the finalized
    forecasting method, and returns per-persona coverage + pinball loss.
    This mirrors forecast.py's _run_evaluation(), but in-memory (no CSV
    read/write) so we can repeat it cleanly across seeds.
    """
    data = generate_data.generate_all_personas(seed=seed)

    weekly = (
        data.groupby("persona")
        .apply(lambda g: forecast.weekly_totals(g))
        .reset_index()
        .rename(columns={"amount": "weekly_income", "date": "week"})
    )

    results = {}
    for persona, group in weekly.groupby("persona"):
        fc = forecast.forecast_persona(group)
        confident_weeks = fc[fc["confident"]]

        if len(confident_weeks) == 0:
            results[persona] = {"coverage": np.nan, "pinball_loss": np.nan, "n": 0}
            continue

        inside_band = (
            (confident_weeks["actual"] >= confident_weeks["floor"]) &
            (confident_weeks["actual"] <= confident_weeks["optimistic"])
        )
        coverage = inside_band.mean()
        avg_pinball = forecast.pinball_loss(
            confident_weeks["actual"], confident_weeks["typical"], 0.5
        ).mean()

        results[persona] = {
            "coverage": coverage,
            "pinball_loss": avg_pinball,
            "n": len(confident_weeks),
        }

    return results


if __name__ == "__main__":
    all_results = []
    for seed in SEEDS:
        seed_results = evaluate_one_seed(seed)
        for persona, metrics in seed_results.items():
            all_results.append({"seed": seed, "persona": persona, **metrics})

    df = pd.DataFrame(all_results)

    print("Per-seed results:\n")
    print(df.to_string(index=False))

    print("\n\nAggregated across seeds (mean ± std):\n")
    summary = df.groupby("persona").agg(
        coverage_mean=("coverage", "mean"),
        coverage_std=("coverage", "std"),
        pinball_mean=("pinball_loss", "mean"),
        pinball_std=("pinball_loss", "std"),
        avg_n=("n", "mean"),
    )
    print(summary.to_string())

    df.to_csv("multiseed_results.csv", index=False)
    print("\nSaved to multiseed_results.csv")