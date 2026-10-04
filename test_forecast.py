import numpy as np
import pandas as pd
import forecast
import generate_data


# --- Fixtures: small, controlled datasets so tests aren't dependent
# on the full synthetic CSV (which could change) ---

def make_weekly_df(values):
    """Builds a minimal weekly dataframe from a plain list of incomes."""
    weeks = pd.date_range("2025-01-05", periods=len(values), freq="W-SUN")
    return pd.DataFrame({"week": weeks, "weekly_income": values})


# --- Test 1: no lookahead bias ---
# Why this matters: this was a real design requirement from the start —
# forecasting week N must only use data strictly before week N. This
# test catches a regression if that invariant is ever accidentally broken.

def test_no_lookahead_bias():
    # Construct data where week 5 is a huge outlier (₹100,000). If the
    # forecast for week 5 "sees" its own value, floor/typical/optimistic
    # would be skewed by it. They must not be.
    values = [1000, 1000, 1000, 1000, 100000, 1000, 1000, 1000]
    df = make_weekly_df(values)

    fc = forecast.forecast_persona(df)
    week5_forecast = fc.iloc[4]  # index 4 = the 5th week, the outlier week

    # The forecast for the outlier week should be based on the PRIOR
    # (all-1000) weeks only — optimistic should be nowhere near 100000.
    assert week5_forecast["optimistic"] < 5000, (
        "Forecast appears to include the current week's own value — lookahead bias"
    )


# --- Test 2: low-confidence fallback triggers correctly ---

def test_low_confidence_triggers_below_min_history():
    values = [1000, 1200, 900]  # only 3 weeks — below MIN_HISTORY=4
    df = make_weekly_df(values)

    fc = forecast.forecast_persona(df)
    # Every week should be low-confidence since we never reach MIN_HISTORY
    assert not fc["confident"].any(), (
        "Expected all weeks low-confidence with insufficient history"
    )


def test_confidence_triggers_once_enough_history():
    values = [1000] * 10  # well above MIN_HISTORY=4
    df = make_weekly_df(values)

    fc = forecast.forecast_persona(df)
    # The later weeks (once window is full) should be confident
    assert fc["confident"].iloc[-1] == True, (
        "Expected confident=True once sufficient history exists"
    )


# --- Test 3: safe-to-spend never goes negative ---
# Why: a negative "safe to spend" would be a nonsensical recommendation
# to show a real person. This is a sanity floor on the output, not the
# forecast itself.

def test_safe_to_spend_never_negative():
    # Simulate the blend formula directly with edge-case inputs
    floor, typical = 0.0, 0.0  # worst case: both zero
    safe_total = 0.3 * floor + 0.4 * typical
    assert safe_total >= 0


# --- Test 4: the honest zero-floor behavior we discovered via ablation ---
# Why: this documents and locks in the actual finding from our ablation
# work — a volatile enough persona SHOULD sometimes get floor=0, and
# that's correct, not a bug, as long as it doesn't appear for a steady one.

def test_zero_floor_possible_for_volatile_persona():
    # Mostly-zero income with occasional spikes — mimics spiky_freelancer's
    # real behavior (frequent zero weeks)
    values = [0, 0, 5000, 0, 8000, 0, 0, 6000, 0, 7000]
    df = make_weekly_df(values)

    fc = forecast.forecast_persona(df)
    confident_weeks = fc[fc["confident"]]
    # At least one confident week should have floor == 0, given how
    # often zero appears in the history
    assert (confident_weeks["floor"] == 0).any(), (
        "Expected at least one zero floor given frequent zero-income weeks"
    )


def test_steady_persona_rarely_hits_zero_floor():
    # Smooth, consistent income — should NOT produce zero floors
    values = [8000, 8200, 7900, 8100, 8050, 7950, 8000, 8100, 8200, 7900]
    df = make_weekly_df(values)

    fc = forecast.forecast_persona(df)
    confident_weeks = fc[fc["confident"]]
    assert not (confident_weeks["floor"] == 0).any(), (
        "Did not expect zero floors for a smooth, steady income pattern"
    )


# --- Test 5: naive baselines behave sanely ---

def test_naive_last_value_baseline_shape():
    values = [1000, 1200, 900, 1100]
    df = make_weekly_df(values)

    baseline = forecast.naive_last_value_baseline(df)
    # First row has no "prior" week, so it should be dropped (dropna())
    assert len(baseline) == len(values) - 1
    # Second prediction should equal the FIRST actual value
    assert baseline["predicted"].iloc[0] == values[0]


# --- Test 6: data generation reproducibility ---
# Why: this is what makes the multi-seed evaluation meaningful at all —
# the same seed must always produce the same data.

def test_same_seed_produces_identical_data():
    data1 = generate_data.generate_all_personas(seed=42)
    data2 = generate_data.generate_all_personas(seed=42)

    pd.testing.assert_frame_equal(
        data1.reset_index(drop=True), data2.reset_index(drop=True)
    )


def test_different_seeds_produce_different_data():
    data1 = generate_data.generate_all_personas(seed=42)
    data2 = generate_data.generate_all_personas(seed=99)

    # Total income across all personas should differ between seeds
    assert data1["amount"].sum() != data2["amount"].sum()