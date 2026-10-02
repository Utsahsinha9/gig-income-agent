import forecast

print("Comparing our quantile method's 'typical' forecast against naive baselines")
print("Metric: Mean Absolute Error (MAE) — lower is better\n")

for persona, group in forecast.weekly.groupby("persona"):
    # Our method's point forecast = the "typical" (median) value
    our_fc = forecast.forecast_persona(group)
    our_confident = our_fc[our_fc["confident"]]
    our_mae = forecast.mean_absolute_error(our_confident["actual"], our_confident["typical"])

    # Naive baselines
    last_val = forecast.naive_last_value_baseline(group)
    last_val_mae = forecast.mean_absolute_error(last_val["actual"], last_val["predicted"])

    flat_mean = forecast.naive_mean_baseline(group)
    flat_mean_mae = forecast.mean_absolute_error(flat_mean["actual"], flat_mean["predicted"])

    print(f"{persona}")
    print(f"  Our method (quantile median):  MAE = {our_mae:.0f}")
    print(f"  Naive last-value baseline:     MAE = {last_val_mae:.0f}")
    print(f"  Naive flat-mean baseline:      MAE = {flat_mean_mae:.0f}")

    best_naive = min(last_val_mae, flat_mean_mae)
    improvement = (best_naive - our_mae) / best_naive * 100
    print(f"  → {'Improvement' if improvement > 0 else 'WORSE'} vs best naive: {improvement:+.1f}%\n")