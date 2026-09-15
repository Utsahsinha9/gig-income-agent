import numpy as np
import pandas as pd
from datetime import datetime, timedelta

np.random.seed(42)  # reproducibility — same personas every run while we're testing

def generate_persona(
    name: str,
    start_date: str,
    weeks: int,
    base_income: float,
    volatility: float,      # std dev as a fraction of base_income
    payout_frequency: float,  # avg payouts per week (irregular timing)
    trend: float = 0.0,       # slight income growth/decline per week
):
    events = []
    current_date = datetime.strptime(start_date, "%Y-%m-%d")

    for week in range(weeks):
        n_payouts = np.random.poisson(payout_frequency)
        week_income_target = base_income * (1 + trend * week / weeks)

        for _ in range(n_payouts):
            amount = max(
                0,
                np.random.normal(
                    week_income_target / max(payout_frequency, 1),
                    volatility * base_income
                )
            )
            day_offset = np.random.randint(0, 7)
            events.append({
                "persona": name,
                "date": current_date + timedelta(days=day_offset),
                "amount": round(amount, 2)
            })

        current_date += timedelta(weeks=1)

    return pd.DataFrame(events)


steady_delivery = generate_persona(
    "steady_delivery_partner", "2025-01-01", weeks=26,
    base_income=8000, volatility=0.15, payout_frequency=3, trend=0.0
)

spiky_freelancer = generate_persona(
    "spiky_freelancer", "2025-01-01", weeks=26,
    base_income=12000, volatility=0.6, payout_frequency=1.2, trend=0.05
)

new_gig_worker = generate_persona(
    "new_gig_worker", "2025-06-01", weeks=6,
    base_income=6000, volatility=0.4, payout_frequency=1.5, trend=0.0
)

all_data = pd.concat([steady_delivery, spiky_freelancer, new_gig_worker])
all_data = all_data.sort_values(["persona", "date"]).reset_index(drop=True)
all_data.to_csv("synthetic_gig_income.csv", index=False)
print(all_data.groupby("persona")["amount"].describe())
