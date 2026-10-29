"""Create a deterministic synthetic dataset to exercise the classification workflow."""
import csv
import random
from pathlib import Path

rng = random.Random(42)
output = Path(__file__).with_name("sample_classification.csv")
with output.open("w", newline="", encoding="utf-8") as handle:
    writer = csv.writer(handle)
    writer.writerow(["age", "monthly_spend", "visits_last_month", "region", "plan", "purchased"])
    for _ in range(600):
        age = rng.randint(18, 75)
        spend = round(rng.uniform(10, 250), 2)
        visits = rng.randint(0, 20)
        region = rng.choice(["north", "south", "east", "west"])
        plan = rng.choice(["basic", "plus", "premium"])
        score = -4.1 + 0.045 * visits + 0.012 * spend + (0.8 if plan == "premium" else 0.25 if plan == "plus" else 0) + (0.3 if region == "north" else 0)
        probability = 1 / (1 + 2.718281828 ** (-score))
        label = int(rng.random() < probability)
        if rng.random() < 0.04:
            spend = ""
        if rng.random() < 0.03:
            region = ""
        writer.writerow([age, spend, visits, region, plan, label])
print(f"Wrote {output}")
