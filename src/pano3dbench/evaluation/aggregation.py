from __future__ import annotations

import random
import statistics


def bootstrap_ci(
    values: list[float], seed: int, samples: int = 2000
) -> tuple[float, float]:
    count = len(values)
    rng = random.Random(seed)
    means: list[float] = []
    for _ in range(samples):
        total = 0.0
        for _ in range(count):
            total += values[rng.randrange(count)]
        means.append(total / count)
    means.sort()
    low = means[int(0.025 * samples)]
    high = means[min(samples - 1, int(0.975 * samples))]
    return low, high


def aggregate(values: list[float], seed: int = 0) -> dict[str, float | int]:
    count = len(values)
    if count == 0:
        return {"count": 0}
    mean = statistics.fmean(values)
    median = statistics.median(values)
    if count == 1:
        return {
            "count": 1,
            "mean": mean,
            "median": median,
            "ci95_low": mean,
            "ci95_high": mean,
        }
    low, high = bootstrap_ci(values, seed)
    return {
        "count": count,
        "mean": mean,
        "median": median,
        "ci95_low": low,
        "ci95_high": high,
    }
