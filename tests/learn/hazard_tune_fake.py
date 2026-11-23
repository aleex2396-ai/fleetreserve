"""Made-up inputs for the tests of the tuning of the neural hazard model (`hazard_tune`).

`DAYS`: three made-up training units, each one site of its own name with a site type in the
committed mappings (`caltech`, `jpl` workplaces, `OSL_S` the other type), 30, 14 and 22 days of
twelve cars (`quantile_fake`). `frames(folder)`: their cars tables. `tiny_grid(folder)`: a grid
file of two fitted candidates (divergence weight 1 and 0.1 at width 32 and learning rate 0.003),
two epoch values (3 and 30, so the present settings are a candidate) and two seeds: 4 candidates.
"""

import json

from quantile_fake import cars, unit_cars

DAYS = {"caltech": 30, "jpl": 14, "OSL_S": 22}
TINY = {
    "about": "made up",
    "seeds": [0, 1],
    "epochs": [3, 30],
    "width": [32],
    "learning_rate": [0.003],
    "kl_weight": [1.0, 0.1],
}


def frames(folder) -> dict:
    return {
        unit: unit_cars(folder, unit, cars(n, seed, site=unit))
        for seed, (unit, n) in enumerate(DAYS.items())
    }


def tiny_grid(folder, name: str = "grid.json"):
    path = folder / name
    path.write_text(json.dumps(TINY))
    return path
