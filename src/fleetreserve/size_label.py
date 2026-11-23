"""The exact safe scale of one offering day's certified shape (plan sections 4 (T1(b)), 7).

Plan section 7's day-size quantile net is trained on "T1(b)'s exact scale of the certified shape on
true training days". For a day prepared by `gate_day.prepare_day` that offers, `day_label` gives:

- `scale`: `exact.box_scale` of the certificate's box (centre `m`, widths `w` up and down) on the
  set functions of the true cars of the steered sub-day (`exact.day_functions`), so the largest
  `lam` with `m +- lam w` inside the true fleet set. When the centre is outside that set
  (`exact.box_inside(f, m, m)` is false) the scale is 0. It is not clipped and may exceed 1.
- `clipped`: `scale` clipped to [0, 1].
- `centre_inside`: whether the centre is inside the true fleet set.
- `fixed_plan_scale`: the smallest of the day's hour thresholds (`fixed_plan_loss.hour_thresholds`
  of the certificate's policy on the true rows), the scale above which the fixed plan's slot loss
  is above 0 in closed form (`gate_calibrate.slot_steps`). It may be negative or infinite.

A day that does not offer has no label: `day_label` raises `ValueError` for it.
"""

from dataclasses import dataclass

import numpy as np

from fleetreserve import exact
from fleetreserve.carsets import true_rows
from fleetreserve.fixed_plan_loss import hour_thresholds
from fleetreserve.gate_day import PreparedDay


@dataclass(frozen=True)
class Label:
    scale: float
    clipped: float
    centre_inside: bool
    fixed_plan_scale: float


def day_label(prepared: PreparedDay) -> Label:
    """The label of an offering day (module text)."""
    if prepared.abstains:
        raise ValueError(f"the day {prepared.date} does not offer, so it has no label")
    policy = prepared.certificate.policy
    functions = exact.day_functions(prepared.day)
    inside = exact.box_inside(functions, policy.m, policy.m)
    scale = exact.box_scale(functions, policy.m, policy.w, policy.w) if inside else 0.0
    thresholds = hour_thresholds(policy, true_rows(prepared.day))
    return Label(
        float(scale),
        float(np.clip(scale, 0.0, 1.0)),
        bool(inside),
        float(np.min(thresholds)),
    )
