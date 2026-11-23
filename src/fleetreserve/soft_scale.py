"""The exact scale of a fixed box shape as a differentiable function of the cars' bounds (plan
section 7, T8; built on section 4, T1(b)).

Inputs, as PyTorch tensors of one dtype with the same leading (batch) dimensions:

- the cars' bounds (`Bounds`), each of shape `(..., cars)`: `departure` (hours after the decision
  time, a real number), `rate` (kW), `cap` (the most energy the car may store, kWh) and `need`
  (the least energy it must have at its departure, kWh). Every car is plugged in from the decision
  time, as the fleet at τ is (plan section 2). A batch of fleets of different sizes is padded with
  cars of rate, cap and need 0, which add 0 to every set function;
- the shape, each of shape `(..., n)` for the `n` service hours 1 to `n`: `centre`, `w_up`,
  `w_down`, as in `exact.box_scale`.

The set of one car (plan section 2's forecast set, `carsets.forecast_rows`; the true one-way set
when the departure is a whole hour). Hours count from 0, the decision hour. The share of hour `t`
before the departure `d` is `f_t(d) = min(max(d - t, 0), 1)`: a car whose departure falls inside
an hour is present for that share of it and draws at most `rate * f_t` in it (a reading: the
plan's "fractional last hour" read as `day.ForecastCar.power_cap`). Its energy in hours 0 to `n`
is at most `cap`, and that energy plus `rate` times the shares of the recovery hours is at least
`need`; hour 0 is free (as in `exact` and `carsets`). The shares of all hours sum to `d`, so with
`s(A)` the sum of the shares of the service hours in `A`, the set functions of T1 read

    b(A) = min(rate * s(A), cap),     p(A) = max(need - rate * (d - s(A)), 0).

With whole-hour departures `s(A)` is the number of the car's hours in `A` and `d - s(A)` the number
of its hours outside `A`, which are T1's own forms. With a fractional last hour they are the set
functions of the box `0 <= u_t <= rate f_t` (hours 0 to `n`) cut by bounds on its total, when that
set is not empty; such a set is a g-polymatroid for any real limits per hour (a box is one, and
cutting a g-polymatroid by bounds on its total keeps one: cited from memory of Frank 2011, not
checked against the book), and the fleet's functions are the sums over cars. This is checked on
made-up fleets, not proved here: `tests/learn/test_soft_scale_exact.py` compares each car's
functions with programmes on `carsets.forecast_rows` and the fleet's scale with vertex
programmes. The formulas need the departure inside the day's hours (`day.Day` refuses a later
one), because the code base's recovery supply counts the shares of the day's recovery hours only.

`true_bounds(day)` gives the true cars' bounds (`departure` the whole hour, `cap` the room, `need`
`(1 - rho)` times the room); `forecast_bounds(day)` the forecasts' (`need` as the forecast states
it, no `rho` applied).

The terms. For every non-empty subset `A` of the service hours (`subset_bits`, the same bit order as
`exact`), an upper term `(b(A) - c(A)) / w_up(A)` when `w_up(A) > 0` and a lower term
`(c(A) - p(A)) / w_down(A)` when `w_down(A) > 0`; a term with a zero width is left out (`+inf` in
`terms`, as `exact.box_scale` omits it). There are at most `2 (2^n - 1) < 2^(n+1)` finite terms;
`term_count` gives their number `N`. No linear programme is solved anywhere here.

- `exact_scale`: the smallest term, the largest scale `lam` with the box
  `c - lam w_down .. c + lam w_up` inside the set when the centre is inside. It is not clamped at 0:
  a negative value says the centre is outside the set on a face that has a width. `exact.box_scale`
  raises instead; the two agree as `max(exact_scale, 0) == box_scale` when the centre is inside.
  With no finite term (every width zero) it is `+inf`, as in `exact.box_scale`.
- `centre_inside`: True when `p(A) <= c(A) <= b(A)` within `tolerance` for every subset `A`,
  widths or not, as `exact.box_scale` checks before it computes (a centre outside on a face with
  zero width is not seen by the terms).
- `soft_scale`: `-sigma log sum_k exp(-term_k / sigma)` over the finite terms, computed with
  `torch.logsumexp`. For `sigma > 0`, `soft <= exact <= soft + sigma ln N` (the sum holds the
  smallest term once and `N` terms in all). With no finite term it is `+inf` with a zero gradient.

Gradients. The terms, `exact_scale` and `soft_scale` are differentiable in all four fields of the
bounds and in the shape except at these points: where a share is at 0 or 1 (a departure on a
whole hour from 1 to `n + 1`); where `rate * s(A)` equals `cap`; where `need - rate (d - s(A))`
equals 0; where a width `w_up(A)` or `w_down(A)` is 0; and, for `exact_scale`, where two terms
are smallest together. At a zero width the function is not differentiable in that width: widths
are not negative and the term is left out there, so at most a derivative from above exists, and
the code gives 0 (a left-out term passes no gradient); on a face with `b(A) = c(A)` (or
`c(A) = p(A)`) the term is 0 at every positive width, so the function jumps at the zero width.

The `min` and `max` are `torch.minimum` and `torch.maximum`, which return one of their arguments
exactly (a bound that does not bind adds no rounding) and, where the two arguments are equal,
give half the gradient to each. So where exactly one of these kinks is met on a car's path to a
term (one share at 0 or 1, or one `min` with the cap, or one `max` with 0, the others away from
their kinks), the gradient is the mean of the two one-sided derivatives (tested in
`tests/learn/test_soft_scale_gradient.py` for the shares, the `min` and the `max`). Where two of
them coincide, each halves what passes through it and the gradient is the product of the halves,
which in general is neither the mean of the one-sided derivatives nor, where the function is
differentiable at the point, its derivative: `b({1})` of one car at departure 2 with rate 2 and
cap 2 has one-sided derivatives 2 and 0 in the departure (mean 1) and gradient 0.5; an empty car
(cap 0) whose departure is the start of the earliest service hour of `A` adds 0 to `b(A)` on
both sides (derivative 0) and gradient `rate / 4` (both tested there; at the start of a later
hour of `A` the car's `s(A)` is already positive, the cap alone binds and the gradient is 0).
Such points have measure zero, but
whole-hour departures meet the shares' kinks. At a tie of smallest terms PyTorch's `amin` shares
the gradient evenly among them (tested there for two terms). The check of T8
(`soft_scale_check`) makes no comparison at a tie of smallest terms, lets no measured step whose
differences show a kink decide a comparison, and records a comparison that shows a kink at both
of its steps as a kink, neither passed nor failed.

How a term depends on the bounds, with the shape fixed. A term is linear in `b(A)` or `p(A)`, so it
is enough to read the two set functions. `s(A)` and `d - s(A)` are piecewise linear in the departure
`d`. A car's `rate * s(A)` enters `b(A)` only when it is below `cap`, and moves with `d` only when
the departure falls inside a service hour that is in `A` (`s(A)` is constant in `d` otherwise);
`rate * (d - s(A))` enters `p(A)` only when `p(A) > 0`, and moves with `d` only when the departure
falls inside an hour outside `A` (hour 0, a recovery hour, or a service hour not in `A`). So each
term is piecewise linear in the caps, needs and rates with the departures fixed, and in the
departures, caps and needs with the rates fixed; jointly in a car's rate and departure it is
piecewise bilinear, not piecewise linear (a sum of products of a rate with a piecewise linear
function of the same car's departure, plus parts linear in the caps and needs), hence locally
Lipschitz and differentiable almost everywhere. `tests/learn/test_soft_scale_soft.py` has one case
where the product enters `b` and the term is quadratic along a line that moves a car's rate and
departure together, and one where `p > 0` and the departure is inside a service hour of `A`, so
`p` stays linear along it. The sandwich does not depend on linearity.

Untested: values and gradients on any device but the CPU in float64; fleets above 4 cars and 4
service hours against programmes; service hours above 4 against `exact.box_scale`.
"""

from dataclasses import dataclass

import lightgbm  # noqa: F401  (before torch: see hazard_net)
import torch

from fleetreserve.day import MAX_SERVICE_HOURS, Day

TOLERANCE = 1e-9  # kWh: slack of `centre_inside`, as `exact.TOLERANCE`


@dataclass(frozen=True)
class Bounds:
    """The cars' bounds, each a tensor of shape `(..., cars)` (module text)."""

    departure: torch.Tensor
    rate: torch.Tensor
    cap: torch.Tensor
    need: torch.Tensor


def _check_n(n_service: int) -> int:
    if not 1 <= n_service <= MAX_SERVICE_HOURS:
        raise ValueError(f"n_service must be 1 to {MAX_SERVICE_HOURS}, got {n_service}")
    return n_service


def subset_bits(n_service: int, dtype=torch.float64, device=None) -> torch.Tensor:
    """`bits[mask, k]` is 1 when service hour `k + 1` is in the subset `mask` (`exact`'s order)."""
    masks = torch.arange(2 ** _check_n(n_service), device=device)
    hours = torch.arange(n_service, device=device)
    return ((masks[:, None] >> hours) & 1).to(dtype)


def _relu(x: torch.Tensor) -> torch.Tensor:
    """`max(x, 0)` by `torch.maximum`: exact, derivative 1/2 at 0 (module text)."""
    return torch.maximum(x, torch.zeros_like(x))


def _smaller(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """`min(a, b)` by `torch.minimum`: exact, the derivative split evenly where a equals b."""
    return torch.minimum(a, b)


def shares(departure: torch.Tensor, n_service: int) -> torch.Tensor:
    """`f_t(d) = min(max(d - t, 0), 1)` for the service hours `t = 1..n`, shape `(..., cars, n)`;
    its derivative in `d` is 1/2 at the two kinks `d = t` and `d = t + 1` (module text)."""
    hours = torch.arange(1, _check_n(n_service) + 1, dtype=departure.dtype, device=departure.device)
    return _smaller(_relu(departure[..., None] - hours), torch.ones((), dtype=departure.dtype))


def set_functions(bounds: Bounds, n_service: int) -> tuple[torch.Tensor, torch.Tensor]:
    """The fleet's `p(A)` and `b(A)` for every subset, each of shape `(..., 2^n)`."""
    bits = subset_bits(n_service, bounds.rate.dtype, bounds.rate.device)
    s = shares(bounds.departure, n_service) @ bits.T  # (..., cars, 2^n)
    rate = bounds.rate[..., None]
    upper = _smaller(rate * s, bounds.cap[..., None]).sum(dim=-2)
    lower = _relu(bounds.need[..., None] - rate * (bounds.departure[..., None] - s)).sum(dim=-2)
    return lower, upper


def _sums(centre, w_up, w_down):
    n = centre.shape[-1]
    bits = subset_bits(n, centre.dtype, centre.device)
    return centre @ bits.T, w_up @ bits.T, w_down @ bits.T


def terms(bounds: Bounds, centre, w_up, w_down) -> torch.Tensor:
    """The upper terms of every subset then the lower terms, shape `(..., 2^(n+1))`; a term with a
    zero width is `+inf` (module text)."""
    n = centre.shape[-1]
    lower, upper = set_functions(bounds, n)
    c, up, down = _sums(centre, w_up, w_down)
    infinite = torch.full_like(c, float("inf"))
    has_up, has_down = up > 0, down > 0
    above = torch.where(has_up, (upper - c) / torch.where(has_up, up, 1.0), infinite)
    below = torch.where(has_down, (c - lower) / torch.where(has_down, down, 1.0), infinite)
    return torch.cat([above, below], dim=-1)


def term_count(w_up, w_down) -> torch.Tensor:
    """`N`: the number of finite terms (non-empty subsets with a positive width up, plus those
    with a positive width down)."""
    bits = subset_bits(w_up.shape[-1], w_up.dtype, w_up.device)
    return ((w_up @ bits.T) > 0).sum(dim=-1) + ((w_down @ bits.T) > 0).sum(dim=-1)


def exact_scale(bounds: Bounds, centre, w_up, w_down) -> torch.Tensor:
    """T1(b): the smallest term, signed, `+inf` when no term is finite (module text)."""
    return terms(bounds, centre, w_up, w_down).amin(dim=-1)


def soft_scale(bounds: Bounds, centre, w_up, w_down, sigma: float) -> torch.Tensor:
    """`-sigma log sum exp(-term / sigma)` over the finite terms (module text)."""
    if not sigma > 0:
        raise ValueError(f"sigma must be positive, got {sigma}")
    # A term left out is +inf and adds exp(-inf) = 0; with none finite the sum is 0, the value
    # +inf, and the gradient 0, because every `+inf` comes from the unselected side of a `where`.
    return -sigma * torch.logsumexp(-terms(bounds, centre, w_up, w_down) / sigma, dim=-1)


def centre_inside(bounds: Bounds, centre, tolerance: float = TOLERANCE) -> torch.Tensor:
    """True when `p(A) - tolerance <= c(A) <= b(A) + tolerance` for every subset `A`."""
    lower, upper = set_functions(bounds, centre.shape[-1])
    c = centre @ subset_bits(centre.shape[-1], centre.dtype, centre.device).T
    return ((c <= upper + tolerance) & (c >= lower - tolerance)).all(dim=-1)


def _tensor(values) -> torch.Tensor:
    return torch.tensor(list(values), dtype=torch.float64)


def true_bounds(day: Day) -> Bounds:
    """The true one-way cars of `day`: whole-hour departure, cap the room, need `(1 - rho)` room."""
    return Bounds(
        _tensor(float(c.departure) for c in day.cars),
        _tensor(c.rate for c in day.cars),
        _tensor(c.room for c in day.cars),
        _tensor(c.need(day.rho) for c in day.cars),
    )


def forecast_bounds(day: Day) -> Bounds:
    """The forecast cars of `day` (`day.forecasts` must be given), as they are stated."""
    if day.forecasts is None:
        raise ValueError("the day has no forecasts")
    return Bounds(
        _tensor(f.departure for f in day.forecasts),
        _tensor(f.rate for f in day.forecasts),
        _tensor(f.cap for f in day.forecasts),
        _tensor(f.need for f in day.forecasts),
    )
