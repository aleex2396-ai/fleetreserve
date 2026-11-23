"""A plan: every car's planned power in hours 0 to `n` as a causal affine function of the requests.

Car `i` plans `constant[i, t] + sum_s gains[i, t, s - 1] zeta_s` in hour `t`, with `zeta_s` the
request variable of service hour `s` (1 to `n`). Causal: the power of hour `t` may depend only on
`zeta_1 .. zeta_t`, which are known at the start of hour `t` (plan section 1: the request of an
hour is known at its start); the decision hour 0 depends on none.

`two_way` says how the plan is run on true cars (`replay`): False for a plan of a one-way rung,
whose negative planned power draws nothing; True for a plan of a two-way rung, whose negative
planned power is a discharge.

The fixed plan (plan section 1) is the certified causal affine policy at scale `beta`
(`from_policy`): `u_i = g_i + beta G_i zeta`, meeting the request `m + beta w zeta`.
"""

from dataclasses import dataclass

import numpy as np

from fleetreserve.policy import AffinePolicy


@dataclass(frozen=True)
class Plan:
    constant: np.ndarray  # (cars, n + 1)
    gains: np.ndarray  # (cars, n + 1, n)
    two_way: bool

    def __post_init__(self):
        constant = np.array(self.constant, dtype=float)
        gains = np.array(self.gains, dtype=float)
        if constant.ndim != 2 or gains.shape != (*constant.shape, constant.shape[1] - 1):
            raise ValueError("constant must be (cars, n + 1) and gains (cars, n + 1, n)")
        if not (np.all(np.isfinite(constant)) and np.all(np.isfinite(gains))):
            raise ValueError("a plan must be finite")
        hours = constant.shape[1]
        later = np.arange(hours - 1)[None, :] >= np.arange(hours)[:, None]  # s - 1 >= t: s > t
        if np.any(gains[:, later] != 0):
            raise ValueError("a plan may not depend on the request of a later hour")
        for value in (constant, gains):
            value.flags.writeable = False
        object.__setattr__(self, "constant", constant)
        object.__setattr__(self, "gains", gains)
        object.__setattr__(self, "two_way", bool(self.two_way))

    @classmethod
    def from_policy(cls, policy: AffinePolicy, beta: float) -> "Plan":
        """The fixed plan: the certified policy at scale `beta`, run one-way."""
        return cls(policy.g, beta * policy.G, two_way=False)

    @property
    def n_service(self) -> int:
        return int(self.constant.shape[1] - 1)

    @property
    def n_cars(self) -> int:
        return int(self.constant.shape[0])

    def powers(self, hour: int, known) -> np.ndarray:
        """Every car's planned power in `hour`, from the request variables known at its start.

        `known` holds `zeta_1 .. zeta_k` for some `k >= hour`; entries beyond `hour` are not read.
        """
        z = np.asarray(known, dtype=float)
        if z.ndim != 1 or z.size < hour:
            raise ValueError(f"hour {hour} needs the first {hour} request variables")
        return self.constant[:, hour] + self.gains[:, hour, :hour] @ z[:hour]

    def remaining(self, hour: int, known) -> tuple[np.ndarray, np.ndarray]:
        """The plan for hours `hour` to `n` given `zeta_1 .. zeta_hour` (`known`, length `hour`).

        Returns `(offset, future)`: `offset[i, j]` is car `i`'s planned power in hour `hour + j`
        with the known request variables put in, and `future[i, j, k]` its gain on
        `zeta_{hour + 1 + k}`."""
        z = np.asarray(known, dtype=float)
        if z.shape != (hour,):
            raise ValueError(f"known must hold exactly the first {hour} request variables")
        offset = self.constant[:, hour:] + self.gains[:, hour:, :hour] @ z
        return offset, self.gains[:, hour:, hour:]
