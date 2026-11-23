"""The removal limits of the three held-out loaders, in one module that a parsing fix may not touch.

A parsing fix after the tag may change how a held-out file is read, never a removal rule or its
limit (plan section 13). The opener accepts a fix that changes a set's loader and its tests and
the plan's section 14 only (the held-out loader's reading H6(b)); this module is none of those, so a
commit that changes a limit here is refused. The loaders import every limit from here and hold
none of their own.

- `LIMIT_KW`: the physical limit of the average power over a session, 22 kW (the European
  three-phase limit, `sessions.PHYSICAL_LIMIT_KW["norway"]`); a session whose energy over its
  plug-in to plug-out time is above it is removed (`load_asr`, `load_korea`; the Norwegian
  loader's rule is `load_norway.to_sessions`', with the same value).
- `ENERGY_ABOVE_KWH`: a session whose energy is not above this value is removed.
- `SLOW`, `FAST`: the `ChargerType` codes of a slow and a fast charger (`load_korea`); only
  slow-charger sessions are used, and every row with `FAST` is removed first.
- `FIRST_TIME`, `LAST_TIME`: the range, in Asia/Seoul wall clock, outside which a Korean time
  raises `LayoutError`.
"""

from fleetreserve.sessions import PHYSICAL_LIMIT_KW

LIMIT_KW = PHYSICAL_LIMIT_KW["norway"]
ENERGY_ABOVE_KWH = 0
SLOW, FAST = 0.0, 1.0
FIRST_TIME = "2021-09-01 00:00:00"
LAST_TIME = "2022-10-31 23:59:59"
