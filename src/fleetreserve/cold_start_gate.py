"""The names by which the cold start (`cold_start`) calls the shared route of a scoring command
that may score held-out units (`sealed_route`, whose module text states each check).

`cold_start` calls every check of the route through this module (`check_units`, `check_folders`,
`require_gate`, `held_out_tables`, `read_two_builds`) and catches its refusals by these names
(`UnitsRefused`, `FoldersRefused`, `SealRefused`); `DEVELOPMENT` and `HELD_OUT` are the two kinds
of a run's units. Each name is the shared module's own object: the checks are written once, in
`sealed_route`, and this module adds nothing to them.
"""

from fleetreserve.sealed_route import (
    DEVELOPMENT,
    HELD_OUT,
    HELD_OUT_UNITS,
    FoldersRefused,
    SealRefused,
    UnitsRefused,
    check_folders,
    check_units,
    held_out_tables,
    read_two_builds,
    require_gate,
)

__all__ = [
    "DEVELOPMENT",
    "HELD_OUT",
    "HELD_OUT_UNITS",
    "FoldersRefused",
    "SealRefused",
    "UnitsRefused",
    "check_folders",
    "check_units",
    "read_two_builds",
    "held_out_tables",
    "require_gate",
]
