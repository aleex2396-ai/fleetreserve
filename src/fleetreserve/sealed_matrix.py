"""The run matrices of the run from the tag and of its rehearsal: for each model run through the
full method on the sealed units, a matrix at that model's own kappa and dial per site type, read
from that model's kappa-choice results by reading R2 (plan sections 6, 7 and 13 WP14).

Command line, from the project's top folder:
    python -m fleetreserve.sealed_matrix <run folder>/kappa-choice.json [three more]
        [--out <folder>]

Inputs: one results file of `kappa_choice` per model of `FULL_METHOD_MODELS` (the main model
`lgbm_tail`, the neural model `hazard_tuned_inputs` and the two earlier forecasts `earlier_gbm`
and `earlier_population`: plan section 7 and WP14), each verified against its run's manifest
(`run_verify.require_verified`: exit 0, not stopped, the file as written) and lying directly in
its run folder. Its model is the one its run names (`run.model`). Kaplan-Meier, pooled LightGBM
and Cox run the fixed plan only (plan WP14) and get no matrix of their own (reading M3).

The reading (`READING`, R2: the owner's decision of 18 November 2026): one kappa per model, the
level the results' `readings.R2.kappa` gives, and per site type the dial `readings.R2.dials`
gives (`kappa_choice` reading K5). A model whose R2 is not defined (no kappa, or no dial for a
site type) could not run: no matrix is written for any model.

Outputs, in `--out` (default the top folder's `gate/`), for each model `<model>`:
- `sealed-matrix/<model>.json`: the matrix of the run from the tag, whose gate units are the nine
  sealed units (`sealed_route.HELD_OUT_UNITS`);
- `rehearsal-matrix/<model>.json`: the same matrix with the development units of the gate
  matrix as its gate units, for the rehearsal (plan WP12);
and for the main model also `sealed-matrix/lgbm_tail-forecast-energy.json` and its rehearsal
twin: the comparison with forecast energy at the headline's own kappa, reading and dial (plan
WP14), in a matrix of its own (reading M2); and `sealed-matrix/lgbm_tail-rho<r>.json` with its
rehearsal twin for each unprotected share r of `SENSITIVITY_RHOS` (0.1 and 0.2): the main model's
matrix with `rho` r in place of the gate matrix's 0, the sensitivity of plan WP15 ("ρ=0.1, 0.2
for the main model only"), at the main model's own kappa and dials (reading M5).
Every shared field (split seed, counts, readings of the truth tables, tolerances, rung, report)
but the sensitivity matrices' `rho` is the committed gate matrix's (`gate/matrix.json`, so the
split rule is the one the sealed truth build records), and every configuration field is that of
its dial family's configuration there, but the kappa, the dial, the reading of section 5 and, for
the comparison, the energy bounds.
The configurations of one model's matrix, at its kappa K and the set D of its site types' dials:
- the main reading of section 5 (`dial_choice.MAIN_READING`, departures seen from the
  rounded-down departure hour), family `main`: the configuration without driver-risk rows
  `main-k<K>-none-empirical` and, for each dial d of D, `main-k<K>-d<d>-empirical`;
- the beside reading (departures seen at the start of the following hour), family `beside`:
  `beside-k<K>-d<d>-empirical` for each dial of D (plan WP14: "the beside reading of Section 5");
- each with its oracle-departure partner (`...-oracle`).
The comparison's matrix holds the main reading's configurations with the energy bounds
`empirical` (family `alternate`). The main reading's configurations make the matrix's one dial
family (`dial_choice.find_family`), so `compare_run`, `corner_run`, `drift_run` and `cold_start`
take each dial of D by `--dial`.

The field `sealed_choice` of each matrix (`gate_config`) records: `reading` R2; `model`; `kappa`;
`rho`, the matrix's unprotected share; `dials`, per site type (`kappa_choice.SITE_TYPE_UNITS`'
names) its dial; `units`, per site type the matrix's gate units of that type (`site_type_of`);
`headline`, per site type the name of the main reading's configuration at its dial;
`comparison`, the comparison's words or null; and
`source`, the kappa-choice file (its run folder, name, SHA-256 and the run's commit). `check`
holds a matrix's `sealed_choice` against its configurations and gate units.

Exit status: 0 written; 1 a file was refused by its run's manifest; 2 could not run (a file
missing, not a kappa-choice results file, two files of one model, a model missing or not of the
full method, R2 not defined for a model).

Readings (alternatives in brackets):
  M1. One file per model and purpose, under `gate/sealed-matrix/` and `gate/rehearsal-matrix/`,
      not one file for every model: a run scores every configuration of its matrix, so one
      matrix holding the four models' pairs would score each model at the others' kappas on
      the sealed units, and its dial configurations at two kappas would be no dial family.
      [Alternative: one file and a run option naming the configurations of its model.]
  M2. The comparison with forecast energy is a matrix of its own, scored by a second
      `learn_run` of the main model: its dial configurations differ from the main ones in the
      energy bounds, and one dial family per matrix is what the commands taking `--dial` read.
      [Alternative: in the main model's matrix, which would make it no dial family.]
  M3. Kaplan-Meier, pooled LightGBM and Cox are scored on the main model's matrix and their
      fixed plan alone is reported (plan WP14: "fixed plan for other models and κ").
      [Alternative: a matrix at the grid's kappa levels.]
  M4. A site type's units: a workplace unit is an a.s.r. unit (`load_asr.UNITS`) or an NREL
      unit, every other unit is of Norway and Korea (plan section 2's split of the windows).
      [Alternative: `learn_features.SITE_TYPES` of the units' sites, which needs their tables.]
  M5. The sensitivity matrices at rho 0.1 and 0.2 keep the kappa and dials R2 gives at rho 0:
      plan WP15 varies rho alone, and the kappa choice is made at the main value. They hold the
      main and beside readings like the main model's matrix; no comparison with forecast energy
      is written at them. [Alternatives: a kappa choice run again at each rho; the main reading
      alone.]
"""

import argparse
import copy
import hashlib
import json
import os
import sys
from pathlib import Path

from fleetreserve import dial_choice, gate_config, kappa_choice
from fleetreserve.load_asr import UNITS as ASR_UNITS
from fleetreserve.run_record import MANIFEST_NAME
from fleetreserve.run_verify import CannotCheck, NotVerified, require_verified
from fleetreserve.sessions import TOP

MAIN_MODEL = "lgbm_tail"
FULL_METHOD_MODELS = ("lgbm_tail", "hazard_tuned_inputs", "earlier_gbm", "earlier_population")
READING = "R2"
SEALED_FOLDER, REHEARSAL_FOLDER = "sealed-matrix", "rehearsal-matrix"
COMPARISON = f"{MAIN_MODEL}-forecast-energy"
SENSITIVITY_RHOS = (0.1, 0.2)
COMPARISON_TEXT = (
    "the comparison with forecast energy at the headline's own kappa, reading and dial (plan "
    "section 13, WP14): energy bounds empirical"
)
WORKPLACES, OTHERS = tuple(kappa_choice.SITE_TYPE_UNITS)  # "workplaces", "Norway and Korea"
TEMPLATE = TOP / "gate" / "matrix.json"
BESIDE_READING = "hour-start"
CHOICE_FIELDS = (
    "reading",
    "model",
    "kappa",
    "rho",
    "dials",
    "units",
    "headline",
    "comparison",
    "source",
)


class CouldNotRun(Exception):
    """The matrices cannot be written (exit 2)."""


def site_type_of(unit: str) -> str:
    """The site type of a unit (reading M4)."""
    return WORKPLACES if unit in ASR_UNITS or unit.startswith("nrel") else OTHERS


def read_choice(path) -> dict:
    """(model, kappa, dials, source) of one kappa-choice results file; raises `NotVerified`
    (`CannotCheck` included) or `CouldNotRun`."""
    file = Path(os.path.abspath(path))
    manifest = require_verified(file)
    where = f"{file.parent.name}/{file.name}"
    if file.name != kappa_choice.RESULTS_NAME or not (file.parent / MANIFEST_NAME).is_file():
        raise CouldNotRun(f"{where} is no {kappa_choice.RESULTS_NAME} lying in its run folder")
    content = file.read_bytes()
    try:
        document = json.loads(content)
        if document["label"] != f"kappa choice: {kappa_choice.LABEL}":
            raise ValueError("its label is not that of kappa_choice")
        model = document["run"]["model"]
        found = document["readings"][READING]
        kappa = found["kappa"]
        dials = {t: (found["dials"][t] or {}).get("dial") for t in kappa_choice.SITE_TYPE_UNITS}
    except (ValueError, KeyError, TypeError, AttributeError) as problem:
        raise CouldNotRun(f"{where} is not a results file of kappa_choice ({problem})") from problem
    if kappa is None or any(d is None for d in dials.values()):
        raise CouldNotRun(
            f"{where}: reading {READING} of the model {model} is not defined (kappa {kappa}, "
            f"dials {dials}); no matrix is written"
        )
    source = {
        "run_folder": file.parent.name,
        "results": file.name,
        "sha256": hashlib.sha256(content).hexdigest(),
        "commit": manifest["commit"],
    }
    return {"model": model, "kappa": kappa, "dials": dials, "source": source}


def _number(value: float) -> str:
    return f"{value:g}"


def _template() -> tuple[dict, dict]:
    """(the gate matrix's document, its dial family's configuration at the first dial)."""
    document = json.loads(TEMPLATE.read_text())
    _, dials = dial_choice.find_family(gate_config.parse(document))
    by_name = {c["name"]: c for c in document["configurations"]}
    return document, by_name[dials[0].name]


def _pair(base: dict, name: str, family: str, kappa, dial, seen: str, energy: str) -> list[dict]:
    """A configuration and its oracle partner."""
    forecast = copy.deepcopy(base)
    forecast.update(
        name=f"{name}-empirical",
        family=family,
        kappa=kappa,
        departure="empirical",
        energy_bounds=energy,
        departure_seen=seen,
        oracle_partner=f"{name}-oracle",
        driver_risk=dial,
    )
    oracle = dict(forecast, name=f"{name}-oracle", departure="oracle", oracle_partner=None)
    return [forecast, oracle]


def configurations(kappa, dials, *, comparison: bool) -> list[dict]:
    """The configurations of one matrix (module text)."""
    _, base = _template()
    k = _number(kappa)
    values = sorted(set(dials.values()))
    main = dial_choice.MAIN_READING
    if comparison:
        found = _pair(
            base, f"forecast-energy-k{k}-none", "alternate", kappa, None, main, "empirical"
        )
        for d in values:
            name = f"forecast-energy-k{k}-d{_number(d)}"
            found += _pair(base, name, "alternate", kappa, d, main, "empirical")
        return found
    found = _pair(base, f"main-k{k}-none", "main", kappa, None, main, "known")
    for d in values:
        found += _pair(base, f"main-k{k}-d{_number(d)}", "main", kappa, d, main, "known")
    for d in values:
        name = f"beside-k{k}-d{_number(d)}"
        found += _pair(base, name, "beside", kappa, d, BESIDE_READING, "known")
    return found


def matrix(choice: dict, units, *, comparison: bool, purpose: str, rho=None) -> dict:
    """One matrix document (module text); `rho` None keeps the gate matrix's."""
    template, _ = _template()
    document = copy.deepcopy(template)
    if rho is not None:
        document["rho"] = rho
    model, kappa, dials = choice["model"], choice["kappa"], choice["dials"]
    found = configurations(kappa, dials, comparison=comparison)
    prefix = "forecast-energy" if comparison else "main"
    document["about"] = (
        f"Run matrix of the {purpose} for the model {model}, written by fleetreserve.sealed_matrix "
        f"from the kappa-choice results {choice['source']['run_folder']}/"
        f"{choice['source']['results']} by reading {READING} (one kappa per model, the dial per "
        f"site type): kappa {_number(kappa)}, dials "
        + ", ".join(f"{t} {_number(d)}" for t, d in dials.items())
        + ". "
        + (COMPARISON_TEXT if comparison else "Main reading of section 5 and the beside reading.")
        + (
            f" The sensitivity at rho {_number(rho)} (plan section 13, WP15)."
            if rho is not None
            else ""
        )
        + (" Every other" if rho is not None else " Every")
        + " shared value is the gate matrix's."
    )
    document["gate_units"] = list(units)
    document["required_units"] = min(document["required_units"], len(units))
    document["configurations"] = found
    by_type = {t: [u for u in units if site_type_of(u) == t] for t in dials}
    document["sealed_choice"] = {
        "reading": READING,
        "model": model,
        "kappa": kappa,
        "rho": document["rho"],
        "dials": dict(dials),
        "units": by_type,
        "headline": {
            t: f"{prefix}-k{_number(kappa)}-d{_number(d)}-empirical" for t, d in dials.items()
        },
        "comparison": COMPARISON_TEXT if comparison else None,
        "source": dict(choice["source"]),
    }
    return document


def check(document: dict) -> list[str]:
    """Why a matrix's `sealed_choice` does not hold against its configurations and gate units
    (module text); empty when it holds."""
    gate_config.parse(document)
    choice = document.get("sealed_choice")
    if not isinstance(choice, dict) or sorted(choice) != sorted(CHOICE_FIELDS):
        return [f"sealed_choice has not the fields {', '.join(CHOICE_FIELDS)}"]
    problems = []
    names = {c["name"]: c for c in document["configurations"]}
    for site_type, dial in choice["dials"].items():
        found = names.get(choice["headline"].get(site_type))
        if found is None or found["kappa"] != choice["kappa"] or found["driver_risk"] != dial:
            problems.append(f"the headline of {site_type} is not its configuration at its dial")
    units = [u for group in choice["units"].values() for u in group]
    if sorted(units) != sorted(document["gate_units"]):
        problems.append("sealed_choice.units are not the gate units")
    for site_type, group in choice["units"].items():
        if any(site_type_of(u) != site_type for u in group):
            problems.append(f"sealed_choice.units.{site_type} holds a unit of another type")
    if choice["rho"] != document["rho"]:
        problems.append("sealed_choice.rho is not the matrix's rho")
    if any(c["kappa"] != choice["kappa"] for c in names.values()):
        problems.append("a configuration is not at the matrix's kappa")
    return problems


def write_all(choices: list[dict], out: Path) -> list[Path]:
    """Every matrix of the module text, in `out`; the files written."""
    from fleetreserve.sealed_route import HELD_OUT_UNITS

    template, _ = _template()
    development = list(template["gate_units"])
    written = []
    for choice in choices:
        kinds = [(choice["model"], False, None)]
        if choice["model"] == MAIN_MODEL:
            kinds.append((COMPARISON, True, None))
            kinds += [(f"{MAIN_MODEL}-rho{_number(r)}", False, r) for r in SENSITIVITY_RHOS]
        for name, comparison, rho in kinds:
            for folder, units, purpose in (
                (SEALED_FOLDER, list(HELD_OUT_UNITS), "run from the tag"),
                (REHEARSAL_FOLDER, development, "rehearsal on development units"),
            ):
                document = matrix(choice, units, comparison=comparison, purpose=purpose, rho=rho)
                problems = check(document)
                if problems:
                    raise CouldNotRun(f"{name}: " + "; ".join(problems))
                path = Path(out) / folder / f"{name}.json"
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps(document, indent=1) + "\n")
                written.append(path)
    return written


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("results", nargs="+", type=Path)
    parser.add_argument("--out", type=Path, default=TOP / "gate")
    args = parser.parse_args(argv)
    try:
        choices = [read_choice(path) for path in args.results]
        models = [c["model"] for c in choices]
        if sorted(models) != sorted(FULL_METHOD_MODELS):
            raise CouldNotRun(
                f"the results are of the models {models}; one file of each of "
                f"{', '.join(FULL_METHOD_MODELS)} is needed"
            )
        written = write_all(choices, args.out)
    except NotVerified as problem:  # CannotCheck included
        code = 2 if isinstance(problem, CannotCheck) else 1
        print(f"{'COULD NOT RUN' if code == 2 else 'REFUSED'}: {problem}")
        return code
    except (CouldNotRun, gate_config.MatrixError) as problem:
        print(f"COULD NOT RUN: {problem}")
        return 2
    for path in written:
        print(f"written: {path} sha256 {hashlib.sha256(path.read_bytes()).hexdigest()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
