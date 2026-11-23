"""The label of a rotation what-if: a run that scores a training unit (plan sections 3, 6).

The plan fits the learned models on the eight training units and scores development and held-out
units. A rotation what-if scores a training unit instead, with a model that was not fitted on it
(`models.lgbm_tail_rotation`): the unit's days are held out of the fit for this run only. It is a
what-if, not a run of the plan, and a matrix asks for it with the field `rotation_whatif: true`
(`gate_config`; absent means a training unit among the gate units is refused).

This module says so in every output of such a run, and only in those: `mark` adds to a results
document the key `rotation_whatif` (what the run is, the scored training units, the training units
left out of the fit, the units fitted on and the evidence sentence of `evidence_sentence`),
`summary_head` gives the same as the first lines of a text summary, and `label` is the one-line
form. All three give nothing for a matrix without the field, so a run of the plan carries no trace.

The evidence sentence states what was read in the plan and the code: the scored training unit was
training data before this run, and where its days chose a setting shared by all units (the
workplace window of plan section 2, chosen by `window_choice` on the counting days of Caltech and
JPL) the sentence names it. No verdict word is used.
"""

from collections.abc import Sequence

from fleetreserve.gate_config import Matrix
from fleetreserve.truth_build import TRAIN_UNITS
from fleetreserve.window_choice import SITES as WINDOW_SITES

KEY = "rotation_whatif"
WHAT = "ROTATION WHAT-IF ON A TRAINING UNIT, NOT A RUN OF THE PLAN"


def scored_units(matrix: Matrix) -> tuple[str, ...]:
    """The training units among the matrix's gate units, in the order of `TRAIN_UNITS`."""
    return tuple(u for u in TRAIN_UNITS if u in matrix.gate_units)


def evidence_sentence(scored: Sequence[str], fit_units: Sequence[str]) -> str:
    """The sentence that says what the scored training units were used for before this run
    (module text); `fit_units` is the model's reading."""
    names = ", ".join(scored)
    fitted = ", ".join(fit_units) or "no unit"
    window = [u for u in scored if u in WINDOW_SITES]
    chosen = (
        f"; the days of {', '.join(window)} helped choose the decision hour and service window "
        f"of plan section 2 (window_choice scored its candidates on the counting days of "
        f"{' and '.join(WINDOW_SITES)}), which all workplace units use"
        if window
        else ""
    )
    return (
        f"The scored training unit(s) {names} were training data before this run{chosen}; the "
        f"model of this run is not fitted on {names} (it is fitted on {fitted}), so the figures "
        "are development evidence, not held-out evidence."
    )


def record(matrix: Matrix, fit_units: Sequence[str]) -> dict | None:
    """What the label holds, or None for a matrix that is not a rotation what-if."""
    if not matrix.rotation_whatif:
        return None
    scored = scored_units(matrix)
    return {
        "what": WHAT,
        "scored_training_units": list(scored),
        "left_out_of_the_fit": [u for u in TRAIN_UNITS if u not in fit_units],
        "fit_units": list(fit_units),
        "evidence": evidence_sentence(scored, fit_units),
    }


def label(matrix: Matrix, fit_units: Sequence[str]) -> str | None:
    """The one-line form of the record, or None."""
    found = record(matrix, fit_units)
    if found is None:
        return None
    return (
        f"{found['what']}: scored {', '.join(found['scored_training_units'])}; left out of the "
        f"fit: {', '.join(found['left_out_of_the_fit'])}"
    )


def summary_head(document: dict) -> list[str]:
    """The first lines of a text summary of a results document: the label and the evidence
    sentence, or nothing for a document without the key `rotation_whatif`."""
    found = document.get(KEY)
    return [] if found is None else [found["what"], found["evidence"]]


def mark(document: dict, matrix: Matrix, fit_units: Sequence[str]) -> None:
    """Add the record to a results document (key `rotation_whatif`); a document with a `label`
    gets the one-line form in front of it and one with `definitions` the evidence sentence."""
    found = record(matrix, fit_units)
    if found is None:
        return
    document[KEY] = found
    if "label" in document:
        document["label"] = f"{label(matrix, fit_units)}. {document['label']}"
    if isinstance(document.get("definitions"), dict):
        document["definitions"][KEY] = found["evidence"]
