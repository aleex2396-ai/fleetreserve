"""The label of a trial made after the held-out results were seen (plan sections 6, 13 and 14).

A trial on development units that is made after the sealed run has ended fills no pre-registered
slot and changes no held-out number. A command run with `--made-after-sealed-run` (`FLAG`, taken by
`learn_run` and `dial_choice`; `pooled_calibration` always carries the label) says so in every
output it writes:

- the key `made_after_sealed_run` (`KEY`) in each results document, holding what the run is and
  the sentence `SENTENCE` (`mark`; a document with a `label` gets the one-line form in front of
  it);
- the first lines of the text summary (`summary_head`);
- one text file in the run folder (`FILE_NAME`, written by `write_note`), so that an output that
  cannot hold a key (a table of rows, a table of forecasts) is covered by a file of the same run
  folder, which the run's manifest lists.

All three give nothing when the flag is not given, so a run without the flag carries no trace and
every existing output is unchanged. `learn_run`, the one of the three commands with a route to
held-out units, could not run when given the flag on that route (`refuse_held_out`): a trial after
the sealed run is made on development units only. `dial_choice` has no such route, and
`pooled_calibration` refuses a unit that is not a development unit by name.

Reading L1. The label is a flag and not a field of the run matrix: the matrix format
(`gate_config`) is unchanged, and `dial_choice`, which takes no matrix of its own, has the same
option as `learn_run`. [Alternative: an optional matrix field, which needs a change to the matrix
reader and does not reach `dial_choice`.]
"""

from datetime import date
from pathlib import Path

KEY = "made_after_sealed_run"
FLAG = "--made-after-sealed-run"
FILE_NAME = "made-after-sealed-run.txt"
WHAT = "TRIAL MADE AFTER HELD-OUT RESULTS WERE SEEN, NOT A RUN OF THE PLAN"
DAY = "24 November 2026"
MONTHS = (
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December",
)


def day_text(day: date) -> str:
    """A date in the words of the sentence ("24 November 2026"), whatever the locale."""
    return f"{day.day} {MONTHS[day.month - 1]} {day.year}"


def sentence(day: str = DAY) -> str:
    """The label sentence for the date `day` (text); `SENTENCE` is the one for `DAY`."""
    return (
        f"Made on {day}, after held-out results were seen, on development units only; "
        "fills no pre-registered slot and changes no held-out number."
    )


SENTENCE = sentence()


def record(day: str | None = None) -> dict:
    """What the label holds: what the run is and the sentence. Without `day` the sentence is
    `SENTENCE`; a command that names the date its run started gives it as `day` (text)."""
    return {"what": WHAT, "note": SENTENCE if day is None else sentence(day)}


def mark(document: dict, enabled: bool, day: str | None = None) -> None:
    """Add the record to a results document (key `made_after_sealed_run`); a document with a
    `label` gets the one-line form in front of it. Nothing happens when `enabled` is false."""
    if not enabled:
        return
    document[KEY] = record(day)
    if "label" in document:
        document["label"] = f"{WHAT}. {document['label']}"


def summary_head(document: dict) -> list[str]:
    """The first lines of a text summary: what the run is and the sentence, or nothing for a
    document without the key."""
    found = document.get(KEY)
    return [] if found is None else [found["what"], found["note"]]


def write_note(folder: Path, enabled: bool, day: str | None = None) -> None:
    """Write `FILE_NAME` into the run folder `folder`; nothing when `enabled` is false."""
    if enabled:
        (Path(folder) / FILE_NAME).write_text(f"{WHAT}\n{record(day)['note']}\n")


def refuse_held_out(enabled: bool, held_out: bool) -> str | None:
    """The reason a run with the flag could not run, or None: the flag is for development units
    only, so it is refused on the route of held-out units."""
    if enabled and held_out:
        return f"{FLAG} is for development units only; the matrix's gate units are held-out units"
    return None
