"""The configurations that failed on a unit in the learn steps of the drive (plan section 13, WP14).

On the sealed route a configuration whose scoring raised an error on a unit is written as
`failed: <error>` in `learn.json`, and the step still exits 0 (`learn_run`); WP14's pass mark
fails there. A drive in which every step exited 0 is therefore not "every step passed" when this
list is not empty. `failed_configurations(source)` reads each learn step that passed
(`learn-<model>`): the list `failed_configurations` of its `learn.json` (`<configuration> | <unit>:
<error>`) and every configuration, or oracle partner, of a unit whose rungs hold the reason
`failed: <error>` (the list names one configuration of a group that failed together, the rungs all
of them), each as `<step>: <configuration> | <unit>: <error>`, without repeats. A learn step
that passed whose `learn.json` cannot be read is returned apart (`unreadable`): the banner
names it and the exit status of the command is 1.

`information_only(source)` is the one decision whether the WP15 parts of the tables are printed
as information only (plan section 13, WP15: "after a held-out run that WP14 fails, the parts
readable from steps that passed are printed as information only"): true unless the drive is
complete and no configuration failed on a unit and no learn result is unreadable, that is, unless
the banner says the run is a passed held-out run. Every WP15 part (the rho 0.1 and 0.2 rows of
`wp16_o1` and `o5-drift`) takes its mark from it; none decides for itself.
"""

from fleetreserve.wp16_drive import Missed, Source

START = "failed:"
SUFFIX = " [information only]"  # added to the first cell of a WP15 row of a table, when marked


def failed_configurations(source: Source) -> tuple[list[str], list[str]]:
    """(the failed configurations, the learn steps whose `learn.json` cannot be read, each as
    `<step>: <why>`): an unreadable results file is never skipped in silence."""
    found: list[str] = []
    unreadable: list[str] = []
    for label, step in source.drive.steps.items():
        if not label.startswith("learn-") or step.status != "passed":
            continue
        try:
            document = source.json(label, "learn.json")
        except Missed as problem:
            unreadable.append(f"{label}: {problem.text}")
            continue
        listed = document.get("failed_configurations")
        for text in listed if isinstance(listed, list) else []:
            if f"{label}: {text}" not in found:
                found.append(f"{label}: {text}")
        units = document.get("units")
        for unit, entry in (units if isinstance(units, dict) else {}).items():
            configurations = entry.get("configurations") if isinstance(entry, dict) else None
            for name, offer in (configurations if isinstance(configurations, dict) else {}).items():
                if not isinstance(offer, dict):
                    continue
                partner = offer.get("oracle_partner")
                partner = partner if isinstance(partner, dict) else {}
                for cell, rungs in (
                    (name, offer.get("rungs")),
                    (partner.get("name"), partner.get("rungs")),
                ):
                    reason = rungs.get("reason") if isinstance(rungs, dict) else None
                    if (
                        isinstance(cell, str)
                        and isinstance(reason, str)
                        and reason.startswith(START)
                    ):
                        text = f"{label}: {cell} | {unit}: {reason.removeprefix(START).strip()}"
                        if text not in found:
                            found.append(text)
    return found, unreadable


def information_only(source: Source) -> bool:
    """Whether the WP15 parts are printed as information only (module text): true when WP14's pass
    mark is not met, as the banner of every table says."""
    failed, unreadable = failed_configurations(source)
    return not (source.drive.complete and not failed and not unreadable)


def information_note(part: str) -> str:
    """The note of a table that holds a WP15 part marked as information only; `part` is the
    subject and its verb, such as "this WP15 part is"."""
    return (
        "Information only: the run's WP14 pass mark is not met (a step failed or did not run, or "
        f"a configuration failed on a unit), so {part} printed as information only and WP15's "
        "pass mark is not passed (plan section 13)."
    )
