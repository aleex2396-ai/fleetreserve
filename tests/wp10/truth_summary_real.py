"""The text of a truth build's summary file, made by the code that writes the real one.

`summary_text(written, setting)` builds a `truth_build.Build` of the setting (a unit entry with
only its counting days, no window, alternative or check), takes its document from
`truth_build.summary_document` (`written` is the tables' SHA-256 per unit and kind, as
`truth_build.write` returns it) and renders it with `truth_build.summary_text`. So the made-up
summary has the shape of the real file: the main setting writes no `setting` key and the held
setting writes it as the first line. A `setting` that is neither (a made-up unknown setting) is
put into the document of a main build before the real text is made.

A made-up truth build in a test writes this text as its `truth_build.SUMMARY_NAME`, so no test
can again hold a summary that the real writer never writes.
"""

from fleetreserve import truth, truth_build

COUNTING_DAYS = 100  # the gate units' minimum (`truth_build.all_findings`): no finding is written


def summary_text(written: dict[str, dict[str, str]], setting: str = truth.MAIN) -> str:
    """`written`: unit -> {"cars": SHA-256, "days": SHA-256}."""
    build = truth_build.Build(setting if setting in truth.SETTINGS else truth.MAIN)
    for unit in written:
        build.summary[unit] = {"counting_days": COUNTING_DAYS}
    document = truth_build.summary_document(build, written)
    if setting not in truth.SETTINGS:
        document["setting"] = setting
    return truth_build.summary_text(document)
