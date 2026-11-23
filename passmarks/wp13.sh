# The pass mark of work package 13 (plan section 13, WP13: "Pass: tag exists, and is the commit
# checked out; held-out hashes match those committed at download and split; held-out list, run
# matrix, seeds (one check item ...); no opening logged"), sourced by passmark.sh. One line per
# clause, in the plan's order; a line is changed only together with tests/test_passmark.py, which
# names all four. Each item is a sub-command of tools/freeze_check.py, whose module text states
# what it reads and its readings F1 to F12. It creates no tag and decides nothing about the
# freeze: the owner decides the tag. Run it in the project's own folder, which holds the sealed
# files and the openings log, with HEAD at the tag and a clean tree.
# The hashes item reads each sealed file's bytes to hash them (as WP1's first item does) and parses
# none; the other three open nothing in the sealed folder.
# The lists item checks the run matrix as WP11's second item does, and reads the run list's seeds
# and the held-out units (sealed_route.HELD_OUT_UNITS) against it and the sealed matrices.
# The last item passes only BEFORE the run from the tag: that run opens every sealed set and each
# opening is logged, so afterwards the item fails by design. WP13's pass is recorded once, at the
# tag, before the run from the tag (WP14).
# No tag, a changed committed entry or file, a held-out unit added, dropped or swapped, a literal
# seed, or a missing one that the step's command would accept, or a logged opening fails its item;
# HEAD elsewhere, an unclean tree, a history cut short, a missing file or one linked outside the
# sealed folder, no matrix written, or an unreadable log holding no opening: the item could not run
# and is never a pass.
check "tag exists: refs/tags/frozen-plan names a commit, and HEAD is that commit in a clean tree (plan 13 WP13)" uv run --frozen python tools/freeze_check.py tag
check "held-out hashes match those committed at download and split: the held-out entries of data-hashes.json unchanged on HEAD's first-parent path since the commit that first holds it, and the sealed files of those sizes and SHA-256 hashes, with no other file in the sealed folder outside any folder in it that is a symbolic link (plan 13 WP13)" uv run --frozen python tools/freeze_check.py hashes
check "held-out list, run matrix, seeds: the nine held-out units of sealed_route.HELD_OUT_UNITS exactly the gate units of every sealed matrix and the units of the run list's sealed-run groups; the run matrix frozen as WP11's item checks it; every seed of the run list's entries one of its fixed seeds, none literal or given twice, and none left at a command's default unless that command's own parser or check refuses it on that step (the owner's decision of 19 November 2026), and the run list's own seed rules (L12, L13) holding (plan 13 WP13)" uv run --frozen python tools/freeze_check.py lists
check "no opening logged: the openings log absent or empty, with the sealed files present in this working copy (plan 13 WP13; passes only before the run from the tag)" uv run --frozen python tools/freeze_check.py openings
