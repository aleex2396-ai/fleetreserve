"""The figures of WP16, drawn with matplotlib from the figure data the table builders return.

`render(spec, folder)` writes one PNG per figure and returns its name. A spec is a dict with a
`kind`:
- `o3`: `points`, one tuple per unit (unit, LP (24), general split, LP (27) largest found, LP (27)
  bound), each a per cent of the ceiling or None: grouped bars per unit; a missing bound is not
  drawn and is named in the figure's note;
- `o4`: `model`, `points`, one tuple per unit (unit, availability cell, shortfall cell), the cells
  `Interval`s or None: two panels with the 95% intervals and the pass marks (90% and 5%);
- `o9`: `curves`, one tuple per model and unit (model, unit, [(k, A_k, Y_k)]): one panel per unit;
- `o10`: `rows`, one tuple per model and baseline (model, baseline, change in per cent or None).

A figure with no value to draw is written as a note-only figure, never skipped, so the file exists
and says why; nothing is filled from elsewhere. The matplotlib group is `figures` (pyproject).
"""

import math
from pathlib import Path

from fleetreserve.wp16_cells import Interval

COLORS = ("#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9")
O3_SERIES = ("LP (24)", "general split", "LP (27) largest found", "LP (27) proved bound")


def _plt():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _number(value) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool) and math.isfinite(value)


def _empty(plt, name: str, text: str):
    figure, axes = plt.subplots(figsize=(6, 2))
    axes.axis("off")
    axes.text(0.5, 0.5, text, ha="center", va="center", wrap=True)
    return figure, name


def _o3(plt, spec: dict):
    points = spec.get("points") or []
    if not points:
        return _empty(plt, "o3-ceiling-shares", "O3: no unit has a figure")
    figure, axes = plt.subplots(figsize=(max(6, 1.1 * len(points)), 4))
    width = 0.2
    for s, series in enumerate(O3_SERIES):
        # a missing value gets no bar (a bar of height 0 would say it was 0)
        have = [i for i, p in enumerate(points) if _number(p[1 + s])]
        axes.bar(
            [i + (s - 1.5) * width for i in have],
            [points[i][1 + s] for i in have],
            width,
            label=series,
            color=COLORS[s],
        )
    axes.set_xticks(range(len(points)), [p[0] for p in points], rotation=30, ha="right")
    axes.set_ylabel("share of the ceiling (%), ratio of means")
    axes.set_title("O3: methods against the ceiling, hourly product, the same days")
    axes.legend()
    missing = [f"{p[0]}: {O3_SERIES[s]}" for p in points for s in range(4) if not _number(p[1 + s])]
    if missing:
        figure.text(0.01, 0.01, "no value (not drawn): " + "; ".join(missing)[:200], fontsize=6)
    figure.tight_layout()
    return figure, "o3-ceiling-shares"


def _o4(plt, spec: dict):
    model = spec.get("model", "model")
    points = [p for p in spec.get("points", [])]
    name = f"o4-availability-shortfall-{model}"
    if not points:
        return _empty(plt, name, f"O4 {model}: no unit has a figure")
    figure, axes = plt.subplots(1, 2, figsize=(max(8, 1.4 * len(points)), 4))
    for axis, k, mark, title in (
        (axes[0], 1, 0.90, "availability (pass: upper end >= 90%)"),
        (axes[1], 2, 0.05, "shortfall (pass: lower end <= 5%)"),
    ):
        for i, point in enumerate(points):
            cell = point[k]
            if not isinstance(cell, Interval) or not _number(cell.point):
                continue
            low = cell.low if _number(cell.low) else cell.point
            high = cell.high if _number(cell.high) else cell.point
            # the interval is drawn as a line between its ends: a percentile interval need not
            # hold the point, so an error bar of point-minus-end lengths could be negative
            axis.vlines(i, min(low, high), max(low, high), color=COLORS[0])
            axis.plot(i, cell.point, "o", color=COLORS[0])
        axis.axhline(mark, color=COLORS[1], linestyle="--")
        axis.set_xticks(range(len(points)), [p[0] for p in points], rotation=30, ha="right")
        axis.set_title(f"{model}: {title}", fontsize=9)
    figure.tight_layout()
    return figure, name


def _o9(plt, spec: dict):
    curves = spec.get("curves") or []
    if not curves:
        return _empty(plt, "o9-cold-start", "O9: no unit has a figure")
    units = sorted({u for _, u, _ in curves})
    models = sorted({m for m, _, _ in curves})
    columns = min(3, len(units))
    rows = math.ceil(len(units) / columns)
    figure, axes = plt.subplots(rows, columns, figsize=(4 * columns, 3 * rows), squeeze=False)
    for axis in axes.flat:
        axis.axis("off")
    for i, unit in enumerate(units):
        axis = axes.flat[i]
        axis.axis("on")
        for m, model in enumerate(models):
            for model_name, unit_name, pts in curves:
                if model_name != model or unit_name != unit:
                    continue
                ks = [p[0] for p in pts]
                axis.plot(
                    ks,
                    [p[1] if _number(p[1]) else math.nan for p in pts],
                    "-o",
                    color=COLORS[m],
                    label=f"{model} A_k",
                )
                axis.plot(
                    ks,
                    [p[2] if _number(p[2]) else math.nan for p in pts],
                    "--s",
                    color=COLORS[m],
                    label=f"{model} Y_k",
                )
        axis.set_title(unit, fontsize=9)
        axis.set_xlabel("local days k")
        axis.legend(fontsize=6)
    figure.suptitle("O9: availability A_k (solid) and offer share Y_k (dashed)")
    figure.tight_layout()
    return figure, "o9-cold-start"


def _o10(plt, spec: dict):
    rows = spec.get("rows") or []
    drawn = [r for r in rows if _number(r[2])]
    if not drawn:
        return _empty(plt, "o10-model-comparison", "O10: no comparison has a figure")
    baselines = sorted({b for _, b, _ in rows})
    models = sorted({m for m, _, _ in rows})
    figure, axes = plt.subplots(figsize=(max(6, 1.6 * len(baselines)), 4))
    width = 0.8 / len(models)
    for m, model in enumerate(models):
        found = {
            b: v for mm, b, v in rows if mm == model and _number(v)
        }  # no bar for a missing one
        have = [i for i, b in enumerate(baselines) if b in found]
        axes.bar(
            [i + m * width for i in have],
            [found[baselines[i]] for i in have],
            width,
            label=model,
            color=COLORS[m],
        )
    axes.set_xticks(
        [i + width * (len(models) - 1) / 2 for i in range(len(baselines))],
        baselines,
        rotation=20,
        ha="right",
    )
    axes.axhline(0, color="black", linewidth=0.5)
    axes.set_ylabel("change of the offer (%)")
    axes.set_title("O10: each model against the baselines")
    axes.legend()
    figure.tight_layout()
    return figure, "o10-model-comparison"


DRAW = {"o3": _o3, "o4": _o4, "o9": _o9, "o10": _o10}


def render(spec: dict, folder: Path) -> str:
    """Draw one figure into `folder` as a PNG; the file name written (module text)."""
    plt = _plt()
    figure, name = DRAW[spec["kind"]](plt, spec)
    path = Path(folder) / f"{name}.png"
    figure.savefig(path, dpi=150, metadata={"Software": None})
    plt.close(figure)
    return path.name
