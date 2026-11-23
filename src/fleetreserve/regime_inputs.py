"""Regime inputs of the departure model: dated rules, known at the decision time τ (plan
sections 3 and 7).

Plan section 3 names the regimes: "NREL managed-charging controller from May 2019; a.s.r. paid
from September 2022, controller from March 2023; COVID (Norway's training sites end before it;
OSL_T, TRO_R span it)". Each column below is a rule on the row's `site` and `date` (the local
date of its day) alone, 1 on and after its first date and 0 before, a fixed date that was public
or known to the site's operator before the day, so every value is known at τ. The same rules hold
for every unit, the held-out units included; no data row is read.

- `regime_nrel_controller`: site `nrel`, date on or after `NREL_CONTROLLER_FROM`, 1 May 2019
  (the plan's "from May 2019", the date at which the NREL units are split).
- `regime_asr_paid`: site `asr`, date on or after `ASR_PAID_FROM`, 1 Sep 2022 (the a.s.r.
  README: "After September 1st, 2022, the price changed to 37 cents/kWh"; the plan's "from
  September 2022"; the a.s.r. loader's `PAID_FROM`). [Alternative: from 2 Sep 2022, reading
  "after" strictly.]
- `regime_asr_controller`: site `asr`, date on or after `ASR_CONTROLLER_FROM`, 1 Mar 2023 (the
  README: "As of March 2023, a model predictive controller was deployed"; the loader's
  `CONTROLLER_FROM`; the README gives no day). The README's later target changes (23 Aug 2023,
  11 Dec 2023) are not regimes of the plan and get no column.
- `regime_covid`: any site, date from `COVID_FROM` to `COVID_TO`, 0 outside. The plan names the
  regime without dates; a reading: the day after the World Health Organization called COVID-19
  a pandemic (its media briefing of 11 Mar 2020), 12 Mar 2020, to the day the WHO ended the
  public health emergency of international concern, 5 May 2023, both announcements dated by
  their day (dates from memory, not checked against a source here). Each value is known at τ:
  a day on or after 12 Mar 2020 follows the announcement, and a day after 5 May 2023 follows the
  end. [Alternatives: national dates of the first closures (Norway 12 Mar 2020, the Netherlands
  15 Mar 2020, California 19 Mar 2020); a threshold on the OxCGRT index (`oxcgrt_inputs`),
  which is then a second reading of the same input.]

`NREL_SITES` and `ASR_SITES` name the sites each site rule covers. A date that cannot be read
raises `ValueError`.
"""

from datetime import date, timedelta

import numpy as np
import pandas as pd

NREL_CONTROLLER_FROM = date(2019, 5, 1)
ASR_PAID_FROM = date(2022, 9, 1)
ASR_CONTROLLER_FROM = date(2023, 3, 1)
COVID_FROM = date(2020, 3, 12)
COVID_TO = date(2023, 5, 5)
NREL_SITES = ("nrel",)
ASR_SITES = ("asr",)
COLUMNS = (
    "regime_nrel_controller",
    "regime_asr_paid",
    "regime_asr_controller",
    "regime_covid",
)


def regime_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """`COLUMNS` for the rows of `frame`, read from its `site` and `date` only (module text)."""
    days = pd.to_datetime(frame["date"].astype(str), format="%Y-%m-%d").dt.date.to_numpy()
    sites = frame["site"].astype(str).to_numpy()
    nrel = np.isin(sites, NREL_SITES)
    asr = np.isin(sites, ASR_SITES)
    after = np.vectorize(lambda d, first: d >= first, otypes=[bool])
    columns = {
        COLUMNS[0]: nrel & after(days, NREL_CONTROLLER_FROM),
        COLUMNS[1]: asr & after(days, ASR_PAID_FROM),
        COLUMNS[2]: asr & after(days, ASR_CONTROLLER_FROM),
        COLUMNS[3]: after(days, COVID_FROM) & ~after(days, COVID_TO + timedelta(days=1)),
    }
    return pd.DataFrame({k: v.astype(int) for k, v in columns.items()}, index=frame.index)
