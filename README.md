# Fleet reserve: an hourly offer from charging electric cars

A car park holds many electric cars that are plugged in at once. Each morning, at a fixed decision time, the car park can offer a box: for each of a few later hours, a range of total charging power in kW that the combined charging of the cars must meet.
This project measures how large such an offer can be on public charging data.

## Outcome
- The held-out run did not meet its pass mark: 26 of its 30 steps passed, and the four that failed compute 95% intervals. The pass mark of work package 15 was also missed.
- The run judged no unit against the guarantee. Six of the nine units offer nothing, and the other three have no interval.
- The main model's offer is 130.153 kWh per counting test day, summed over the nine units. 51.03 and 78.48 of it come from one Dutch office car park in its two periods.
- A search of the extreme request paths inside each offered box finds at least one breaking path on 101 of 150 searched days at the three units that offer with the learned forecast, and on 0 of 250 with true departures; the two counts are not a matched comparison (`docs/results.md`, section 11).

## Where to start
- `docs/README.md`: the overview, the layout of the repository and how to run the checks.
- `docs/results.md`: the full account, with every table.
- `docs/plan.md`: the specification, frozen at the git tag `frozen-plan`.
- `docs/prior-looks.md`: every look at the data before the tag, and the looks taken after it.

The command that checks the code is `bash check.sh` (exit 0 passed, 1 a check failed, 2 a check could not run).
