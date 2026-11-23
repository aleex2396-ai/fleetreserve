"""The parts of the neural hazard model (plan section 7, "Ablations of the hazard model: no set
encoder, no site embedding, no day factor; calendar-only inputs"), as changes to
`hazard_settings.HazardSettings`.

`PARTS` maps the name of each part to the fields it changes; every other field is left as it is.
`CALENDAR_INPUTS` are the inputs the calendar-only part keeps (`HazardSettings.inputs`).
`fleetreserve.hazard_tune` measures the parts on the training sites, and the models
`hazard_tuned_no_set`, `hazard_tuned_no_site`, `hazard_tuned_no_day` and `hazard_tuned_calendar`
apply one part each to the tuned settings.

This module needs no package of the `learn` dependency group.
"""

CALENDAR_INPUTS = ("weekday", "month", "day_of_year", "public_holiday")
PARTS: dict[str, dict] = {
    "no_set_encoder": {"set_encoder": False},
    "no_site_embedding": {"site_embedding": False},
    "no_day_factor": {"day_factor": False},
    "no_set_encoder_no_day_factor": {"set_encoder": False, "day_factor": False},
    "calendar_only_inputs": {"inputs": CALENDAR_INPUTS},
}
