"""The drift-guard update of the scale beta (plan section 6, T10).

After a day with loss `loss`, beta moves to beta + eta * (alpha - loss). A scale of zero or
below abstains. With `clip` the new scale is not allowed below 0; the unclipped rule is the one
with the stated bound on the average loss.
"""


def step(beta, loss, alpha, eta, *, clip):
    """The scale after one day with loss `loss`."""
    new_beta = beta + eta * (alpha - loss)
    return max(new_beta, 0) if clip else new_beta


def run(loss_rule, days, alpha, eta, *, beta_1=0, clip):
    """Run `days` days from `beta_1`; `loss_rule(beta)` gives the day's loss. Return the losses."""
    beta = beta_1
    losses = []
    for _ in range(days):
        loss = loss_rule(beta)
        losses.append(loss)
        beta = step(beta, loss, alpha, eta, clip=clip)
    return losses
