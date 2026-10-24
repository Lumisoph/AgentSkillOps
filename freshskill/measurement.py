"""Measurement noise for paired skill-utility estimates.

Stage 5.7 / E0-E1 (experiment_design.md v3.1 section 3).

A skill's utility at one checkpoint is a paired difference over ``N`` tasks::

    X_j = won(full) - won(full - s_i)  in {-1, 0, +1}
    u   = (h - m) / N

Because ``u`` is a sample mean of the paired outcomes, its variance is the
variance of ``X`` divided by ``N``, *not* the variance of a difference of two
independent proportions::

    Var(u) = [(p_+ + p_-) - u^2] / N
    s_hat  = sqrt([(h + m)/N - u^2] / N)

This is the object that makes the identifiability problem visible: at N=140
the median standard error is ~0.028, the same order as the effects being
decided on (~0.02-0.06).
"""

import math
from statistics import NormalDist

_NORMAL = NormalDist()


# ---------------------------------------------------------------- point est.
def total(h, m, neutral):
    return h + m + neutral


def utility(h, m, neutral):
    """u = (h - m) / N; 0 for an empty sample."""
    n = total(h, m, neutral)
    return (h - m) / n if n else 0.0


def utility_variance(h, m, neutral):
    """Plug-in variance of the paired-mean estimator.

    ``n == 1`` has no within-cell variance, but returning 0 would make a
    single task look infinitely precise (and NaN would silently make every
    downstream probability zero).  X is ternary, so Var(X) <= 1 and the
    worst-case bound ``1/n`` is the honest choice here.
    """
    n = total(h, m, neutral)
    if n == 0:
        return 0.0
    if n == 1:
        return 1.0
    u = (h - m) / n
    return max((h + m) / n - u * u, 0.0) / n


def standard_error(h, m, neutral):
    return math.sqrt(utility_variance(h, m, neutral))


def zscore(h, m, neutral):
    se = standard_error(h, m, neutral)
    if not se or math.isnan(se):
        return float("nan")
    return utility(h, m, neutral) / se


def ci95(h, m, neutral, z=1.959963984540054):
    u = utility(h, m, neutral)
    se = standard_error(h, m, neutral)
    return (u - z * se, u + z * se)


def p_value_two_sided(z):
    if math.isnan(z):
        return float("nan")
    return math.erfc(abs(z) / math.sqrt(2.0))


# ---------------------------------------------------------------- planning
def required_n(delta, activity=0.115, alpha=0.05, power=0.8):
    """Tasks per skill per checkpoint for a two-sided test of size alpha.

    ``activity`` is p_+ + p_-, the fraction of tasks on which the skill
    changes the outcome at all (``u^2`` is negligible next to it, so it is
    dropped from the variance).
    """
    if delta <= 0:
        raise ValueError("delta must be positive")
    z_a = _NORMAL.inv_cdf(1.0 - alpha / 2.0)
    z_b = _NORMAL.inv_cdf(power)
    return math.ceil((z_a + z_b) ** 2 * activity / delta ** 2)


# ---------------------------------------------------------------- multiplicity
def bonferroni(pvalues, alpha=0.05):
    """Return (threshold, n_rejected) for a Bonferroni correction."""
    m = len(pvalues)
    if not m:
        return alpha, 0
    thr = alpha / m
    return thr, sum(1 for p in pvalues if p <= thr)


def benjamini_hochberg(pvalues, alpha=0.05):
    """Return (qvalues, rejected_mask) in the input order."""
    m = len(pvalues)
    if not m:
        return [], []
    order = sorted(range(m), key=lambda i: pvalues[i])
    q = [0.0] * m
    running = 1.0
    for rank in range(m, 0, -1):
        idx = order[rank - 1]
        running = min(running, pvalues[idx] * m / rank)
        q[idx] = running
    return q, [x <= alpha for x in q]


# ---------------------------------------------------------------- null model
def null_proportions(h, m, neutral):
    """Symmetric null with the same *activity* as the observation.

    Keeps p_+ + p_- (how often the skill changes an outcome) and sets
    p_+ = p_-, so the true utility is exactly zero.  Used by the E1 bootstrap:
    the empirical counts define the *population*, but the hypothesis being
    tested is "this skill does nothing".
    """
    n = total(h, m, neutral)
    if not n:
        return (0.0, 0.0, 1.0)
    active = (h + m) / n
    p_pos = active / 2.0
    p_neg = active / 2.0
    return (p_pos, p_neg, max(1.0 - active, 0.0))


def summarize(h, m, neutral):
    """All per-cell statistics used by the audit table."""
    u = utility(h, m, neutral)
    se = standard_error(h, m, neutral)
    z = u / se if se else float("nan")
    lo, hi = ci95(h, m, neutral)
    return {
        "n": total(h, m, neutral),
        "u": u,
        "se": se,
        "z": z,
        "p": p_value_two_sided(z),
        "ci95": (lo, hi),
        "significant": bool(abs(z) >= 1.959963984540054),
    }
