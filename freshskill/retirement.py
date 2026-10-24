"""Retirement rules compared in Stage 5.7 / E1-E3.

All rules share one interface::

    rule.decide(series, status="active") -> KEEP | RETIRE | READMIT

``series`` is a time-ordered list of ``{"step", "y", "N", "s2"}`` (see
:func:`series_from_timeline`).  ``KEEP`` means "leave the skill active" when
``status == "active"``, and "stay retired" when ``status == "retired"``.

Rules
-----
B1 ``SLIMStyle``          threshold + patience (SLIM Eq.6, stylised)
B2 ``SkillForgeStyle``    trial -> active -> retired hysteresis, absorbing
B3 ``ConfirmRule``        two consecutive below-threshold cycles (Stage 5.6)
B4 ``PosteriorRule``      P(AR(1) stays harmful for H cycles) >= 1-alpha
B5 ``PosteriorVOIRule``   B4 plus value-of-information measurement allocation

B1/B2 are *stylised* reimplementations: the published methods operate on long
sequences with EMA smoothing, which our 3-5 checkpoint budget does not
provide.  E1 therefore reports "threshold rules under this budget", not
"SLIM as published".

The decision threshold and the cost ratio are the same object::

    retire  <=>  c_FR (1 - p) < c_MH p  <=>  p > c_FR / (c_FR + c_MH)
    alpha = c_MH / (c_FR + c_MH)

so ``alpha = 0.05`` means a false retirement is 19x costlier than a missed
retirement.  ``VOI`` uses the same costs but only chooses *what to measure
next*; it never overrides the decision rule.

Readmission is the same cost comparison in reverse.  Retiring while useful
costs ``c_FR``; readmitting a harmful skill costs ``c_MH`` plus one probe
(``c_probe = c_MH``) that must be paid to learn whether it recovered::

    readmit  <=>  (1 - p_U) c_MH + c_probe < p_U c_FR
    <=>  p_U > (c_MH + c_probe) / (c_FR + c_MH) = 2 alpha

``p_U`` here is P(utility is above ``theta`` *now*), not
``p_recovery`` (P(*any* of the next H cycles clears ``theta_keep``)): the
latter is ~0.3 even for a clearly harmful skill and would readmit everything.
The dead band ``[alpha, 2 alpha]`` is what stops the rule from flapping.
"""

import math
import random

from .measurement import standard_error
from .state_space import (
    p_persistent,
    p_recovery,
    posterior_after_observation,
    predictive,
    run_series,
)

KEEP = "KEEP"
RETIRE = "RETIRE"
READMIT = "READMIT"


# ---------------------------------------------------------------- helpers
def costs_from_alpha(alpha):
    """Normalised (c_false_retire, c_missed_harm) with c_missed_harm = 1."""
    if not (0.0 < alpha < 1.0):
        raise ValueError("alpha must be in (0, 1)")
    return (1.0 - alpha) / alpha, 1.0


def alpha_from_costs(c_false_retire, c_missed_harm):
    total = c_false_retire + c_missed_harm
    if total <= 0:
        raise ValueError("costs must be positive")
    return c_missed_harm / total


def series_from_timeline(timeline, skill_id):
    """Build the rule input from a ``UtilityTimeline`` entry."""
    out = []
    for step in timeline.steps(skill_id):
        c = timeline.counts(skill_id, step) or {}
        h, m, neutral = c.get("help", 0), c.get("harm", 0), c.get("neutral", 0)
        N = c.get("usage", h + m + neutral)
        if N <= 0:  # an unmeasured checkpoint is not an observation of zero
            continue
        out.append({
            "step": step,
            "y": timeline.utility(skill_id, step),
            "N": N,
            "s2": standard_error(h, m, neutral) ** 2,
        })
    return out


# ---------------------------------------------------------------- B1
class SLIMStyle:
    """Threshold + patience.  Retire after ``patience`` consecutive cycles."""

    def __init__(self, theta=-0.02, n_min=0, patience=1):
        self.theta = theta
        self.n_min = n_min
        self.patience = patience

    def decide(self, series, status="active"):
        if status == "retired":
            return RETIRE  # absorbing, by construction
        streak = 0
        for e in reversed(series):
            if e["y"] < self.theta and e.get("N", 10 ** 9) >= self.n_min:
                streak += 1
            else:
                break
        return RETIRE if streak >= self.patience else KEEP


# ---------------------------------------------------------------- B2
class SkillForgeStyle:
    """trial -> active -> retired state machine (absorbing retired).

    A skill stays in *trial* until it first clears ``delta_stable``.  Trial is
    not a hiding place: a skill that never clears the bar but keeps producing
    clearly harmful results is retired after ``patience`` consecutive cycles.
    (The earlier version recomputed the state from ``series[0]`` only, so a
    skill whose first checkpoint was negative could never retire.)
    """

    def __init__(self, delta_stable=0.0, delta_demote=-0.01,
                 delta_retire=-0.02, patience=1):
        self.delta_stable = delta_stable
        self.delta_demote = delta_demote
        self.delta_retire = delta_retire
        self.patience = patience

    def decide(self, series, status="active"):
        if status == "retired":
            return RETIRE  # absorbing by construction
        if not series:
            return KEEP  # an unmeasured skill must never be retired
        first_active = next((i for i, e in enumerate(series)
                             if e["y"] >= self.delta_stable), None)
        start = first_active if first_active is not None else 0
        streak = 0
        for e in series[start:]:
            y = e["y"]
            if y < self.delta_retire:
                streak += 1
                if streak >= self.patience:
                    return RETIRE
            elif y > self.delta_demote:
                streak = 0
        return KEEP


# ---------------------------------------------------------------- B3
class ConfirmRule(SLIMStyle):
    """Two consecutive cycles below theta (Stage 5.6)."""

    def __init__(self, theta=-0.02, n_min=0, cycles=2):
        super().__init__(theta=theta, n_min=n_min, patience=cycles)


# ---------------------------------------------------------------- B4
class PosteriorRule:
    """Retire iff P(the skill stays harmful for H cycles) >= 1 - alpha."""

    def __init__(self, phi, tau, alpha=0.05, theta=-0.02, theta_keep=0.02,
                 H=2, M=2000, seed=0, c_probe=1.0):
        self.phi = phi
        self.tau = tau
        self.alpha = alpha
        self.theta = theta
        self.theta_keep = theta_keep
        self.H = H
        self.M = M
        self.seed = seed
        self.c_fr, self.c_mh = costs_from_alpha(alpha)
        self.c_probe = c_probe

    # -- posterior --------------------------------------------------------
    def posterior(self, series):
        ys = [e["y"] for e in series]
        s2s = [e["s2"] for e in series]
        sm_m, sm_P = run_series(self.phi, self.tau, ys, s2s)
        return sm_m[-1], sm_P[-1]

    def harm_probability(self, series):
        """P(current utility is below theta)."""
        from .state_space import p_harm
        mu, var = self.posterior(series)
        return p_harm(mu, var, self.theta)

    def useful_probability(self, series):
        """P(current utility is above theta) -- the readmission statistic."""
        return 1.0 - self.harm_probability(series)

    @property
    def readmit_threshold(self):
        """(c_MH + c_probe) / (c_FR + c_MH), i.e. 2 alpha when c_probe=c_MH."""
        return ((self.c_mh + self.c_probe) / (self.c_fr + self.c_mh))

    def persistence_probability(self, series):
        mu, var = self.posterior(series)
        rng = random.Random(self.seed)
        return p_persistent(mu, var, self.phi, self.tau, self.H, self.theta,
                            M=self.M, rng=rng)

    def recovery_probability(self, series):
        mu, var = self.posterior(series)
        rng = random.Random(self.seed)
        return p_recovery(mu, var, self.phi, self.tau, self.H, self.theta_keep,
                          M=self.M, rng=rng)

    # -- decision ---------------------------------------------------------
    def decide(self, series, status="active"):
        if not series:
            return RETIRE if status == "retired" else KEEP
        if status == "retired":
            return (READMIT if self.useful_probability(series)
                    >= self.readmit_threshold else RETIRE)
        if self.persistence_probability(series) >= 1.0 - self.alpha:
            return RETIRE
        return KEEP

    # -- loss -------------------------------------------------------------
    def expected_loss(self, p):
        return min(self.c_fr * (1.0 - p), self.c_mh * p)


# ---------------------------------------------------------------- B5
class PosteriorVOIRule(PosteriorRule):
    """B4 plus a value-of-information rule for *which skill to measure next*."""

    def voi(self, series, s2_next=None, grid=81, M=500):
        """Value of measuring this skill once more, in loss units.

        Both branches are scored on the *same* decision -- the one taken at
        T+1 -- so the difference is information, not a shifted horizon.  The
        no-measurement branch therefore starts from the prior state at T+1
        (variance ``phi^2 P + tau^2``, without the observation noise); the
        measurement branch conditions on y at T+1 and then evaluates the same
        H-cycle window.  (Previously the no-measurement branch was evaluated
        at T, one cycle earlier.)
        """
        if s2_next is None:
            s2_next = series[-1]["s2"]
        mu_T, P_T = self.posterior(series)
        prior_var = self.phi ** 2 * P_T + self.tau ** 2
        prior_mean = self.phi * mu_T
        rng = random.Random(self.seed)
        p0 = p_persistent(prior_mean, prior_var, self.phi, self.tau, self.H,
                          self.theta, M=max(M, 800), rng=rng)
        loss_now = self.expected_loss(p0)

        mean, var = predictive(mu_T, P_T, self.phi, self.tau, s2_next)
        sd = math.sqrt(var)
        if sd <= 0:
            return 0.0
        expected_after = 0.0
        total_w = 0.0
        step = 8.0 * sd / (grid - 1)
        for i in range(grid):
            y = mean - 4.0 * sd + i * step
            w = math.exp(-0.5 * ((y - mean) / sd) ** 2)
            post_m, post_P = posterior_after_observation(
                mu_T, P_T, self.phi, self.tau, s2_next, y)
            p = p_persistent(post_m, post_P, self.phi, self.tau, self.H,
                             self.theta, M=M, rng=random.Random(self.seed + i))
            expected_after += w * self.expected_loss(p)
            total_w += w
        expected_after /= total_w
        return max(loss_now - expected_after, 0.0)

    def next_target(self, series_by_skill, cost=0.0):
        """Return (skill_id, voi) with the largest VOI - cost, or (None, 0)."""
        best_id, best = None, cost
        for sid, series in series_by_skill.items():
            v = self.voi(series)
            if v > best:
                best_id, best = sid, v
        return best_id, (best if best_id else 0.0)
