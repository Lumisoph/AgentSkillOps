"""Hierarchical AR(1) state-space model for skill utility trajectories.

Stage 5.7 / E2 (experiment_design.md v3.1 section 5).

    mu_i(t) = phi * mu_i(t-1) + eta_i(t),   eta_i(t) ~ N(0, tau_i^2)
    y_i(t)  = mu_i(t) + eps_i(t),           eps_i(t) ~ N(0, s_i(t)^2)

``phi`` is shared across skills (with 6 skills x 3-5 checkpoints a per-skill
regression coefficient is not identifiable); ``tau_i`` is partially pooled,
``log tau_i ~ N(log tau_0, sigma_tau^2)``.

Why AR(1) and not a random walk: a random walk (phi=1) forecasts the current
value forever, so a skill that dipped is expected never to come back.  Fitting
a random walk therefore *overstates persistence* and makes retirement more
likely -- it would manufacture the very error this stage is trying to avoid.
``phi`` is also the direct answer to RQ-B: phi -> 0 means harm is transient,
phi -> 1 means it persists.

Everything here is pure Python (no numpy) so the self-checks are deterministic
and the module stays importable in any environment.
"""

import math
import random

CHI2_1_95 = 1.920729  # -2 ln L / 2 threshold for a 95% interval on one parameter
REF_SPAN = 20.0       # Stage 5.5 checkpoint spacing (policy steps) = one unit


def gaps_from_steps(steps, ref=REF_SPAN):
    """Spacing of each observation from the previous one, in reference units.

    Element 0 is unused (the first observation initialises the state) and is
    set to 1.0 so the list stays aligned with ``y``.  A timeline with t=0, 20,
    40 gives [1, 1, 1]; t=0, 10, 20, 30, 40 gives [1, 0.5, 0.5, 0.5, 0.5].
    """
    return [1.0] + [(b - a) / ref for a, b in zip(steps[:-1], steps[1:])]


# ---------------------------------------------------------------- filtering
class AR1Filter:
    """Kalman filter/smoother for one skill's series with known obs variance."""

    def __init__(self, phi, tau, prior_mean=0.0, prior_var=None):
        if not (0.0 <= phi <= 1.0):
            raise ValueError("phi must be in [0, 1]")
        if tau < 0:
            raise ValueError("tau must be non-negative")
        self.phi = float(phi)
        self.tau = float(tau)
        self.prior_mean = float(prior_mean)
        if prior_var is None:
            # Diffuse prior on the utility scale (u lives in [-1, 1]).  The
            # stationary AR(1) variance tau^2/(1-phi^2) is *not* a safe default:
            # when tau sits at the grid floor it collapses to ~0 and anchors the
            # state at zero, which silently turns the rule into "never retire"
            # and destroys its power.
            prior_var = 1.0
        if not prior_var or prior_var <= 1e-12 or math.isnan(prior_var):
            prior_var = 1.0
        self.prior_var = float(prior_var)

    # -- transition -------------------------------------------------------
    def _transition(self, gap):
        """(a, Q) over ``gap`` reference intervals.

        The AR(1) is parametrised per *reference* interval, so an observation
        that is ``gap`` intervals after the previous one has state factor
        ``phi**gap`` and integrated innovation variance
        ``tau^2 (1 - phi^(2 gap)) / (1 - phi^2)`` (``tau^2 gap`` at phi=1).
        Without this, a series sampled twice as often has its per-observation
        persistence read as ``phi**0.5`` -- a pure time-unit artefact.
        """
        a = self.phi ** gap
        if abs(1.0 - self.phi ** 2) < 1e-12:
            Q = self.tau ** 2 * gap
        else:
            Q = self.tau ** 2 * (1.0 - self.phi ** (2.0 * gap)) / \
                (1.0 - self.phi ** 2)
        return a, Q

    # -- forward ----------------------------------------------------------
    def filter(self, ys, s2s, gaps=None):
        if len(ys) != len(s2s):
            raise ValueError("ys and s2s must have the same length")
        if gaps is None:
            gaps = [1.0] * len(ys)
        elif len(gaps) != len(ys):
            raise ValueError("gaps and ys must have the same length")
        m, P = self.prior_mean, self.prior_var
        pred_m, pred_P, filt_m, filt_P, factors = [], [], [], [], []
        ll = 0.0
        for y, s2, gap in zip(ys, s2s, gaps):
            if gap <= 0:
                raise ValueError("gaps must be positive")
            a, Q = self._transition(gap)
            mp = a * m
            Pp = a ** 2 * P + Q
            v = Pp + s2
            if v <= 0:
                v = 1e-12
            ll += -0.5 * (math.log(2.0 * math.pi * v) + (y - mp) ** 2 / v)
            K = Pp / v
            m = mp + K * (y - mp)
            P = (1.0 - K) * Pp
            pred_m.append(mp)
            pred_P.append(Pp)
            filt_m.append(m)
            filt_P.append(P)
            factors.append(a)
        return {
            "pred_mean": pred_m,
            "pred_var": pred_P,
            "filt_mean": filt_m,
            "filt_var": filt_P,
            "factor": factors,
            "loglik": ll,
        }

    def loglik(self, ys, s2s, gaps=None):
        return self.filter(ys, s2s, gaps)["loglik"]

    # -- backward ---------------------------------------------------------
    def smooth(self, ys, s2s, gaps=None):
        f = self.filter(ys, s2s, gaps)
        n = len(ys)
        sm_m = list(f["filt_mean"])
        sm_P = list(f["filt_var"])
        for t in range(n - 2, -1, -1):
            denom = f["pred_var"][t + 1]
            if denom <= 1e-12:
                C = 0.0
            else:
                C = f["filt_var"][t] * f["factor"][t + 1] / denom
            sm_m[t] = f["filt_mean"][t] + C * (sm_m[t + 1] - f["pred_mean"][t + 1])
            sm_P[t] = f["filt_var"][t] + C ** 2 * (sm_P[t + 1] - f["pred_var"][t + 1])
            sm_P[t] = max(sm_P[t], 0.0)
        return sm_m, sm_P


# ---------------------------------------------------------------- estimation
def _default_phis():
    return [i / 20.0 for i in range(21)]           # 0.00 .. 1.00


def _default_taus():
    # log-spaced 1e-4 .. 0.2
    lo, hi, n = math.log(1e-4), math.log(0.2), 25
    return [math.exp(lo + (hi - lo) * i / (n - 1)) for i in range(n)]


def _default_tau0_grid(taus):
    return [t for t in taus if 1e-4 <= t <= 0.2]


def _loglik_conditioned(phi, tau, ys, s2s, gaps=None):
    """Conditional likelihood: the first observation initialises the state.

    Using a diffuse prior instead lets the prior dominate at small n (it pulls
    phi towards 0); using an informative prior makes the interval falsely
    narrow.  Conditioning on y_0 is prior-free and honest: with 3-5 points the
    interval is simply wide.

    ``gaps[k]`` is the spacing between observation k-1 and k in units of the
    reference interval (1.0 for the Stage 5.5 20-step checkpoints).  The 5-point
    timeline adds t=10/t=30, so its interior gaps are 0.5; treating those as
    full steps would read the per-observation persistence as ``phi**0.5``.
    """
    if len(ys) < 2:
        return 0.0
    f = AR1Filter(phi, tau, prior_mean=ys[0],
                  prior_var=max(s2s[0], 1e-12))
    return f.loglik(ys[1:], s2s[1:], gaps[1:] if gaps else None)


def run_series(phi, tau, ys, s2s, gaps=None):
    """Smoothed posterior for one series, conditioned on the first observation."""
    if len(ys) < 2:
        return list(ys), [max(s, 1e-12) for s in s2s]
    f = AR1Filter(phi, tau, prior_mean=ys[0], prior_var=max(s2s[0], 1e-12))
    sm_m, sm_P = f.smooth(ys[1:], s2s[1:], gaps[1:] if gaps else None)
    return [ys[0]] + sm_m, [max(s2s[0], 1e-12)] + sm_P


def fit_panel(series, phis=None, taus=None, tau0_grid=None, sigma_tau=0.5):
    """Fit shared ``phi`` and partially pooled ``tau_i`` by penalised likelihood.

    ``series``: list of dicts ``{"id": str, "y": [...], "s2": [...]}``.

    Returns a dict with the best ``phi``, ``tau0``, per-skill ``tau``, the
    profile likelihood over ``phi`` (for a likelihood-ratio interval) and a
    ``degenerate`` flag when ``tau0`` collapses to zero (the model then says
    "utility is constant", i.e. no evidence of dynamics).
    """
    if not series:
        raise ValueError("empty panel")
    phis = phis or _default_phis()
    taus = taus or _default_taus()
    tau0_grid = tau0_grid or _default_tau0_grid(taus)
    log_taus = [math.log(t) for t in taus]

    # ll[phi_i][skill_j][tau_k]
    ll = []
    for phi in phis:
        per_skill = []
        for s in series:
            row = []
            for tau in taus:
                row.append(_loglik_conditioned(phi, tau, s["y"], s["s2"],
                                               s.get("gaps")))
            per_skill.append(row)
        ll.append(per_skill)

    best = None
    profile = []
    for pi, phi in enumerate(phis):
        best_phi_score = -float("inf")
        for tau0 in tau0_grid:
            log_tau0 = math.log(tau0)
            score = 0.0
            choice = []
            for sj, _ in enumerate(series):
                best_local, best_tau = -float("inf"), taus[0]
                for ti, tau in enumerate(taus):
                    pen = -0.5 * ((log_taus[ti] - log_tau0) / sigma_tau) ** 2
                    val = ll[pi][sj][ti] + pen
                    if val > best_local:
                        best_local, best_tau = val, tau
                score += best_local
                choice.append(best_tau)
            if score > best_phi_score:
                best_phi_score = score
                if best is None or score > best["loglik"]:
                    best = {
                        "phi": phi,
                        "tau0": tau0,
                        "tau": {s["id"]: t for s, t in zip(series, choice)},
                        "loglik": score,
                    }
        profile.append((phi, best_phi_score))

    best["profile"] = profile
    best["phi_ci95"] = _profile_interval(profile)
    best["degenerate"] = best["tau0"] < 1e-3
    best["phi_fitted"] = best["phi"]
    if best["degenerate"]:
        # tau -> 0 with phi < 1 describes a deterministic *decay* to zero, not
        # "no dynamics".  The degenerate model we actually want is a constant
        # mean, i.e. phi = 1.  Using the fitted phi here silently shrinks every
        # utility towards zero by phi^t and makes the rule confidently wrong.
        best["phi"] = 1.0
    return best


def _profile_interval(profile):
    top = max(v for _, v in profile)
    cut = top - CHI2_1_95
    inside = [p for p, v in profile if v >= cut]
    if not inside:
        return (float("nan"), float("nan"))
    return (min(inside), max(inside))


# ---------------------------------------------------------------- posteriors
def sample_paths(mu_T, P_T, phi, tau, H, M=2000, rng=None):
    """M paths of length H from the posterior at time T (finite horizon only)."""
    if H <= 0:
        raise ValueError("H must be positive")
    rng = rng or random.Random(0)
    sd = math.sqrt(max(P_T, 0.0))
    paths = []
    for _ in range(M):
        mu = rng.gauss(mu_T, sd) if sd > 0 else mu_T
        path = []
        for _h in range(H):
            mu = phi * mu + (rng.gauss(0.0, tau) if tau > 0 else 0.0)
            path.append(mu)
        paths.append(path)
    return paths


def p_persistent(mu_T, P_T, phi, tau, H, theta, M=2000, rng=None):
    """P(mu stays below ``theta`` for all of the next H checkpoints).

    H must be finite: with phi < 1 the AR(1) process returns to zero, and with
    phi = 1 the random walk is recurrent -- in either case the probability of
    staying below a fixed threshold *forever* is zero.
    """
    paths = sample_paths(mu_T, P_T, phi, tau, H, M=M, rng=rng)
    return sum(1 for p in paths if all(v < theta for v in p)) / len(paths)


def p_recovery(mu_T, P_T, phi, tau, H, theta_keep, M=2000, rng=None):
    """P(at least one of the next H checkpoints rises above ``theta_keep``)."""
    paths = sample_paths(mu_T, P_T, phi, tau, H, M=M, rng=rng)
    return sum(1 for p in paths if any(v > theta_keep for v in p)) / len(paths)


def p_harm(mu, var, theta):
    """P(mu < theta) for a Gaussian posterior."""
    if var <= 0:
        return 1.0 if mu < theta else 0.0
    return 0.5 * math.erfc((mu - theta) / math.sqrt(2.0 * var))


# ---------------------------------------------------------------- VOI pieces
def predictive(mu_T, P_T, phi, tau, s2):
    """Predictive for one more measurement at T+1: mean, variance."""
    mean = phi * mu_T
    var = phi ** 2 * P_T + tau ** 2 + s2
    return mean, max(var, 1e-12)


def posterior_after_observation(mu_T, P_T, phi, tau, s2, y):
    """Posterior over mu_{T+1} given one new observation y."""
    mp = phi * mu_T
    pp = phi ** 2 * P_T + tau ** 2
    v = pp + s2
    if v <= 0:
        return mp, 0.0
    K = pp / v
    return mp + K * (y - mp), max((1.0 - K) * pp, 0.0)
