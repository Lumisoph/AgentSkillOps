"""Self-check for Stage 5.7 (design doc section 8, ten checks).

Pure Python, no model, deterministic.  Exit 0 iff all checks pass.

Usage:
    python experiments/selfcheck_stage57.py
"""

import math
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from freshskill import UtilityTimeline  # noqa: E402
from freshskill.measurement import standard_error, utility, utility_variance  # noqa: E402
from freshskill.retirement import (  # noqa: E402
    KEEP,
    READMIT,
    RETIRE,
    ConfirmRule,
    PosteriorRule,
    PosteriorVOIRule,
    SLIMStyle,
    SkillForgeStyle,
    alpha_from_costs,
    costs_from_alpha,
    series_from_timeline,
)
from freshskill.state_space import (  # noqa: E402
    AR1Filter,
    fit_panel,
    gaps_from_steps,
    p_persistent,
)

TIMELINE = Path(__file__).resolve().parent.parent / \
    "results/stage55_measure/utility_store_timeline.json"


def check(cond, msg):
    print(f"[{'PASS' if cond else 'FAIL'}] {msg}")
    return bool(cond)


def _ar1(n, phi, tau, mu0, rng, s2=0.028 ** 2):
    ys, mu = [], mu0
    for _ in range(n):
        mu = phi * mu + (rng.gauss(0, tau) if tau > 0 else 0.0)
        ys.append(mu + rng.gauss(0, math.sqrt(s2)))
    return {"id": "s", "y": ys, "s2": [s2] * n}


def main():
    ok = True

    # 1. variance formula matches the explicit paired sample
    h, m, neutral = 6, 10, 124
    xs = [1] * h + [-1] * m + [0] * neutral
    mean = sum(xs) / len(xs)
    var = sum((x - mean) ** 2 for x in xs) / len(xs) / len(xs)
    ok &= check(abs(utility(h, m, neutral) - mean) < 1e-12,
                "u equals the sample mean of paired outcomes")
    ok &= check(abs(utility_variance(h, m, neutral) - var) < 1e-9,
                "SE^2 equals Var(X)/N (not the two-proportion difference)")

    # 2. Kalman identities: precision-weighted average (flat prior) and
    #    the same with an N(0,1) prior.
    ys, s2s = [0.01, -0.02, 0.03], [0.001, 0.002, 0.0015]
    mu, _ = AR1Filter(1.0, 0.0, prior_var=1e9).smooth(ys, s2s)
    want = sum(y / s for y, s in zip(ys, s2s)) / sum(1 / s for s in s2s)
    ok &= check(abs(mu[-1] - want) < 1e-6,
                "tau=0, flat prior -> precision-weighted average")
    mu2, _ = AR1Filter(1.0, 0.0, prior_var=1.0).smooth(ys, s2s)
    want2 = sum(y / s for y, s in zip(ys, s2s)) / (sum(1 / s for s in s2s) + 1.0)
    ok &= check(abs(mu2[-1] - want2) < 1e-6,
                "tau=0, N(0,1) prior -> shrunk precision-weighted average")

    # 3. phi recovery on a synthetic AR(1) panel where the signal is strong
    #    enough to be identifiable (20 points, process sd > obs se)
    rng = random.Random(3)
    panel = [_ar1(20, 0.5, 0.03, 0.0, rng, s2=0.01 ** 2) for _ in range(6)]
    for i, p in enumerate(panel):
        p["id"] = f"s{i}"
    fit = fit_panel(panel)
    lo, hi = fit["phi_ci95"]
    ok &= check(0.35 <= fit["phi"] <= 0.75 and lo <= 0.5 <= hi,
                f"phi recovery: fitted {fit['phi']:.2f}, CI=[{lo:.2f},{hi:.2f}] "
                f"covers 0.50")

    # 3b. in the *real* regime (5 points, SE ~ 0.028) phi must be reported as
    #     practically unidentifiable -- the interval should be wide and cover 0.5.
    rng = random.Random(33)
    weak = [_ar1(5, 0.5, 0.03, 0.0, rng) for _ in range(6)]
    for i, p in enumerate(weak):
        p["id"] = f"s{i}"
    fit_weak = fit_panel(weak)
    w_lo, w_hi = fit_weak["phi_ci95"]
    ok &= check(w_hi - w_lo >= 0.5 and w_lo <= 0.5 <= w_hi,
                f"real budget: phi CI is wide and covers 0.50 "
                f"({w_lo:.2f}..{w_hi:.2f})")

    # 3c. unequal checkpoint spacing: the 5-point timeline samples the same
    #     process at half the 3-point interval.  Fitting half-steps as full
    #     steps reads the persistence as phi**0.5 (a time-unit artefact), so
    #     the same data must fit lower when `gaps` is supplied.
    def _ar1_gap(n, gap, phi, tau, rng, s2=0.01 ** 2):
        a = phi ** gap
        q = tau ** 2 * (1.0 - phi ** (2.0 * gap)) / (1.0 - phi ** 2)
        mu, ys = 0.0, []
        for _ in range(n):
            mu = a * mu + rng.gauss(0, math.sqrt(q))
            ys.append(mu + rng.gauss(0, math.sqrt(s2)))
        return ys

    rng = random.Random(35)
    with_gaps, no_gaps = [], []
    for i in range(8):
        ys = _ar1_gap(21, 0.5, 0.7, 0.03, rng)
        steps = [k * 10 for k in range(21)]
        with_gaps.append({"id": f"g{i}", "y": ys, "s2": [1e-4] * 21,
                          "gaps": gaps_from_steps(steps)})
        no_gaps.append({"id": f"g{i}", "y": ys, "s2": [1e-4] * 21})
    fit_g = fit_panel(with_gaps)
    fit_n = fit_panel(no_gaps)
    ok &= check(fit_g["phi"] <= 0.78 and
                fit_n["phi"] >= fit_g["phi"] + 0.08,
                f"half-step spacing: phi={fit_g['phi']:.2f} with gaps vs "
                f"{fit_n['phi']:.2f} ignoring them (true 0.70)")

    # 4. random-walk endpoint
    rng = random.Random(4)
    panel_rw = [_ar1(20, 1.0, 0.03, 0.0, rng, s2=0.01 ** 2) for _ in range(6)]
    for i, p in enumerate(panel_rw):
        p["id"] = f"s{i}"
    fit_rw = fit_panel(panel_rw)
    ok &= check(fit_rw["phi"] >= 0.8,
                f"RW endpoint: fitted phi {fit_rw['phi']:.2f} >= 0.80")

    # 5. tau -> 0 degeneracy
    flat = [{"id": f"s{i}", "y": [0.01] * 5, "s2": [0.028 ** 2] * 5} for i in range(6)]
    fit_flat = fit_panel(flat)
    ok &= check(fit_flat["degenerate"],
                f"constant series -> degenerate flag (tau0={fit_flat['tau0']:.2e})")

    # 6. finite horizon: monotone in H and never 0 by accident
    ps = [p_persistent(-0.06, 0.0005 ** 2, 0.8, 0.02, H, -0.02,
                       M=4000, rng=random.Random(6)) for H in (1, 2, 3)]
    ok &= check(ps[0] >= ps[1] >= ps[2] > 0.0,
                f"p_persistent monotone in H and non-zero: {['%.3f' % p for p in ps]}")

    # 7. persistent vs transient
    pp = p_persistent(-0.06, 0.0002, 0.95, 0.02, 2, -0.02, M=4000, rng=random.Random(7))
    tp = p_persistent(-0.06, 0.0002, 0.20, 0.02, 2, -0.02, M=4000, rng=random.Random(7))
    ok &= check(pp > 0.8 and tp < 0.3,
                f"phi=0.95 -> {pp:.2f} (>0.8), phi=0.20 -> {tp:.2f} (<0.3)")

    # 8. reversibility: a retired skill whose posterior recovers is readmitted
    rule = PosteriorRule(phi=0.5, tau=0.02, theta=-0.02, theta_keep=0.02, H=2,
                         M=3000)
    good = [{"step": i, "y": y, "N": 140, "s2": 0.0001}
            for i, y in enumerate([0.10, 0.15, 0.20])]
    ok &= check(rule.decide(good, status="retired") == READMIT,
                "retired + recovered posterior -> READMIT")
    # ... but not while the posterior is still clearly harmful (dead band)
    bad = [{"step": i, "y": y, "N": 140, "s2": 0.0001}
           for i, y in enumerate([-0.05, -0.06])]
    ok &= check(rule.decide(bad, status="retired") == RETIRE,
                "retired + still-harmful posterior -> stays RETIRE")

    # 8b. B2 must be able to leave trial: a skill that is negative from the
    #     first checkpoint and stays negative is retired after `patience`.
    b2 = SkillForgeStyle(delta_retire=-0.02, patience=2)
    always_bad = [{"step": i, "y": -0.05, "N": 140, "s2": 0.0001}
                  for i in range(4)]
    ok &= check(b2.decide(always_bad) == RETIRE,
                "B2: all-negative series -> RETIRE (trial can fail)")
    ok &= check(b2.decide(always_bad[:1]) == KEEP,
                "B2: one bad cycle < patience -> KEEP")
    ok &= check(b2.decide([]) == KEEP,
                "B2: unmeasured skill -> KEEP (never retire on no data)")

    # 9. VOI: ~0 when the decision is already settled, positive only near the
    #    decision boundary.  With c_FR=19 the loss is linear in p for
    #    p < 1-alpha, so information has value only where the posterior is
    #    close enough to the threshold for the observation to cross it.
    voi_rule = PosteriorVOIRule(phi=0.9, tau=0.03, theta=-0.02, H=2, M=400)
    clear = [{"step": i, "y": y, "N": 140, "s2": 0.0001}
             for i, y in enumerate([0.20, 0.21, 0.20])]
    far = [{"step": i, "y": y, "N": 140, "s2": 0.0001}
           for i, y in enumerate([-0.30, -0.30, -0.30])]
    edge = [{"step": i, "y": y, "N": 140, "s2": 0.0002}
            for i, y in enumerate([-0.08, -0.08, -0.08])]
    v_clear, v_far, v_edge = (voi_rule.voi(clear), voi_rule.voi(far),
                              voi_rule.voi(edge))
    ok &= check(v_clear < 0.01 and v_far < 0.01 and v_edge > 0.02,
                f"VOI clear={v_clear:.4f}, far={v_far:.4f} (<0.01), "
                f"boundary={v_edge:.4f} (>0.02)")

    # 10a. alpha <-> cost equivalence
    c_fr, c_mh = costs_from_alpha(0.05)
    ok &= check(abs(c_fr - 19.0) < 1e-9 and abs(c_mh - 1.0) < 1e-9,
                "alpha=0.05 <-> c_false_retire = 19 x c_missed_harm")
    ok &= check(abs(alpha_from_costs(19.0, 1.0) - 0.05) < 1e-12,
                "cost ratio 19:1 -> alpha=0.05")
    ok &= check(rule.decide(good) == KEEP,
                "the same series is NOT retired (decision rule is not reversed)")

    # 10b. baselines reproduce the section 3.4 table on the real timeline
    if TIMELINE.exists():
        tl = UtilityTimeline.load(str(TIMELINE))
        slim = SLIMStyle(theta=-0.02, patience=1)
        confirm = ConfirmRule(theta=-0.02, cycles=2)
        slim20, confirm40 = set(), set()
        for sid in tl.skill_ids():
            if tl.utility(sid, 20) < -0.02:
                slim20.add(sid)
            series = series_from_timeline(tl, sid)
            if confirm.decide(series) == RETIRE:
                confirm40.add(sid)
        ok &= check(slim20 == {"gen_002", "gen_003", "gen_004", "gen_005", "gen_006"},
                    f"single snapshot@20 retires {sorted(slim20)}")
        ok &= check(confirm40 == {"gen_002", "gen_003", "gen_006"},
                    f"confirm retires {sorted(confirm40)}")
    else:
        print("[skip] timeline not found; section 3.4 reproduction skipped")
        ok &= True

    print()
    print("ALL PASS" if ok else "SOME CHECKS FAILED")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
