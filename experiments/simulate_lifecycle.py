"""E3: retire / keep / readmit under three generators with known ground truth.

Stage 5.7 / E3.  E1 only says how often a rule fires on noise; E3 says whether
the *right* skill is retired, and what it costs in measurements.  Three
generators, because a rule that only wins under the model it assumes has not
been validated:

    (a) ar1_*    mu_i(t) = phi mu_i(t-1) + eta, eta ~ N(0, tau^2)   [matched]
    (b) step     mu = 0, then at a random cycle drops to -delta and stays
                 (SkillForge's delayed obsolescence)
    (c) mixture  mostly 0, occasional harmful spells of 2-3 cycles
                 (transient harm -- the case B4 is supposed to survive)

Rules: B1 SLIMStyle, B2 SkillForgeStyle, B3 ConfirmRule, B4 PosteriorRule
(two persistence assumptions, so the effect of a wrong phi is visible), and
B5 = B4 + a measurement-allocation score, reported only at the reference
slice because allocation only changes the *cost* axis.

B4/B5 are *reversible*: a retired skill is re-measured every second cycle
(``probe_every``) and readmitted when P(utility > theta) clears the
cost-derived bar (see ``freshskill.retirement``).  Probing is what makes the
reversal testable -- a retired skill that is never measured can never be
shown to have recovered.  Probe episodes are counted in ``meas`` and
separately in ``probes`` so the price of reversibility is visible.

Metrics per replicate (6 skills, T decision cycles):
    FRR        fraction of skills retired while their true mu >= 0
    missed     fraction of skills *persistently* harmful over the H-cycle
               window after the last decision and still in the bank.  A skill
               that is harmful for one cycle of a transient spell is NOT
               missed -- retiring it would be the error.
    flips      mean number of active<->retired changes per skill
    meas       total episodes spent (skills x cycles measured x n), including
               probes of retired skills
    probes     number of probe measurements (subset of ``meas``)
    bank       sum of the true mu of skills left active, averaged over that
               same forward window (higher is better)

``bank`` is the outcome that matters: a false retirement removes a useful
skill, a missed retirement keeps a harmful one.

Usage:
    python experiments/simulate_lifecycle.py --reps 50
    python experiments/simulate_lifecycle.py --reps 20 --gens ar1_0.5,step
"""

import argparse
import json
import math
import os
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from freshskill.retirement import (  # noqa: E402
    KEEP,
    READMIT,
    RETIRE,
    ConfirmRule,
    PosteriorRule,
    SLIMStyle,
    SkillForgeStyle,
)

ACTIVITY = 0.115     # help+harm share of tasks (E0)
TAU_TRUE = 0.03


# ---------------------------------------------------------------- generators
def _stationary_sd(sd, phi):
    return sd * math.sqrt(max(1.0 - phi ** 2, 1e-9))


def gen_ar1(rng, n_skills, T, phi, sd=0.05):
    tau = _stationary_sd(sd, phi)
    out = []
    for _ in range(n_skills):
        mu = rng.gauss(0.0, sd)
        path = []
        for _t in range(T):
            mu = phi * mu + (rng.gauss(0.0, tau) if tau > 0 else 0.0)
            path.append(mu)
        out.append(path)
    return out


def gen_step(rng, n_skills, T, delta=0.05, t0_range=(2, 6)):
    out = []
    for _ in range(n_skills):
        t0 = rng.randint(*t0_range)
        out.append([0.0 if t < t0 else -delta for t in range(T)])
    return out


def gen_mixture(rng, n_skills, T, delta=0.05, p_enter=0.12, spell=(2, 3)):
    out = []
    for _ in range(n_skills):
        path = []
        in_spell = False
        while len(path) < T:
            if not in_spell and rng.random() < p_enter:
                path.extend([-delta] * rng.randint(*spell))
                in_spell = True
            else:
                # at least one benign cycle between spells, otherwise two
                # "occasional" spells merge into one long one
                path.append(0.0)
                in_spell = False
        out.append(path[:T])
    return out


GENERATORS = {
    "ar1_0.5": lambda rng, k, T: gen_ar1(rng, k, T, 0.5),
    "ar1_0.9": lambda rng, k, T: gen_ar1(rng, k, T, 0.9),
    "step": gen_step,
    "mixture": gen_mixture,
}


# ---------------------------------------------------------------- measurement
def measure(rng, mu, n):
    """Draw (h, m, neutral) at activity ACTIVITY and return the rule entry."""
    a = ACTIVITY
    u = max(-a + 1e-6, min(a - 1e-6, mu))
    p_pos = (a + u) / 2.0
    p_neg = (a - u) / 2.0
    h = rng.binomialvariate(n, p_pos)
    cond = p_neg / (1.0 - p_pos) if p_pos < 1.0 else 0.0
    m = rng.binomialvariate(n - h, cond) if cond > 0 else 0
    neutral = n - h - m
    y = (h - m) / n
    # plug-in variance of the paired mean: [(h+m)/N - u^2]/N
    s2 = max(((h + m) / n - y * y) / n, 1e-8)
    return {"y": y, "N": n, "s2": s2}


# ---------------------------------------------------------------- rules
def threshold_rules(theta, patience):
    return {
        f"B1_p{patience}": SLIMStyle(theta=theta, patience=patience),
        f"B2_p{patience}": SkillForgeStyle(delta_retire=theta, patience=patience),
    }


def posterior_rule(phi, theta, H, M=150, alpha=0.05, tau=TAU_TRUE):
    return PosteriorRule(phi=phi, tau=tau, alpha=alpha, theta=theta, H=H,
                         M=M)


def _rule_tau(gen_name, phi):
    """Innovation sd the rule should use in a generator's world.

    ``gen_ar1`` keeps the stationary sd at 0.05, so its innovation sd is
    ``0.05 * sqrt(1 - phi^2)``.  Using a fixed tau=0.03 made the "matched"
    AR(1) rows matched in phi but *not* in tau, so the phi=0.5 vs phi=0.9
    comparison mixed a phi error with a tau error.  Step/mixture have no true
    tau (the truth is piecewise constant), so the modelling default stays.
    """
    if gen_name.startswith("ar1"):
        return _stationary_sd(0.05, phi)
    return TAU_TRUE


class OracleMu:
    """Perfect measurement, one-cycle threshold -- the limit of B1 as noise -> 0.

    It is the upper bound for *detection*, not for *value*: with no memory it
    retires a skill that dips for one cycle and then recovers.
    """

    def __init__(self, theta):
        self.theta = theta

    def decide(self, series=None, status="active", truth=None, future=None):
        if truth is None or truth >= self.theta:
            return KEEP
        return RETIRE


class OraclePersistent:
    """Perfect measurement + the persistence criterion -- the ceiling of B4.

    Retires iff the true utility is below theta now *and* for the next H
    cycles.  The gap between this and B4 is the price of not knowing the state.
    """

    def __init__(self, theta, H):
        self.theta = theta
        self.H = H

    def decide(self, series=None, status="active", truth=None, future=None):
        if truth is None or truth >= self.theta:
            return KEEP
        if future is None:
            return KEEP
        return RETIRE if all(v < self.theta for v in future) else KEEP


# ---------------------------------------------------------------- simulation
def run_rule(rule, truth, n, rng, theta, T, is_oracle=False,
             budget=None, H=2, probe_every=None):
    """One replicate: returns metrics for one rule on one set of true paths.

    The decision at cycle ``t`` protects ``truth[t+1 : t+1+H]`` (the H cycles
    that follow it), and the metrics use ``truth[T : T+H]`` -- the same window
    the last decision (``t = T-1``) was protecting.  ``truth`` must therefore
    have ``T + H`` entries.  An earlier version scored ``truth[T-1 : ...]``,
    which is one cycle *before* the window the rule actually decides on.
    """
    k = len(truth)
    series = [[] for _ in range(k)]
    status = ["active"] * k
    false_retire = missed = flips = 0
    falsely_retired = set()   # per-skill: FRR is a fraction of *skills*
    meas = probes = 0
    # Cumulative exposure: sum of the true utility of every skill that is in
    # the bank at the start of each cycle.  `bank` only scores the final
    # window, so a rule that retires a harmful skill K cycles late looks as
    # good as one that never let it in.  This is the quantity the c_MH term of
    # the cost model actually charges for (higher is better).  The decision at
    # t is taken after cycle t is observed, so retirement takes effect at t+1.
    cum = 0.0
    # Realised welfare costs, in SR*cycle units.  A skill that is out of the
    # bank while its true utility is fine loses that utility every cycle
    # (``lost``); a skill that stays in the bank while truly harmful does
    # damage every cycle (``harm``).  These two numbers ARE the c_FR and c_MH
    # sides of the cost model, measured instead of assumed, so alpha can be
    # calibrated by minimising their sum rather than by picking a ratio.
    lost = harm = 0.0

    for t in range(T):
        cum += sum(truth[i][t] for i in range(k) if status[i] == "active")
        for i in range(k):
            if status[i] == "retired":
                if truth[i][t] >= theta:
                    lost += truth[i][t]
            elif truth[i][t] < theta:
                harm += -truth[i][t]
        active = [i for i in range(k) if status[i] == "active"]
        if is_oracle:
            chosen = set()                      # the oracle measures nothing
        elif budget is not None and len(active) > budget:
            # Allocation: measure the skills whose persistence probability is
            # closest to the decision boundary 1-alpha -- an observation can
            # only change the decision there.  ``p*(1-p)`` (the previous
            # surrogate) peaks at 1/2 and ignores alpha entirely: at
            # alpha=0.05 the boundary is 0.95, so it would never buy the
            # marginal skills it exists to resolve.  This is still a
            # one-dimensional surrogate for ``PosteriorVOIRule.voi``, not the
            # exact VOI (which also weighs how far past the boundary the
            # posterior is), but it now points at the right place.
            boundary = 1.0 - rule.alpha
            scored = []
            for i in active:
                if len(series[i]) < 2:
                    scored.append((1.0, i))       # unknown -> always measure
                    continue
                p = rule.persistence_probability(series[i])
                scored.append((abs(p - boundary), i))
            # deterministic tie-break: distance ascending, skill id ascending
            scored.sort(key=lambda sp: (sp[0], sp[1]))
            chosen = {i for _, i in scored[:budget]}
        else:
            chosen = set(active)

        # reversibility requires evidence: probe retired skills periodically
        if probe_every and not is_oracle:
            for i in range(k):
                if status[i] == "retired" and t % probe_every == 0:
                    series[i].append(measure(rng, truth[i][t], n))
                    meas += n
                    probes += 1

        for i in range(k):
            if status[i] != "active":
                continue
            if i in chosen:
                series[i].append(measure(rng, truth[i][t], n))
                meas += n

        if is_oracle:
            acts = [rule.decide(truth=truth[i][t],
                                future=truth[i][t + 1:t + 1 + H])
                    for i in range(k)]
        else:
            # a skill that was never measured has no evidence against it
            acts = [rule.decide(series[i], status=status[i]) if series[i]
                    else (RETIRE if status[i] == "retired" else KEEP)
                    for i in range(k)]

        for i in range(k):
            if status[i] == "retired":
                if acts[i] == READMIT:
                    status[i] = "active"
                    flips += 1
            elif acts[i] == RETIRE:
                status[i] = "retired"
                flips += 1
                if truth[i][t] >= 0.0:
                    # A reversible rule can retire the same skill twice; the
                    # reported FRR is still "fraction of skills", so count the
                    # skill once.  The event count is kept for diagnostics.
                    false_retire += 1
                    falsely_retired.add(i)

    # Forward window the last decision (t = T-1) was protecting.
    window = truth[0][T:T + H]
    assert len(window) == H, "truth paths must have T+H cycles"
    for i in range(k):
        if status[i] == "retired":
            continue
        fut = truth[i][T:T + H]
        if all(v < theta for v in fut):
            missed += 1
    bank = sum(
        (sum(truth[i][T:T + H]) / H)
        for i in range(k) if status[i] != "retired")
    return {"frr": len(falsely_retired) / k,
            "frr_events": false_retire / k,
            "missed": missed / k, "flips": flips / k,
            "meas": meas, "probes": probes, "bank": bank,
            "cum": cum / k, "lost": lost / k, "harm": harm / k}


PROBE_EVERY = 2
_MATCHED_PHI = {"ar1_0.5": 0.5, "ar1_0.9": 0.9}  # retired skills are re-measured every second cycle


def _stream(seed, name):
    """Per-rule RNG keyed by the rule's *name*, not by its table position.

    Positional keying made the same ``B4_phi90_a30`` row differ between the
    main table and the B5 reference table; a stable name makes every row
    reproducible on its own.
    """
    return random.Random(f"{seed}:{name}")


def _run_named(paths, rule, name, n, theta, T, seed, **kw):
    rrng = _stream(seed, name)
    agg = {"frr": 0.0, "frr_events": 0.0, "missed": 0.0, "flips": 0.0,
           "meas": 0.0, "probes": 0.0, "bank": 0.0, "cum": 0.0,
           "lost": 0.0, "harm": 0.0}
    for truth in paths:
        m = run_rule(rule, truth, n, rrng, theta, T, **kw)
        for key in agg:
            agg[key] += m[key] / len(paths)
    return agg


def evaluate(gen_name, reps, n, theta, T, seed, patience_list=(1, 2, 3),
             phi_list=(0.5, 0.9), alpha_list=(0.05, 0.30), H=2, M=150):
    rng = random.Random(seed)
    # T decisions plus H future cycles for the forward-window metrics
    paths = [GENERATORS[gen_name](rng, 6, T + H) for _ in range(reps)]

    results = {}

    # threshold rules: mean over the 6 skills and over replicates
    for patience in patience_list:
        for name, rule in threshold_rules(theta, patience).items():
            results[name] = _run_named(paths, rule, name, n, theta, T, seed)

    # B4 under two persistence assumptions x two cost ratios, plus oracle.
    # alpha is the false-retirement share of cost: alpha=0.05 means a false
    # retirement is 19x costlier than a missed harm; 0.30 means 2.33x.
    for phi in phi_list:
        for alpha in alpha_list:
            name = f"B4_phi{int(phi*100)}_a{int(alpha*100)}"
            rule = posterior_rule(phi, theta, H, M, alpha=alpha,
                                  tau=_rule_tau(gen_name, phi))
            results[name] = _run_named(paths, rule, name, n, theta, T, seed,
                                       probe_every=PROBE_EVERY)

    results["oracle_mu"] = _run_named(paths, OracleMu(theta), "oracle_mu",
                                      n, theta, T, seed, is_oracle=True, H=H)
    results["oracle_persist"] = _run_named(
        paths, OraclePersistent(theta, H), "oracle_persist", n, theta, T, seed,
        is_oracle=True, H=H)
    return results, paths


def evaluate_b5(paths, n, theta, T, seed, budget=3, phi=0.9, alpha=0.30, H=2,
                M=150, tau=TAU_TRUE):
    rule = posterior_rule(phi, theta, H, M, alpha=alpha, tau=tau)
    name = f"B5_b{budget}_phi{int(phi*100)}_a{int(alpha*100)}"
    return _run_named(paths, rule, name, n, theta, T, seed, budget=budget,
                      probe_every=PROBE_EVERY)


# ---------------------------------------------------------------- reporting
def _fmt(row):
    return (f"{row['frr']:>7.3f} {row['missed']:>7.3f} {row['flips']:>6.2f} "
            f"{row['meas']:>8.0f} {row.get('probes', 0):>7.0f} {row['bank']:>7.3f} "
            f"{row.get('cum', 0.0):>7.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gens", default="ar1_0.5,ar1_0.9,step,mixture")
    ap.add_argument("--ns", default="70,140,280,560")
    ap.add_argument("--thetas", default="-0.01,-0.02,-0.03")
    ap.add_argument("--patiences", default="1,2,3")
    ap.add_argument("--reps", type=int, default=50)
    ap.add_argument("--T", type=int, default=10)
    ap.add_argument("--H", type=int, default=2)
    ap.add_argument("--M", type=int, default=150)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results/stage57_e3")
    ap.add_argument("--alphas", default="0.05,0.1,0.2,0.3,0.5,0.7",
                    help="alpha grid for the cost calibration block")
    ap.add_argument("--calib-phi", type=float, default=-1.0,
                    help="force the calibration rule's phi (default: match "
                         "the generator)")
    ap.add_argument("--calib-only", action="store_true",
                    help="only run the alpha calibration block")
    ap.add_argument("--b5-only", action="store_true",
                    help="skip the main rule table and rerun only the B5 "
                         "allocation comparison")
    args = ap.parse_args()

    gens = args.gens.split(",")
    ns = [int(x) for x in args.ns.split(",")]
    thetas = [float(x) for x in args.thetas.split(",")]
    patiences = tuple(int(x) for x in args.patiences.split(","))

    table = {}
    print(f"E3: {args.reps} reps x 6 skills x T={args.T} cycles")
    if not (args.b5_only or args.calib_only):
        for gen in gens:
            for n in ns:
                for theta in thetas:
                    res, paths = evaluate(gen, args.reps, n, theta, args.T,
                                          seed=args.seed,
                                          patience_list=patiences,
                                          H=args.H, M=args.M)
                    table[f"{gen}|n{n}|th{theta}"] = res
                    print(f"\n--- {gen}  n={n}  theta={theta} ---")
                    print(f"{'rule':12} {'FRR':>7} {'missed':>7} {'flips':>6} "
                          f"{'meas':>8} {'probes':>7} {'bank':>7} "
                          f"{'cum':>7}")
                    for name, row in res.items():
                        print(f"{name:12} {_fmt(row)}")

    if args.calib_only:
        gens_for_b5 = []
    else:
        gens_for_b5 = gens
    # B5 allocation at the reference slice only (it changes the cost axis).
    # The comparator is the same B4 (phi=0.9, alpha=0.30) measuring everything.
    print(f"\n=== B5 (allocation, budget=3 of 6) at n=140 theta=-0.02, alpha=0.30 ===")
    print(f"{'rule':18} {'FRR':>7} {'missed':>7} {'flips':>6} "
          f"{'meas':>8} {'probes':>7} {'bank':>7} {'cum':>7}")
    for gen in gens_for_b5:
        # Same truth paths as the main table (random.Random(seed) does not
        # depend on the generator name) and the same name-keyed measurement
        # stream for the B4 reference, so the B4 row is unchanged.
        rng = random.Random(args.seed)
        paths = [GENERATORS[gen](rng, 6, args.T + args.H)
                 for _ in range(args.reps)]
        tau_ref = _rule_tau(gen, 0.9)
        rule_ref = posterior_rule(0.9, -0.02, args.H, args.M, alpha=0.30,
                                  tau=tau_ref)
        ref = {"B4_phi90_a30": _run_named(
            paths, rule_ref, "B4_phi90_a30", 140, -0.02, args.T,
            args.seed, probe_every=PROBE_EVERY)}
        b5 = evaluate_b5(paths, 140, -0.02, args.T, seed=args.seed,
                         H=args.H, M=args.M, alpha=0.30, tau=tau_ref)
        table[f"{gen}|n140|th-0.02|B5_budget3"] = b5
        print(f"{'B4_phi90_a30_' + gen:18} {_fmt(ref['B4_phi90_a30'])}")
        print(f"{'B5_' + gen:18} {_fmt(b5)}")

    # ---- alpha calibration: measure both sides of the cost model ----------
    # E3's main table fixes alpha (0.05 / 0.30) and therefore *assumes* the
    # cost ratio.  Here alpha is swept and the two costs are measured in the
    # same units (SR*cycle): `lost` = benefit given up by retiring a skill
    # that was fine, `harm` = damage taken while a harmful skill stayed in.
    # The alpha that minimises lost+harm is what the mechanism's own dynamics
    # imply -- no 19:1 assumption.
    alphas = [float(a) for a in args.alphas.split(",") if a.strip()]
    print(f"(rule phi: " + ", ".join(
        f"{g}={args.calib_phi if args.calib_phi >= 0 else _MATCHED_PHI.get(g, 0.9)}"
        for g in gens) + ")")
    print(f"\n=== alpha calibration (B4, matched phi, n=140, theta=-0.02) ===")
    print(f"{'gen':10} {'alpha':>6} {'FRR':>7} {'missed':>7} "
          f"{'lost':>8} {'harm':>8} {'total':>8}")
    calib = {}
    for gen in gens:
        rng = random.Random(args.seed)
        paths = [GENERATORS[gen](rng, 6, args.T + args.H)
                 for _ in range(args.reps)]
        rows = {}
        # Match the rule's persistence to the generator's by default: alpha*
        # is only interpretable as a *calibration* when the model is the one
        # that fits the world.  --calib-phi forces a fixed (mis-specified)
        # value, e.g. 0.9 against the phi=0.5 world.
        phi_rule = (args.calib_phi if args.calib_phi >= 0
                    else _MATCHED_PHI.get(gen, 0.9))
        for alpha in alphas:
            name = (f"B4_phi{int(round(phi_rule * 100))}"
                    f"_a{int(round(alpha * 100))}")
            rule = posterior_rule(phi_rule, -0.02, args.H, args.M,
                                  alpha=alpha,
                                  tau=_rule_tau(gen, phi_rule))
            r = _run_named(paths, rule, name, 140, -0.02, args.T, args.seed,
                           probe_every=PROBE_EVERY)
            rows[alpha] = r
            total = r["lost"] + r["harm"]
            print(f"{gen:10} {alpha:>6.2f} {r['frr']:>7.3f} {r['missed']:>7.3f} "
                  f"{r['lost']:>8.3f} {r['harm']:>8.3f} {total:>8.3f}")
        best = min(rows, key=lambda a: rows[a]["lost"] + rows[a]["harm"])
        calib[gen] = {"rows": {str(a): rows[a] for a in rows},
                      "best_alpha": best,
                      "best_total": rows[best]["lost"] + rows[best]["harm"]}
        print(f"{gen:10} {'best':>6} {best:>6.2f}  -> total "
              f"{rows[best]['lost'] + rows[best]['harm']:.3f}")
    table["alpha_calibration"] = calib

    if args.out:
        os.makedirs(args.out, exist_ok=True)
        path = os.path.join(args.out, "e3_simulation.json")
        with open(path, "w") as f:
            json.dump({"reps": args.reps, "T": args.T, "H": args.H,
                       "M": args.M, "gens": gens, "ns": ns,
                       "thetas": thetas, "patiences": list(patiences),
                       "results": table}, f, indent=2, ensure_ascii=False)
        print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
