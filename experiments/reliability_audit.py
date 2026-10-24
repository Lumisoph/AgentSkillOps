"""E0 + E1 for Stage 5.7: identifiability audit and false-retirement rates.

Modes
-----
``audit``       per-cell u/SE/z/CI, multiplicity, power curve, hierarchical
                AR(1) shrinkage (E0).
``bootstrap``   false-retirement rate under H0 (true u = 0) and detection rate
                under H1, for B1/B2/B3/B4 (E1-P1).
``nscan``       the same under n in {140, 280, 560, 1000} -- answers "why not
                just measure 1000 tasks?" (E1-P1b).
``calibration`` split-half calibration on the real paired CSVs: does the
                stated probability of harm match the held-out frequency?
                (E1-P2; accuracy comparison is not used because the half-sample
                SE is the same order as theta).
``all``         audit + bootstrap + nscan (calibration is opt-in, it needs CSVs).

Usage
-----
    python experiments/reliability_audit.py \
        --mode all --timeline results/stage55_measure/utility_store_timeline.json

    python experiments/reliability_audit.py \
        --mode calibration --measure-dir results/stage55_measure \
        --policies sft,ckpt20,ckpt40
"""

import argparse
import csv
import json
import math
import os
import random
import sys
from pathlib import Path
from statistics import NormalDist

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from freshskill import UtilityTimeline  # noqa: E402
from freshskill.measurement import (  # noqa: E402
    benjamini_hochberg,
    bonferroni,
    null_proportions,
    p_value_two_sided,
    required_n,
    standard_error,
    summarize,
)
from freshskill.retirement import (  # noqa: E402
    KEEP,
    RETIRE,
    ConfirmRule,
    PosteriorRule,
    SLIMStyle,
    SkillForgeStyle,
    series_from_timeline,
)
from freshskill.state_space import (  # noqa: E402
    fit_panel,
    gaps_from_steps,
    predictive,
)

ALPHA = 0.05
SD_U = 0.05   # stationary sd of the latent utility path in the correlated null


# ---------------------------------------------------------------- loading
def load_panel(timeline):
    panel = []
    for sid in timeline.skill_ids():
        series = series_from_timeline(timeline, sid)
        if not series:
            continue
        panel.append({
            "id": sid,
            "y": [e["y"] for e in series],
            "s2": [e["s2"] for e in series],
            "gaps": gaps_from_steps([e["step"] for e in series]),
        })
    return panel


def fit_model(panel, **kwargs):
    try:
        return fit_panel(panel, **kwargs)
    except Exception as exc:  # pragma: no cover - defensive
        print(f"[warn] model fit failed: {exc}", file=sys.stderr)
        return {"phi": 0.5, "phi_fitted": 0.5, "tau0": 0.02,
                "tau": {s["id"]: 0.02 for s in panel},
                "profile": [], "phi_ci95": (float("nan"), float("nan")),
                "degenerate": False, "fallback": True}


# ---------------------------------------------------------------- E0
def run_audit(timeline, out_dir):
    rows, pvals = [], []
    print("\n=== E0: per-cell utility, noise and significance ===")
    print(f"{'skill':9} {'t':>4} {'h':>3} {'m':>3} {'h-m':>4} {'u':>8} {'SE':>7} "
          f"{'z':>7} {'p':>7}  {'95% CI':>18}")
    for sid in timeline.skill_ids():
        for step in timeline.steps(sid):
            c = timeline.counts(sid, step)
            h, m = c["help"], c["harm"]
            neutral = c.get("neutral", c.get("usage", h + m) - h - m)
            s = summarize(h, m, neutral)
            pvals.append(s["p"])
            rows.append({"skill": sid, "step": step, "h": h, "m": m,
                         "neutral": neutral, "u": s["u"], "se": s["se"],
                         "z": s["z"], "p": s["p"], "ci95": list(s["ci95"])})
            print(f"{sid:9} {step:>4.0f} {h:>3} {m:>3} {h-m:>4} "
                  f"{s['u']:>8.4f} {s['se']:>7.4f} {s['z']:>7.2f} {s['p']:>7.4f} "
                  f"  [{s['ci95'][0]:+.4f}, {s['ci95'][1]:+.4f}]")

    thr, n_bonf = bonferroni(pvals, ALPHA)
    q, rejected = benjamini_hochberg(pvals, ALPHA)
    abs_z = [abs(r["z"]) for r in rows]
    print(f"\ncells={len(rows)}  mean|z|={sum(abs_z)/len(abs_z):.2f}  "
          f"frac|z|<1={sum(z < 1 for z in abs_z)/len(abs_z):.2f}  "
          f"max|z|={max(abs_z):.2f}")
    print(f"Bonferroni threshold p<={thr:.5f} -> {n_bonf} significant")
    print(f"Benjamini-Hochberg      -> {sum(rejected)} significant")

    print("\npower (alpha=0.05, power=0.8, activity=0.115):")
    for delta in (0.02, 0.03, 0.04, 0.05, 0.06):
        print(f"  delta={delta:.2f}  N={required_n(delta):>5d}")

    model = fit_model(load_panel(timeline))
    lo, hi = model["phi_ci95"]
    note = ""
    if model.get("degenerate"):
        note = (f"  [DEGENERATE: tau0 ~ 0; using phi=1, "
                f"fitted phi={model.get('phi_fitted', float('nan')):.2f}, "
                f"CI computed at the fitted phi]")
    print(f"\nhierarchical AR(1): phi={model['phi']:.2f}  "
          f"95% profile CI=[{lo:.2f}, {hi:.2f}]  tau0={model['tau0']:.4f}{note}")
    print("  per-skill tau (partial pooling): " +
          ", ".join(f"{k}={v:.4f}" for k, v in sorted(model["tau"].items())))

    result = {
        "cells": rows,
        "multiplicity": {"bonferroni_threshold": thr,
                         "bonferroni_significant": n_bonf,
                         "bh_significant": int(sum(rejected))},
        "power": {str(d): required_n(d) for d in (0.02, 0.03, 0.04, 0.05, 0.06)},
        "model": {"phi": model["phi"],
                  "phi_fitted": model.get("phi_fitted", model["phi"]),
                  "phi_ci95": list(model["phi_ci95"]),
                  "tau0": model["tau0"], "tau": model["tau"],
                  "degenerate": model["degenerate"],
                  "profile": model["profile"]},
    }
    _dump(out_dir, "e0_audit.json", result)
    return model


# ---------------------------------------------------------------- sampling
def _draw(rng, probs, N):
    """Draw (help, harm, neutral) from a multinomial with the given probs."""
    p_pos, p_neg, _p_zero = probs
    h = rng.binomialvariate(N, p_pos)
    # among the remaining (not-positive) trials, this is the negative share
    cond = p_neg / (1.0 - p_pos) if p_pos < 1.0 else 0.0
    m = rng.binomialvariate(N - h, cond) if cond > 0 else 0
    return h, m, N - h - m


def _shifted_probs(probs, delta):
    """Same activity rate, utility shifted to ``delta``."""
    active = probs[0] + probs[1]
    p_pos = max(0.0, (active + delta) / 2.0)
    p_neg = max(0.0, (active - delta) / 2.0)
    return (p_pos, p_neg, max(1.0 - p_pos - p_neg, 0.0))


def _draw_latent(rng, probs, N, u):
    """Draw (h, m, neutral) with a *given* latent utility u (within activity)."""
    active = probs[0] + probs[1]
    u = max(-active + 1e-6, min(active - 1e-6, u))
    p_pos = (active + u) / 2.0
    p_neg = (active - u) / 2.0
    h = rng.binomialvariate(N, p_pos)
    cond = p_neg / (1.0 - p_pos) if p_pos < 1.0 else 0.0
    m = rng.binomialvariate(N - h, cond) if cond > 0 else 0
    return h, m, N - h - m


def _series_from_counts(counts, steps):
    out = []
    for (h, m, neutral), step in zip(counts, steps):
        out.append({"step": step, "y": (h - m) / (h + m + neutral),
                    "N": h + m + neutral,
                    "s2": standard_error(h, m, neutral) ** 2})
    return out


def _make_rules(model, theta, alpha, H):
    """Threshold baselines.  B4 is handled separately (one tau_i per skill)."""
    return {
        "B1_slim_p1": SLIMStyle(theta=theta, patience=1),
        "B1_slim_p2": SLIMStyle(theta=theta, patience=2),
        "B2_skillforge": SkillForgeStyle(delta_retire=theta, patience=1),
        "B3_confirm": ConfirmRule(theta=theta, cycles=2),
    }


# At three checkpoints phi is not identified (E0 reports a CI spanning
# [0, 0.95]), so E1 must *assume* a persistence level instead of pretending the
# point estimate means anything.  The (phi, alpha) grid below makes the
# dependence explicit: this is precisely why E2 exists.
B4_SETTINGS = ((0.5, 0.05), (0.5, 0.30), (0.9, 0.05), (0.9, 0.30))
B4_TAU = 0.03
ALT_DELTAS = (-0.04, -0.08)


def _b4_name(phi, alpha):
    return f"B4_p{int(phi * 100):02d}_a{int(alpha * 100):02d}"


def _posterior_rules(model, theta, settings, H):
    """One posterior rule per (phi, alpha, skill)."""
    return {(p, a, sid): PosteriorRule(phi=p, tau=B4_TAU, alpha=a,
                                       theta=theta, H=H, M=600)
            for (p, a) in settings for sid in model["tau"]}


def _rule_names(rules):
    return list(rules) + [_b4_name(p, a) for p, a in B4_SETTINGS]


# ---------------------------------------------------------------- E1
def run_bootstrap(timeline, model, out_dir, reps=1000, theta=-0.02,
                  alpha=ALPHA, H=2, seed=0, rho=0.0, sd_u=0.0):
    steps_by = {}
    probs_by = {}
    # Only panel skills: a skill with no measured checkpoint is dropped by
    # load_panel (N<=0 is not an observation), so it has no tau_i and no
    # posterior rule.  Iterating timeline.skill_ids() here would KeyError.
    for sid in sorted(model["tau"]):
        steps = timeline.steps(sid)
        steps_by[sid] = steps
        c = timeline.counts(sid, steps[-1])
        h, m = c.get("help", 0), c.get("harm", 0)
        neutral = c.get("neutral", c.get("usage", h + m) - h - m)
        probs_by[sid] = null_proportions(h, m, neutral)

    rules = _make_rules(model, theta, alpha, H)
    posterior_rules = _posterior_rules(model, theta, B4_SETTINGS, H)

    all_names = _rule_names(rules)
    null = {name: 0.0 for name in all_names}
    flips = {name: 0 for name in all_names}
    alt = {name: {d: 0.0 for d in ALT_DELTAS} for name in all_names}

    rng = random.Random(seed)
    n_skills = len(probs_by)
    for _rep in range(reps):
        # Dynamic null (sd_u > 0): the skill's true utility fluctuates across
        # checkpoints around 0; rho is the share of that variance shared by
        # all checkpoints.  A retirement is *false* iff the skill is not
        # actually harmful at the decision point, i.e. u_T >= theta.  With
        # sd_u = 0 this collapses to the static null (u_t = 0 everywhere).
        null_series = {}
        u_last = {}
        for sid, p in probs_by.items():
            # sd_u == 0 is the static null: draw nothing, so this path consumes
            # exactly the same randomness as the original design and the two
            # runs are bit-identical.
            z_shared = rng.gauss(0.0, 1.0) if sd_u > 0.0 else 0.0
            counts, us = [], []
            for _ in steps_by[sid]:
                if sd_u > 0.0:
                    u = sd_u * (math.sqrt(rho) * z_shared +
                                math.sqrt(1.0 - rho) * rng.gauss(0.0, 1.0))
                else:
                    u = 0.0
                counts.append(_draw_latent(rng, p, 140, u))
                us.append(u)
            null_series[sid] = _series_from_counts(counts, steps_by[sid])
            u_last[sid] = us[-1]
        alt_series = {d: {sid: _series_from_counts(
            [_draw(rng, _shifted_probs(p, d), 140) for _ in steps_by[sid]],
            steps_by[sid])
            for sid, p in probs_by.items()} for d in ALT_DELTAS}

        for name, rule in rules.items():
            acts = []
            for sid in probs_by:
                act = rule.decide(null_series[sid])
                acts.append(act)
                if act == RETIRE and u_last[sid] >= theta:
                    null[name] += 1.0 / n_skills
            flips[name] += (len(set(acts)) > 1)
            for d in ALT_DELTAS:
                alt[name][d] += sum(
                    rule.decide(alt_series[d][sid]) == RETIRE
                    for sid in probs_by) / n_skills

        for (p, a) in B4_SETTINGS:
            name = _b4_name(p, a)
            acts = []
            for sid in probs_by:
                act = posterior_rules[(p, a, sid)].decide(null_series[sid])
                if act == RETIRE and u_last[sid] >= theta:
                    null[name] += 1.0 / n_skills
                acts.append(act)
                for d in ALT_DELTAS:
                    alt[name][d] += (posterior_rules[(p, a, sid)].decide(
                        alt_series[d][sid]) == RETIRE) / n_skills
            flips[name] += (len(set(acts)) > 1)

    print(f"\n=== E1-P1: bootstrap under a symmetric-activity null "
          f"({reps} reps, n=140, theta={theta}, alpha={alpha}, rho={rho}) ===")
    head = f"{'rule':16} {'FRR (null)':>10}" + "".join(
        f"{'detect u=%.2f' % d:>14}" for d in ALT_DELTAS) + f"{'disagree':>10}"
    print(head)
    out = {}
    for name in all_names:
        frr = null[name] / reps
        dets = {str(d): alt[name][d] / reps for d in ALT_DELTAS}
        # "disagree": the rule splits the six skills in a single replicate.
        # Named differently from E3's `flips` (which counts state changes over
        # T cycles) on purpose -- the two are not the same statistic.
        out[name] = {"frr": frr, "detect": dets,
                     "disagree": flips[name] / reps}
        print(f"{name:16} {frr:>10.3f}" + "".join(
            f"{dets[str(d)]:>14.3f}" for d in ALT_DELTAS) +
            f"{flips[name]/reps:>10.3f}")
    _dump(out_dir, "e1_bootstrap.json",
          {"reps": reps, "n": 140, "theta": theta, "alpha": alpha,
           "null": "symmetric-activity", "rho": rho, "sd_u": sd_u,
           "alt_deltas": list(ALT_DELTAS),
           "rules": out})
    return out


def run_nscan(timeline, model, out_dir, sizes=(140, 280, 560, 1000),
              reps=400, theta=-0.02, alpha=ALPHA, H=2, seed=1):
    steps_by = {}
    probs_by = {}
    # Only panel skills: a skill with no measured checkpoint is dropped by
    # load_panel (N<=0 is not an observation), so it has no tau_i and no
    # posterior rule.  Iterating timeline.skill_ids() here would KeyError.
    for sid in sorted(model["tau"]):
        steps = timeline.steps(sid)
        steps_by[sid] = steps
        c = timeline.counts(sid, steps[-1])
        h, m = c.get("help", 0), c.get("harm", 0)
        neutral = c.get("neutral", c.get("usage", h + m) - h - m)
        probs_by[sid] = null_proportions(h, m, neutral)

    rules = _make_rules(model, theta, alpha, H)
    posterior_rules = _posterior_rules(model, theta, B4_SETTINGS, H)
    names = _rule_names(rules)
    rng = random.Random(seed)
    table = {}
    for N in sizes:
        acc = {name: 0.0 for name in names}
        for _rep in range(reps):
            for sid, probs in probs_by.items():
                series = _series_from_counts(
                    [_draw(rng, probs, N) for _ in steps_by[sid]],
                    steps_by[sid])
                for name, rule in rules.items():
                    acc[name] += (rule.decide(series) == RETIRE) / (reps * len(probs_by))
                for (p, a) in B4_SETTINGS:
                    acc[_b4_name(p, a)] += (
                        posterior_rules[(p, a, sid)].decide(series) == RETIRE
                    ) / (reps * len(probs_by))
        table[N] = acc

    print(f"\n=== E1-P1b: false-retirement rate vs n (H0, {reps} reps) ===")
    header = "rule".ljust(16) + "".join(f"n={n}".rjust(10) for n in sizes)
    print(header)
    for name in names:
        print(name.ljust(16) + "".join(f"{table[n][name]:.3f}".rjust(10) for n in sizes))
    _dump(out_dir, "e1_nscan.json",
          {"reps": reps, "theta": theta, "alpha": alpha, "sizes": list(sizes),
           "table": {str(N): v for N, v in table.items()}})


# ---------------------------------------------------------------- E1-P2
def _read_won(path):
    with open(path) as f:
        return {int(r["seed"]): int(r["won"]) for r in csv.DictReader(f)}


def _paired_counts(base, loo, seeds):
    """Paired help/harm/neutral counts over the seeds *present in both* files.

    Seeds missing from either file are dropped, not counted as neutral: a
    missing run is missing data, not evidence that the skill did nothing.
    """
    h = m = present = 0
    for s in seeds:
        if s not in base or s not in loo:
            continue
        present += 1
        d = base[s] - loo[s]
        if d > 0:
            h += 1
        elif d < 0:
            m += 1
    return h, m, present - h - m


def run_calibration(measure_dir, policies, skills, out_dir, splits=200,
                    theta=-0.02, seed=2, phi_override=None):
    measure_dir = Path(measure_dir)
    data = {}
    for policy in policies:
        base_path = measure_dir / f"base_{policy}.csv"
        if not base_path.exists():
            print(f"[skip] missing {base_path}", file=sys.stderr)
            continue
        base = _read_won(base_path)
        for skill in skills:
            loo_path = measure_dir / f"loo_{policy}_{skill}.csv"
            if not loo_path.exists():
                print(f"[skip] missing {loo_path}", file=sys.stderr)
                continue
            data[(policy, skill)] = (base, _read_won(loo_path))
    if not data:
        print("[error] no paired CSVs found", file=sys.stderr)
        return

    seeds = sorted(next(iter(data.values()))[0])
    rng = random.Random(seed)
    records = []
    Nd = NormalDist()
    for _ in range(splits):
        rng.shuffle(seeds)
        half = len(seeds) // 2
        A, B = set(seeds[:half]), set(seeds[half:])

        # Fit on every checkpoint *except the last*, and score both methods on
        # the same out-of-sample event: "is the last checkpoint's utility (on
        # the held-out task half B) below theta?".  Using the last checkpoint
        # in the fit and then predicting it (the previous version) lets the
        # Bayes path peek at the target and makes the comparison meaningless.
        panel_A, targets = [], {}
        for skill in skills:
            ys, s2s, used = [], [], []
            for policy in policies[:-1]:
                if (policy, skill) not in data:
                    continue
                base, loo = data[(policy, skill)]
                h, m, neutral = _paired_counts(base, loo, A)
                if h + m + neutral == 0:
                    continue
                ys.append((h - m) / (h + m + neutral))
                s2s.append(standard_error(h, m, neutral) ** 2)
                used.append(policy)
            if len(ys) < 2:  # the filter needs at least one transition
                continue
            if (policies[-1], skill) not in data:  # no held-out target
                continue
            panel_A.append({"id": skill, "y": ys, "s2": s2s, "used": used})
            targets[skill] = policies[-1]
        if not panel_A:
            continue
        if phi_override is None:
            model = fit_model(panel_A)
        else:
            # Sensitivity: phi is unidentified on two checkpoints (the fit is
            # degenerate in ~45% of splits), so the Bayes calibration number
            # must be shown to survive a fixed persistence assumption.
            model = fit_model(panel_A, phis=[phi_override])
            model["phi"] = phi_override

        for p in panel_A:
            sid = p["id"]
            rule = PosteriorRule(phi=model["phi"], tau=model["tau"][sid],
                                 alpha=ALPHA, theta=theta, H=2)
            series_A = [{"step": i, "y": y, "N": half, "s2": s2}
                        for i, (y, s2) in enumerate(zip(p["y"], p["s2"]))]

            # plugin: last observed deviation, no model
            se_A = math.sqrt(p["s2"][-1])
            if se_A > 0:
                p_plugin = Nd.cdf((theta - p["y"][-1]) / se_A)
            else:
                p_plugin = 1.0 if p["y"][-1] < theta else 0.0

            # bayes: one-step predictive from the A-half posterior; the
            # observation variance of the target checkpoint is proxied by the
            # previous checkpoint's s2 (same task count).
            mu, var = rule.posterior(series_A)
            mean, pv = predictive(mu, var, rule.phi, rule.tau, p["s2"][-1])
            p_bayes = Nd.cdf((theta - mean) / math.sqrt(pv))

            policy = targets[sid]
            base, loo = data[(policy, sid)]
            h, m, neutral = _paired_counts(base, loo, B)
            n_B = h + m + neutral
            if n_B == 0:
                continue
            y_B = (h - m) / n_B
            outcome = 1 if y_B < theta else 0

            records.append({"skill": sid, "policy": policy,
                            "p_plugin": p_plugin, "p_bayes": p_bayes,
                            "outcome": outcome, "y_A": p["y"][-1], "y_B": y_B})

    def ece(probs, obs, bins=5):
        total, err = len(probs), 0.0
        for b in range(bins):
            lo, hi = b / bins, (b + 1) / bins
            idx = [i for i, p in enumerate(probs) if lo <= p < hi or (b == bins - 1 and p == 1.0)]
            if not idx:
                continue
            conf = sum(probs[i] for i in idx) / len(idx)
            acc = sum(obs[i] for i in idx) / len(idx)
            err += len(idx) / total * abs(conf - acc)
        return err

    if not records:
        print("[error] no calibration records (need >= 3 policies)", file=sys.stderr)
        return
    pp = [r["p_plugin"] for r in records]
    pb = [r["p_bayes"] for r in records]
    yy = [r["outcome"] for r in records]
    print(f"\n=== E1-P2: split-half calibration ({len(records)} decisions) ===")
    print(f"target: {policies[-1]} utility on held-out tasks; "
          f"both methods see {policies[0]}..{policies[-2]} only")
    print(f"{'method':10} {'ECE':>7} {'mean p':>8} {'observed harm rate':>20}")
    print(f"{'plugin':10} {ece(pp, yy):>7.3f} {sum(pp)/len(pp):>8.3f} {sum(yy)/len(yy):>20.3f}")
    print(f"{'bayes':10} {ece(pb, yy):>7.3f} {sum(pb)/len(pb):>8.3f} {sum(yy)/len(yy):>20.3f}")
    _dump(out_dir, "e1_calibration.json",
          {"splits": splits, "theta": theta,
           "target_policy": policies[-1],
           "fit_policies": list(policies[:-1]),
           "phi_override": phi_override,
           "ece_plugin": ece(pp, yy), "ece_bayes": ece(pb, yy),
           "mean_p_plugin": sum(pp) / len(pp), "mean_p_bayes": sum(pb) / len(pb),
           "observed_harm_rate": sum(yy) / len(yy), "records": records})


# ---------------------------------------------------------------- utils
def _dump(out_dir, name, obj):
    if not out_dir:
        return
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, name), "w") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="audit",
                    choices=["audit", "bootstrap", "nscan", "calibration", "all"])
    ap.add_argument("--timeline",
                    default="results/stage55_measure/utility_store_timeline.json")
    ap.add_argument("--out", default="results/stage57_reliability")
    ap.add_argument("--reps", type=int, default=1000)
    ap.add_argument("--theta", type=float, default=-0.02)
    ap.add_argument("--alpha", type=float, default=ALPHA)
    ap.add_argument("--H", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--measure-dir", default="results/stage55_measure")
    ap.add_argument("--policies", default="sft,ckpt20,ckpt40")
    ap.add_argument("--skills", default="gen_001,gen_002,gen_003,gen_004,gen_005,gen_006")
    ap.add_argument("--splits", type=int, default=200)
    ap.add_argument("--rho", type=float, default=0.0,
                    help="latent correlation across a skill's checkpoints in "
                         "the dynamic null (needs --sd-u > 0)")
    ap.add_argument("--sd-u", type=float, default=0.0,
                    help="sd of the latent utility path in the null; 0 = the "
                         "static null (u_t = 0, the earlier design)")
    ap.add_argument("--phi-override", type=float, default=None,
                    help="fix phi in the calibration (sensitivity check); "
                         "default: fit it, which is unidentified on 2 points")
    args = ap.parse_args()

    timeline = UtilityTimeline.load(args.timeline)
    model = None
    if args.mode in ("audit", "all", "bootstrap", "nscan"):
        model = run_audit(timeline, args.out) if args.mode in ("audit", "all") \
            else fit_model(load_panel(timeline))
    if args.mode in ("bootstrap", "all"):
        run_bootstrap(timeline, model, args.out, reps=args.reps,
                      theta=args.theta, alpha=args.alpha, H=args.H,
                      seed=args.seed, rho=args.rho, sd_u=args.sd_u)
    if args.mode in ("nscan", "all"):
        run_nscan(timeline, model, args.out, theta=args.theta,
                  alpha=args.alpha, H=args.H, seed=args.seed + 1)
    if args.mode in ("calibration", "all"):
        run_calibration(args.measure_dir, args.policies.split(","),
                        args.skills.split(","), args.out,
                        splits=args.splits, theta=args.theta,
                        phi_override=args.phi_override)


if __name__ == "__main__":
    main()
