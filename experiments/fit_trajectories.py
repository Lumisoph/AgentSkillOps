"""E2: fit the hierarchical AR(1) on the measured utility trajectories.

Stage 5.7 / E2 (experiment_design.md v3.1 section 5, implementation doc
section "E2").  Answers RQ-B directly:

    phi -> 0   harm is transient  (no case for permanent retirement)
    phi -> 1   harm persists     (retirement is justified)

The point of adding t=10/t=30 is *identifiability*: at three checkpoints the
95% profile interval for phi spans [0, 0.95] (E0), which is why E1 has to
evaluate B4 on an assumed grid.  This script reports the interval before and
after the two extra points, and refuses to over-claim when it stays wide.

Outputs
-------
- fitted phi, 95% profile CI, tau0, per-skill tau (partial pooling)
- smoothed per-skill trajectories (what the filter actually believes)
- p_persistent / p_recovery for each skill at the last checkpoint, under both
  the fitted model and the E1 assumption grid (rho in {0.5, 0.9})
- an explicit "separable / not separable" verdict

Usage:
    python experiments/fit_trajectories.py \
        --timeline 3pt=results/stage55_measure/utility_store_timeline.json \
        --timeline 5pt=results/stage55_measure/utility_store_timeline_5pt.json \
        --out results/stage57_e2
"""

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from freshskill import UtilityTimeline  # noqa: E402
from freshskill.retirement import series_from_timeline  # noqa: E402
from freshskill.state_space import (  # noqa: E402
    fit_panel,
    gaps_from_steps,
    p_persistent,
    p_recovery,
    run_series,
)

# p_persistent does not depend on alpha (alpha only sets the decision
# threshold), so the conditional table is keyed by phi alone.
PHI_GRID = (0.5, 0.9)
THETA_KEEP = 0.02


def build_panel(timeline):
    panel = []
    for sid in timeline.skill_ids():
        s = series_from_timeline(timeline, sid)
        if s:
            panel.append({"id": sid,
                          "y": [e["y"] for e in s],
                          "s2": [e["s2"] for e in s],
                          "gaps": gaps_from_steps([e["step"] for e in s])})
    return panel


def separability(ci):
    """Can the fitted interval separate transient (<=0.5) from persistent (>=0.9)?"""
    lo, hi = ci
    if lo != lo or hi != hi:
        return "unknown"
    if hi <= 0.60:
        return "transient"
    if lo >= 0.85:
        return "persistent"
    if lo <= 0.50 and hi >= 0.90:
        return "not separable"
    return "partial"


def analyse(label, timeline, theta, H, M=2000):
    panel = build_panel(timeline)
    if not panel:
        print(f"[{label}] empty timeline -- skipped")
        return None
    model = fit_panel(panel)
    lo, hi = model["phi_ci95"]
    verdict = separability(model["phi_ci95"])
    lengths = sorted({len(p["y"]) for p in panel})
    span = (f"{lengths[0]}" if len(lengths) == 1
            else f"{lengths[0]}-{lengths[-1]}")
    print(f"\n=== E2 [{label}] {len(panel)} skills x {span} checkpoints ===")
    deg = ""
    if model["degenerate"]:
        # The reported phi is the constant-mean model (1.0); the interval
        # belongs to the *fitted* phi, so the point may lie outside it.
        deg = (f"  [DEGENERATE tau0~0: using phi=1; "
               f"fitted phi={model.get('phi_fitted', model['phi']):.2f}]")
    print(f"phi={model['phi']:.2f}  95% CI=[{lo:.2f}, {hi:.2f}]  "
          f"tau0={model['tau0']:.4f}{deg}")
    print(f"separability: {verdict}")

    skills = []
    for p in panel:
        sid = p["id"]
        tau = model["tau"][sid]
        sm_m, sm_P = run_series(model["phi"], tau, p["y"], p["s2"],
                                p["gaps"])
        mu, var = sm_m[-1], sm_P[-1]
        # report the decisions both under the fit and under the E1 grid, so a
        # wide CI cannot silently become a confident statement
        cond = {}
        for phi in PHI_GRID:
            cond[f"phi{int(phi*100)}"] = {
                "p_persistent": p_persistent(mu, var, phi, tau, H, theta,
                                             M=M),
                "p_recovery": p_recovery(mu, var, phi, tau, H, THETA_KEEP,
                                         M=M),
            }
        row = {
            "skill": sid,
            "tau": tau,
            "y": p["y"],
            "smoothed": sm_m,
            "mu_T": mu,
            "sd_T": var ** 0.5,
            "trend": sm_m[-1] - sm_m[0],
            "conditional": cond,
        }
        skills.append(row)
        sm = "  ".join(f"{v:+.3f}" for v in sm_m)
        print(f"  {sid:9} tau={tau:.4f}  smoothed: {sm}   "
              f"mu_T={mu:+.3f}+-{var**0.5:.3f}  "
              f"P_persist(fit,phi={model['phi']:.2f})="
              f"{p_persistent(mu, var, model['phi'], tau, H, theta, M=M):.3f}")

    return {
        "label": label,
        "n_skills": len(panel),
        "n_points": max(lengths),
        "n_points_by_skill": {p["id"]: len(p["y"]) for p in panel},
        "phi": model["phi"],
        "phi_fitted": model.get("phi_fitted", model["phi"]),
        "phi_ci95": list(model["phi_ci95"]),
        "tau0": model["tau0"],
        "degenerate": model["degenerate"],
        "profile": model["profile"],
        "separability": verdict,
        "skills": skills,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--timeline", action="append", required=True,
                    help="label=path (repeatable, e.g. 3pt=... 5pt=...)")
    ap.add_argument("--theta", type=float, default=-0.02)
    ap.add_argument("--H", type=int, default=2)
    ap.add_argument("--out", default="results/stage57_e2")
    args = ap.parse_args()

    results = []
    for spec in args.timeline:
        if "=" not in spec:
            ap.error("--timeline must be label=path")
        label, path = spec.split("=", 1)
        if not Path(path).exists():
            print(f"[skip] {label}: missing {path}", file=sys.stderr)
            continue
        r = analyse(label, UtilityTimeline.load(path), args.theta, args.H)
        if r is not None:
            results.append(r)

    if len(results) >= 2:
        print("\n=== identifiability gain ===")
        print(f"{'timeline':10} {'points':>7} {'phi':>6} {'CI':>16} {'width':>7}  verdict")
        for r in results:
            lo, hi = r["phi_ci95"]
            width = hi - lo if hi == hi and lo == lo else float("nan")
            print(f"{r['label']:10} {r['n_points']:>7} {r['phi']:>6.2f} "
                  f"{'[%.2f,%.2f]' % (lo, hi):>16} {width:>7.2f}  "
                  f"{r['separability']}")

    if args.out:
        os.makedirs(args.out, exist_ok=True)
        path = os.path.join(args.out, "e2_fit.json")
        with open(path, "w") as f:
            json.dump([r for r in results if r], f, indent=2, ensure_ascii=False)
        print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
