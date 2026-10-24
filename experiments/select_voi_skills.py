"""Rank skills by value of information, to choose the E2 measurement subset.

Stage 5.7 / E2.  The measurement budget is 2 new checkpoints (t=10, t=30) and
the question is which skills' LOO cells to buy.  A skill is worth measuring if
one more observation could plausibly change its retirement decision, i.e. if
the expected loss after observing it is lower than the loss of deciding now.

Because phi is not identified at three checkpoints (E0), the ranking is taken
over the same assumed-persistence grid E1 uses, and the reported score is the
*mean* VOI across settings.  That makes the selection robust to the
persistence assumption rather than tuned to one value of it.

The VOI is computed on the series up to the last measured checkpoint: it asks
"how much would one more measurement of this skill be worth now", which is the
same budget question as "which skill's trajectory do we most need to fill in".
It ranks skills; it never overrides a retirement decision.

Usage:
    python experiments/select_voi_skills.py --top 3
    python experiments/select_voi_skills.py --top 3 --format shell   # for bash
"""

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

from freshskill import UtilityTimeline  # noqa: E402
from freshskill.retirement import PosteriorVOIRule, series_from_timeline  # noqa: E402
from reliability_audit import B4_SETTINGS, B4_TAU  # noqa: E402


def voi_table(timeline, theta=-0.02, H=2, M=400):
    rows = []
    for sid in timeline.skill_ids():
        series = series_from_timeline(timeline, sid)
        if not series:
            continue
        per_setting = {}
        for (phi, alpha) in B4_SETTINGS:
            rule = PosteriorVOIRule(phi=phi, tau=B4_TAU, alpha=alpha,
                                    theta=theta, H=H, M=M)
            # VOI is in units of c_MH; c_FR = (1-alpha)/alpha changes by 19x
            # across the alpha grid, so averaging raw VOI would let the
            # alpha=0.05 rows dominate purely through their cost scale.
            # Normalising by c_FR makes the score "share of a false
            # retirement's cost that one measurement could save".
            per_setting[(phi, alpha)] = rule.voi(series) / rule.c_fr
        vals = list(per_setting.values())
        score = sum(vals) / len(vals)
        var = sum((v - score) ** 2 for v in vals) / len(vals)
        rows.append({
            "skill": sid,
            "voi_mean": score,
            "voi_sd": var ** 0.5,
            "voi_max": max(vals),
            "per_setting": {f"p{int(p*100)}_a{int(a*100)}": v
                            for (p, a), v in per_setting.items()},
            "y_last": series[-1]["y"],
        })
    rows.sort(key=lambda r: r["voi_mean"], reverse=True)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--timeline",
                    default="results/stage55_measure/utility_store_timeline.json")
    ap.add_argument("--theta", type=float, default=-0.02)
    ap.add_argument("--H", type=int, default=2)
    ap.add_argument("--top", type=int, default=3)
    ap.add_argument("--format", choices=["table", "shell"], default="table")
    args = ap.parse_args()

    tl = UtilityTimeline.load(args.timeline)
    rows = voi_table(tl, theta=args.theta, H=args.H)

    if args.format == "shell":
        print(" ".join(r["skill"] for r in rows[:args.top]))
        return

    print(f"VOI at the last measured checkpoint (theta={args.theta}, H={args.H})")
    print("VOI is normalised by c_FR per setting; sd is across the 4 settings.")
    print(f"{'skill':10} {'VOI mean':>10} {'VOI sd':>8} {'VOI max':>9} "
          f"{'u(t_last)':>10}   per (phi,alpha)")
    for r in rows:
        detail = "  ".join(f"{k}={v:.4f}" for k, v in r["per_setting"].items())
        print(f"{r['skill']:10} {r['voi_mean']:>10.4f} {r['voi_sd']:>8.4f} "
              f"{r['voi_max']:>9.4f} {r['y_last']:>10.4f}   {detail}")
    print(f"\nE2 subset (top {args.top}): "
          + " ".join(r["skill"] for r in rows[:args.top]))


if __name__ == "__main__":
    main()
