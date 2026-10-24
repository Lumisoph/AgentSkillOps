#!/usr/bin/env python
"""P1: TS Only vs Full on 1.5B ckpt20, valid_train (200 independent tasks).

Pre-registered in docs/stage57_p1_prereg.md.
Primary: ts_only - full SR non-inferiority (95% exact conditional CI lower
bound > -0.05) AND lower prompt_tokens/episode. Exact McNemar, bootstrap cost
CI, and exploratory per-routed-group Delta_G.

    python experiments/analyze_p1.py --out results/stage57_p1/p1_valid_train.json
"""
import argparse
import csv
import json
from pathlib import Path

from analyze_p2 import mcnemar, mean


def load(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def routed_group(skill_ids):
    prefixes = {
        s.split("_")[0]
        for s in skill_ids.split(",")
        if s and not s.startswith("gen_")
    }
    if len(prefixes) > 1:
        raise SystemExit(f"mixed routed prefixes: {prefixes}")
    return next(iter(prefixes)) if prefixes else "none"


def bootstrap_ci(diffs, reps=5000, seed=0):
    import random
    rng = random.Random(seed)
    n = len(diffs)
    means = []
    for _ in range(reps):
        means.append(sum(diffs[rng.randrange(n)] for _ in range(n)) / n)
    means.sort()
    return means[int(0.025 * reps)], means[int(0.975 * reps)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="results/stage57_p1")
    ap.add_argument("--out", default="results/stage57_p1/p1_valid_train.json")
    args = ap.parse_args()

    d = Path(args.dir)
    full = load(d / "base_full.csv")
    ts = load(d / "base_ts_only.csv")
    n = len(full)
    if len(ts) != n:
        raise SystemExit(f"ts_only {len(ts)} rows vs full {n}")
    for i in range(n):
        for c in (full, ts):
            pass
        if (full[i]["task_id"], full[i]["seed"]) != (ts[i]["task_id"], ts[i]["seed"]):
            raise SystemExit(f"row {i} misaligned")

    y_full = [int(float(r["won"])) for r in full]
    y_ts = [int(float(r["won"])) for r in ts]

    m = mcnemar(y_ts, y_full)  # b = ts-only wins, c = full wins
    ci = m["ci95_exact"]
    noninf = bool(ci is not None and float(ci[0]) > -0.05)

    p_full = [float(r["prompt_tokens"]) for r in full]
    p_ts = [float(r["prompt_tokens"]) for r in ts]
    diff = [a - b for a, b in zip(p_ts, p_full)]
    lo_ci, hi_ci = bootstrap_ci(diff)
    cost_down = mean(diff) < 0

    # per routed group (routing from full bank; identical for ts_only)
    groups = {}
    for i in range(n):
        groups.setdefault(routed_group(full[i]["skill_ids"]), []).append(i)
    by_group = {}
    for g, idx in sorted(groups.items()):
        yf = [y_full[i] for i in idx]
        yt = [y_ts[i] for i in idx]
        mg = mcnemar(yt, yf)
        pf = [p_full[i] for i in idx]
        pt = [p_ts[i] for i in idx]
        by_group[g] = {
            "n": len(idx),
            "wins": {"full": sum(yf), "ts_only": sum(yt)},
            "delta_ts_only_minus_full": mg["delta"],
            "ts_only_only": mg["b"], "full_only": mg["c"],
            "ci95_exact": mg["ci95_exact"],
            "p_two_sided": mg["p_two_sided"],
            "prompt_diff_mean": mean([a - b for a, b in zip(pt, pf)]),
        }

    out = {
        "n": n,
        "dataset": "eval_valid_train",
        "cells": {
            "full": {"wins": sum(y_full), "sr": mean(y_full),
                     "prompt_tokens_mean": mean(p_full)},
            "ts_only": {"wins": sum(y_ts), "sr": mean(y_ts),
                        "prompt_tokens_mean": mean(p_ts)},
        },
        "primary": {
            "delta_ts_only_minus_full": m["delta"],
            "ts_only_only": m["b"], "full_only": m["c"],
            "ci95_exact": ci,
            "p_two_sided": m["p_two_sided"],
            "non_inferiority_passed": noninf,
            "margin": -0.05,
            "prompt_diff_mean": mean(diff),
            "prompt_diff_pct": mean(diff) / mean(p_full),
            "prompt_diff_ci95_boot": [lo_ci, hi_ci],
            "cost_down": cost_down,
            "h1_passed": noninf and cost_down,
        },
        "by_routed_group": by_group,
    }

    Path(args.out).write_text(json.dumps(out, indent=2) + "\n")

    c = out["cells"]
    print(f"n={n}  dataset=eval_valid_train (ckpt20)")
    print(f"  full    SR={c['full']['sr']:.4f} ({c['full']['wins']}/{n})  "
          f"prompt/ep={c['full']['prompt_tokens_mean']:.0f}")
    print(f"  ts_only SR={c['ts_only']['sr']:.4f} ({c['ts_only']['wins']}/{n})  "
          f"prompt/ep={c['ts_only']['prompt_tokens_mean']:.0f}")
    p = out["primary"]
    ci_s = "—" if ci is None else f"[{ci[0]:+.4f},{ci[1]:+.4f}]"
    print(f"  ts_only-full: delta={p['delta_ts_only_minus_full']:+.4f} "
          f"({p['ts_only_only']}/{p['full_only']}) CI={ci_s} "
          f"p2={p['p_two_sided']:.4f}")
    print(f"  non-inferiority(-0.05): {'PASS' if noninf else 'FAIL'} | "
          f"prompt diff={p['prompt_diff_mean']:+.0f} ({p['prompt_diff_pct']:+.1%}) "
          f"boot CI=[{lo_ci:+.0f},{hi_ci:+.0f}] | H1={'PASS' if out['primary']['h1_passed'] else 'FAIL'}")
    print("  per routed group (exploratory):")
    for g, r in by_group.items():
        ci_g = "—" if r["ci95_exact"] is None else \
            f"[{r['ci95_exact'][0]:+.3f},{r['ci95_exact'][1]:+.3f}]"
        print(f"    {g:4} n={r['n']:3} full={r['wins']['full']:2} "
              f"ts={r['wins']['ts_only']:2} delta={r['delta_ts_only_minus_full']:+.3f} "
              f"CI={ci_g} prompt_diff={r['prompt_diff_mean']:+.0f}")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
