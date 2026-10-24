#!/usr/bin/env python
"""2x2 factorial decomposition of the 1.5B skill bank at ckpt40.

Cells (all same 140-task order, seed 0, paired by row index):

                task-specific absent   task-specific present
    general absent        none                  task_specific_only
    general present       general_only          full

Effects on the success-rate scale
    G  = ((full - task_specific_only) + (general_only - none)) / 2
    TS = ((full - general_only) + (task_specific_only - none)) / 2
    interaction = full - general_only - task_specific_only + none

Pairwise cells get an exact McNemar test; the factorial effects get a
task-level paired bootstrap (percentile CI), because the interaction variance
is not a simple binomial.

    python experiments/analyze_factorial.py \
        --out results/stage57_factorial/factorial_ckpt40.json
"""
import argparse
import csv
import json
import math
import os
import random
import sys

DEFAULTS = {
    "none": "results/stage57_noskill/base_ckpt40_noskill.csv",
    "full": "results/stage55_measure/base_ckpt40.csv",
    "general_only": "results/stage57_factorial/base_ckpt40_general_only.csv",
    "task_specific_only":
        "results/stage57_factorial/base_ckpt40_task_specific_only.csv",
}
# Cell order for the 2x2 table.
CELL_ORDER = ["none", "general_only", "task_specific_only", "full"]


def load(path):
    with open(path) as f:
        rows = list(csv.DictReader(f))
    return rows


def binom_two_sided(b, n):
    if n == 0:
        return 1.0
    p_obs = math.comb(n, b) / 2.0 ** n
    p = sum(math.comb(n, k) / 2.0 ** n for k in range(n + 1)
            if math.comb(n, k) / 2.0 ** n <= p_obs + 1e-12)
    return min(1.0, p)


def wins(rows):
    return [int(float(r["won"])) for r in rows]


def sr(y):
    return sum(y) / len(y)


def mcnemar(y_cell, y_ref):
    b = sum(1 for a, r in zip(y_cell, y_ref) if a == 1 and r == 0)
    c = sum(1 for a, r in zip(y_cell, y_ref) if a == 0 and r == 1)
    return {"delta": (b - c) / len(y_cell), "b": b, "c": c,
            "p": binom_two_sided(min(b, c), b + c)}


def effects(Y):
    full = sr(Y["full"])
    gen = sr(Y["general_only"])
    ts = sr(Y["task_specific_only"])
    none = sr(Y["none"])
    return {
        "G": ((full - ts) + (gen - none)) / 2.0,
        "TS": ((full - gen) + (ts - none)) / 2.0,
        "interaction": full - gen - ts + none,
    }


def bootstrap(Y, reps, seed=0):
    n = len(Y["none"])
    rng = random.Random(seed)
    draws = {k: [] for k in ("G", "TS", "interaction")}
    idx = list(range(n))
    for _ in range(reps):
        sample = [rng.randrange(n) for _ in idx]
        resampled = {k: [Y[k][i] for i in sample] for k in Y}
        e = effects(resampled)
        for k in draws:
            draws[k].append(e[k])
    out = {}
    for k, vals in draws.items():
        vals.sort()
        out[k] = {"ci95": [vals[int(0.025 * reps)],
                           vals[int(0.975 * reps) - 1]],
                  "mean": sum(vals) / reps}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    ap.add_argument("--bootstrap", type=int, default=5000)
    for k, v in DEFAULTS.items():
        ap.add_argument(f"--{k.replace('_', '-')}", default=v)
    args = ap.parse_args()
    paths = {k: getattr(args, k) for k in DEFAULTS}

    rows = {}
    for k, p in paths.items():
        if not os.path.exists(p):
            raise SystemExit(f"missing cell {k}: {p}")
        rows[k] = load(p)
    n = len(rows["none"])
    for k in rows:
        if len(rows[k]) != n:
            raise SystemExit(f"row count mismatch in {k}")
        ref = [r["task_id"] for r in rows["none"]]
        if [r["task_id"] for r in rows[k]] != ref:
            raise SystemExit(f"task order mismatch in {k}")

    # Integrity: the injected groups must match the cell name.  The CSV
    # column exists only for runs after 2026-10-20; fall back to skill_ids.
    integrity = {}
    for k in ("general_only", "task_specific_only"):
        gen = ts = 0
        for r in rows[k]:
            ids = [s for s in r["skill_ids"].split(",") if s]
            gen += any(s.startswith("gen_") for s in ids)
            ts += any(not s.startswith("gen_") for s in ids)
        integrity[k] = {"episodes_with_general": gen,
                        "episodes_with_task_specific": ts}
        if k == "general_only" and (gen != n or ts != 0):
            raise SystemExit(f"{k} leaked task-specific or lost general skills")
        if k == "task_specific_only" and (ts != n or gen != 0):
            raise SystemExit(f"{k} leaked general or lost task-specific skills")

    Y = {k: wins(rows[k]) for k in rows}
    result = {
        "n": n,
        "cells": {k: {"sr": sr(Y[k]), "wins": sum(Y[k])} for k in CELL_ORDER},
        "effects": effects(Y),
        "bootstrap": bootstrap(Y, args.bootstrap),
        "pairwise": {},
        "integrity": integrity,
    }
    # Same decomposition within each task type (descriptive: the per-type n is
    # 13-35, so no CIs here).
    types = []
    for tt in dict.fromkeys(r["task_type"] for r in rows["none"]):
        idx = [i for i, r in enumerate(rows["none"]) if r["task_type"] == tt]
        Yt = {k: [Y[k][i] for i in idx] for k in Y}
        types.append({
            "task_type": tt, "n": len(idx),
            "cells": {k: {"sr": sr(Yt[k]), "wins": sum(Yt[k])}
                      for k in CELL_ORDER},
            "effects": effects(Yt),
        })
    result["by_task_type"] = types
    contrasts = [
        ("full", "none"), ("full", "task_specific_only"),
        ("full", "general_only"), ("general_only", "none"),
        ("task_specific_only", "none"),
        ("general_only", "task_specific_only"),
    ]
    for a, b in contrasts:
        result["pairwise"][f"{a}_vs_{b}"] = mcnemar(Y[a], Y[b])

    print(f"n={n}")
    print("cell success rates:")
    for k in CELL_ORDER:
        print(f"  {k:22s} {result['cells'][k]['sr']:.4f} "
              f"({result['cells'][k]['wins']}/{n})")
    e = result["effects"]
    bs = result["bootstrap"]
    print("\nfactorial effects (SR scale):")
    print(f"  G  (general group)   {e['G']:+.4f}  "
          f"95% CI [{bs['G']['ci95'][0]:+.4f}, {bs['G']['ci95'][1]:+.4f}]")
    print(f"  TS (task-specific)   {e['TS']:+.4f}  "
          f"95% CI [{bs['TS']['ci95'][0]:+.4f}, {bs['TS']['ci95'][1]:+.4f}]")
    print(f"  interaction          {e['interaction']:+.4f}  "
          f"95% CI [{bs['interaction']['ci95'][0]:+.4f}, "
          f"{bs['interaction']['ci95'][1]:+.4f}]")
    print("\npairwise (exact McNemar):")
    for k, v in result["pairwise"].items():
        print(f"  {k:42s} d={v['delta']:+.4f} "
              f"b/c={v['b']}/{v['c']} p={v['p']:.4f}")

    print("\nby task type (descriptive):")
    for t in result["by_task_type"]:
        c = t["cells"]
        e = t["effects"]
        print(f"  {t['task_type']:32s} n={t['n']:3d} "
              f"none={c['none']['wins']:2d} gen={c['general_only']['wins']:2d} "
              f"ts={c['task_specific_only']['wins']:2d} full={c['full']['wins']:2d}"
              f"  G={e['G']:+.3f} TS={e['TS']:+.3f} int={e['interaction']:+.3f}")

    if args.out:
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(result, f, indent=1)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
