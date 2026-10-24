#!/usr/bin/env python
"""Task-conditioned and routed-group audit of the 2x2 factorial (ckpt40).

P0 (audit, no statistics)
    * which task-specific group the template router actually injected per row;
    * whether that group agrees with the ALFWorld ``task_type`` label;
    * per-cell prompt/response token cost.

P1 (paired attribution)
    For each stratum (nominal task_type, and routed skill group) compute the
    2x2 cells, the G / TS / interaction effects, paired bootstrap CIs and
    exact McNemar tests.  The routed-group strata are the primary family for
    "does adding this task-specific group help?"; the task_type strata are
    kept for continuity with the earlier report.  BH is applied within each
    family of six.

Everything is paired by row index on the same 140-task order, seed 0.

    python experiments/analyze_task_conditioned.py \
        --out results/stage57_factorial/task_conditioned_ckpt40.json
"""
import argparse
import csv
import json
import math
import os
import random

DEFAULTS = {
    "none": "results/stage57_noskill/base_ckpt40_noskill.csv",
    "full": "results/stage55_measure/base_ckpt40.csv",
    "general_only": "results/stage57_factorial/base_ckpt40_general_only.csv",
    "task_specific_only":
        "results/stage57_factorial/base_ckpt40_task_specific_only.csv",
}
CELL_ORDER = ["none", "general_only", "task_specific_only", "full"]


def load(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def binom_two_sided(b, n):
    if n == 0:
        return 1.0
    p_obs = math.comb(n, b) / 2.0 ** n
    p = sum(math.comb(n, k) / 2.0 ** n for k in range(n + 1)
            if math.comb(n, k) / 2.0 ** n <= p_obs + 1e-12)
    return min(1.0, p)


def sr(y):
    return sum(y) / len(y)


def mcnemar(y_cell, y_ref):
    """Paired difference with exact McNemar p and two 95% CIs.

    ci_exact inverts the exact binomial on the discordant pairs: the
    Clopper-Pearson interval for p = b/(b+c) maps to the difference scale as
    (2p-1)(b+c)/n.  ci_wald is the usual McNemar standard error.
    """
    n = len(y_cell)
    b = sum(1 for a, r in zip(y_cell, y_ref) if a == 1 and r == 0)
    c = sum(1 for a, r in zip(y_cell, y_ref) if a == 0 and r == 1)
    delta = (b - c) / n
    m = b + c
    ci_wald = None
    if m > 0:
        se = math.sqrt(max(0.0, m - (b - c) ** 2 / n)) / n
        ci_wald = [delta - 1.96 * se, delta + 1.96 * se]
    ci_exact = None
    if m > 0:
        from scipy.stats import beta as _beta
        lo = 0.0 if b == 0 else _beta.ppf(0.025, b, m - b + 1)
        hi = 1.0 if b == m else _beta.ppf(0.975, b + 1, m - b)
        ci_exact = [(2 * lo - 1) * m / n, (2 * hi - 1) * m / n]
    return {"delta": delta, "b": b, "c": c,
            "p": binom_two_sided(min(b, c), m),
            "ci_exact": ci_exact, "ci_wald": ci_wald}


def effects(Y):
    full, gen = sr(Y["full"]), sr(Y["general_only"])
    ts, none = sr(Y["task_specific_only"]), sr(Y["none"])
    return {"G": ((full - ts) + (gen - none)) / 2.0,
            "TS": ((full - gen) + (ts - none)) / 2.0,
            "interaction": full - gen - ts + none}


def bootstrap(Y, reps, seed=0):
    n = len(Y["none"])
    if n == 0:
        return {k: {"ci95": [0.0, 0.0], "mean": 0.0}
                for k in ("G", "TS", "interaction")}
    rng = random.Random(seed)
    draws = {k: [] for k in ("G", "TS", "interaction")}
    for _ in range(reps):
        sample = [rng.randrange(n) for _ in range(n)]
        e = effects({k: [Y[k][i] for i in sample] for k in Y})
        for k in draws:
            draws[k].append(e[k])
    out = {}
    for k, vals in draws.items():
        vals.sort()
        out[k] = {"ci95": [vals[int(0.025 * reps)],
                           vals[int(0.975 * reps) - 1]],
                  "mean": sum(vals) / reps}
    return out


def bh(pvals):
    """Benjamini-Hochberg adjusted p-values (same order as input)."""
    m = len(pvals)
    order = sorted(range(m), key=lambda i: pvals[i])
    adj = [1.0] * m
    running = 1.0
    for rank in range(m, 0, -1):
        i = order[rank - 1]
        running = min(running, pvals[i] * m / rank)
        adj[i] = min(1.0, running)
    return adj


def routed_group(row):
    """The task-specific group the template router injected for this row."""
    prefixes = {s.split("_")[0] for s in row["skill_ids"].split(",")
                if s and not s.startswith("gen_")}
    if len(prefixes) > 1:
        raise SystemExit(f"mixed routed prefixes in {row['task_id']}: {prefixes}")
    return next(iter(prefixes)) if prefixes else "none"


def mean(xs):
    return sum(xs) / len(xs) if xs else 0.0


def cost_summary(rows):
    all_p = [float(r["prompt_tokens"]) for r in rows]
    all_r = [float(r["response_tokens"]) for r in rows]
    steps = [float(r["steps"]) for r in rows]
    won = [r for r in rows if int(float(r["won"])) == 1]
    return {
        "episodes": len(rows),
        "prompt_tokens_mean": mean(all_p),
        "prompt_tokens_per_step": mean([p / s for p, s in zip(all_p, steps)
                                       if s > 0]),
        "prompt_tokens_mean_won": mean([float(r["prompt_tokens"]) for r in won]),
        "response_tokens_mean": mean(all_r),
        "steps_mean": mean(steps),
        "won": len(won),
    }


def stratum(rows_idx, Y, rows, reps):
    Yt = {k: [Y[k][i] for i in rows_idx] for k in Y}
    return {
        "n": len(rows_idx),
        "cells": {k: {"sr": sr(Yt[k]), "wins": sum(Yt[k])}
                  for k in CELL_ORDER},
        "effects": effects(Yt),
        "bootstrap": bootstrap(Yt, reps),
        "pairwise": {
            "full_vs_general_only": mcnemar(Yt["full"], Yt["general_only"]),
            "full_vs_task_specific_only":
                mcnemar(Yt["full"], Yt["task_specific_only"]),
            "task_specific_only_vs_none":
                mcnemar(Yt["task_specific_only"], Yt["none"]),
        },
        "cost": {k: cost_summary([rows[k][i] for i in rows_idx])
                 for k in CELL_ORDER},
        "discordant": {
            "full_only_vs_general_only": [
                rows["none"][i]["task_id"] for i in rows_idx
                if Y["full"][i] == 1 and Y["general_only"][i] == 0],
            "general_only_only_vs_full": [
                rows["none"][i]["task_id"] for i in rows_idx
                if Y["full"][i] == 0 and Y["general_only"][i] == 1],
            "full_only_vs_task_specific_only": [
                rows["none"][i]["task_id"] for i in rows_idx
                if Y["full"][i] == 1 and Y["task_specific_only"][i] == 0],
            "task_specific_only_only_vs_full": [
                rows["none"][i]["task_id"] for i in rows_idx
                if Y["full"][i] == 0 and Y["task_specific_only"][i] == 1],
        },
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    ap.add_argument("--bootstrap", type=int, default=5000)
    for k, v in DEFAULTS.items():
        ap.add_argument(f"--{k.replace('_', '-')}", default=v)
    args = ap.parse_args()
    paths = {k: getattr(args, k) for k in DEFAULTS}

    rows = {k: load(p) for k, p in paths.items()}
    n = len(rows["none"])
    ref = [(r["task_id"], r["task_type"]) for r in rows["none"]]
    for k in rows:
        if [(r["task_id"], r["task_type"]) for r in rows[k]] != ref:
            raise SystemExit(f"task order/type mismatch in {k}")

    # ---- P0 audit ------------------------------------------------------
    # `none` has no skill ids and `general_only` has no task-specific ids, so
    # the routed group is read from the cells that actually inject it and
    # then checked for agreement between `full` and `task_specific_only`.
    routed = {"full": [routed_group(r) for r in rows["full"]]}
    routed["task_specific_only"] = [routed_group(r) for r in rows["task_specific_only"]]
    if routed["task_specific_only"] != routed["full"]:
        raise SystemExit("routed group differs between full and task_specific_only")
    routed["none"] = routed["general_only"] = routed["full"]
    cross = {}
    for i, r in enumerate(rows["none"]):
        key = (r["task_type"], routed["full"][i])
        cross[key] = cross.get(key, 0) + 1

    # per-cell token cost (descriptive)
    costs = {k: cost_summary(rows[k]) for k in CELL_ORDER}

    # ---- P1 strata -----------------------------------------------------
    Y = {k: [int(float(r["won"])) for r in rows[k]] for k in CELL_ORDER}

    def build(keyfn):
        keys = list(dict.fromkeys(keyfn(i) for i in range(n)))
        out = []
        for key in keys:
            idx = [i for i in range(n) if keyfn(i) == key]
            s = stratum(idx, Y, rows, args.bootstrap)
            s["stratum"] = key
            out.append(s)
        # Multiplicity: two contrasts per stratum (TS|gen and G|ts).  BH is
        # applied over all 2*k tests in this family, matching the "12
        # contrasts" plan; Bonferroni is reported alongside for reference.
        ps = []
        for s in out:
            ps.append(s["pairwise"]["full_vs_general_only"]["p"])
            ps.append(s["pairwise"]["full_vs_task_specific_only"]["p"])
        adj = bh(ps)
        m = len(ps)
        for j, s in enumerate(out):
            s["multiplicity"] = {
                "family_size": m,
                "full_vs_general_only": {"bh": adj[2 * j],
                                         "bonferroni": min(1.0, ps[2 * j] * m)},
                "full_vs_task_specific_only": {
                    "bh": adj[2 * j + 1],
                    "bonferroni": min(1.0, ps[2 * j + 1] * m)},
            }
        return out

    by_type = build(lambda i: rows["none"][i]["task_type"])
    by_group = build(lambda i: routed["full"][i])

    result = {
        "n": n,
        "audit": {
            "routing_rule":
                "template memory: _detect_task_type(description) -> one "
                "task-specific group; gen_001..006 always added",
            "task_type_x_routed_group": [
                {"task_type": k[0], "routed_group": k[1], "n": v}
                for k, v in sorted(cross.items())],
            "routing_agrees_with_task_type": {
                tt: [x for x in sorted(cross) if x[0] == tt]
                for tt in dict.fromkeys(x[0] for x in sorted(cross))},
            "cost_by_cell": costs,
        },
        "by_task_type": by_type,
        "by_routed_group": by_group,
    }
    if args.out:
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(result, f, indent=1)

    # ---- report --------------------------------------------------------
    print(f"n={n}\n")
    print("P0  task_type -> routed task-specific group (per episode)")
    for (tt, g), v in sorted(cross.items()):
        print(f"  {tt:32s} -> {g:5s}  n={v}")
    print("\nP0  token cost by cell (mean per episode; prompt tokens)")
    for k in CELL_ORDER:
        c = costs[k]
        print(f"  {k:20s} prompt={c['prompt_tokens_mean']:8.0f} "
              f"per_step={c['prompt_tokens_per_step']:7.0f} "
              f"prompt_won={c['prompt_tokens_mean_won']:8.0f} "
              f"response={c['response_tokens_mean']:7.0f} "
              f"steps={c['steps_mean']:.1f} SR={c['won']}/{c['episodes']}")

    def cell(x):
        return "—" if x is None else f"[{x[0]:+.3f},{x[1]:+.3f}]"

    for title, table in [("P1a  nominal task_type", by_type),
                         ("P1b  routed task-specific group", by_group)]:
        print(f"\n{title}")
        print("  stratum                     n   none gen  ts full  "
              "| TS|gen  Δ     exact 95% CI      p     q    "
              "| G|ts   Δ     exact 95% CI      p     q")
        for s in table:
            c = s["cells"]
            pt = s["pairwise"]["full_vs_general_only"]
            pg = s["pairwise"]["full_vs_task_specific_only"]
            mt = s["multiplicity"]["full_vs_general_only"]
            mg = s["multiplicity"]["full_vs_task_specific_only"]
            print(f"  {s['stratum']:26s} {s['n']:3d}  "
                  f"{c['none']['wins']:3d} {c['general_only']['wins']:3d} "
                  f"{c['task_specific_only']['wins']:4d} {c['full']['wins']:4d}  "
                  f"| {pt['delta']:+.3f} {cell(pt['ci_exact']):18s} "
                  f"{pt['p']:.3f} {mt['bh']:.3f}  "
                  f"| {pg['delta']:+.3f} {cell(pg['ci_exact']):18s} "
                  f"{pg['p']:.3f} {mg['bh']:.3f}")
    if args.out:
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
