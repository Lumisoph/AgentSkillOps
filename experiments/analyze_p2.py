#!/usr/bin/env python
"""P2: independent-task validation of the group-conditional activation rule.

Five paired cells on ALFWorld valid_unseen (eval_out_of_distribution), same
task order / seed 0:

    full            general + routed task-specific group (baseline)
    ts_only         routed task-specific group only
    general_only    general skills only
    selective       general + task-specific only for routed groups cle/coo
    selective_cle   general + task-specific only for routed group cle

Primary endpoint (pre-registered in docs/stage57_p2_prereg.md):
    selective vs full: SR non-inferiority (95% CI lower bound > -0.05) and
    lower prompt-token cost.

    python experiments/analyze_p2.py \
        --out results/stage57_p2/p2_valid_unseen.json
"""
import argparse
import csv
import json
import math
import os
import random

DEFAULTS = {
    "full": "results/stage57_p2/base_full.csv",
    "ts_only": "results/stage57_p2/base_ts_only.csv",
    "general_only": "results/stage57_p2/base_general_only.csv",
    "selective": "results/stage57_p2/base_selective.csv",
    "selective_cle": "results/stage57_p2/base_selective_cle.csv",
}
CELL_ORDER = ["full", "ts_only", "general_only", "selective",
              "selective_cle"]


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


def binom_one_sided_greater(b, n):
    """P(X >= b) under Binomial(n, 0.5)."""
    if n == 0:
        return 1.0
    return sum(math.comb(n, k) for k in range(b, n + 1)) / 2.0 ** n


def mcnemar(y_cell, y_ref):
    n = len(y_cell)
    b = sum(1 for a, r in zip(y_cell, y_ref) if a == 1 and r == 0)
    c = sum(1 for a, r in zip(y_cell, y_ref) if a == 0 and r == 1)
    delta = (b - c) / n
    m = b + c
    ci = None
    if m > 0:
        from scipy.stats import beta as _beta
        lo = 0.0 if b == 0 else _beta.ppf(0.025, b, m - b + 1)
        hi = 1.0 if b == m else _beta.ppf(0.975, b + 1, m - b)
        ci = [(2 * lo - 1) * m / n, (2 * hi - 1) * m / n]
    return {"delta": delta, "b": b, "c": c,
            "p_two_sided": binom_two_sided(min(b, c), m),
            # directional: is `cell` more often the winner than `ref`?
            "p_greater": binom_one_sided_greater(b, m),
            "ci95_exact": ci}


def mean(xs):
    return sum(xs) / len(xs) if xs else 0.0


def median(xs):
    if not xs:
        return 0.0
    s = sorted(xs)
    k = len(s)
    return s[k // 2] if k % 2 else 0.5 * (s[k // 2 - 1] + s[k // 2])


def boot_ci(vals, reps=10000, seed=0):
    if not vals:
        return None
    rng = random.Random(seed)
    n = len(vals)
    draws = sorted(sum(vals[rng.randrange(n)] for _ in range(n)) / n
                   for _ in range(reps))
    return [draws[int(0.025 * reps)], draws[int(0.975 * reps) - 1]]


def cost(rows):
    p = [float(r["prompt_tokens"]) for r in rows]
    s = [float(r["steps"]) for r in rows]
    return {
        "prompt_tokens_mean": mean(p),
        "prompt_tokens_median": median(p),
        "prompt_tokens_per_step": mean([a / b for a, b in zip(p, s) if b > 0]),
        "response_tokens_mean": mean([float(r["response_tokens"]) for r in rows]),
        "steps_mean": mean(s),
        "won": sum(int(float(r["won"])) for r in rows),
        "n": len(rows),
        "tokens_per_success": (sum(p) / sum(int(float(r["won"])) for r in rows)
                               if any(int(float(r["won"])) for r in rows)
                               else None),
    }


def routed_group(row):
    prefixes = {x.split("_")[0] for x in row["skill_ids"].split(",")
                if x and not x.startswith("gen_")}
    if len(prefixes) > 1:
        raise SystemExit(f"mixed routed prefixes: {prefixes}")
    return next(iter(prefixes)) if prefixes else "none"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    ap.add_argument("--boot", type=int, default=10000)
    for k, v in DEFAULTS.items():
        ap.add_argument(f"--{k.replace('_', '-')}", default=v)
    args = ap.parse_args()
    paths = {k: getattr(args, k) for k in DEFAULTS}

    rows = {k: load(p) for k, p in paths.items()}
    n = len(rows["full"])
    ref = [(r["task_id"], r["task_type"]) for r in rows["full"]]
    for k in rows:
        if len(rows[k]) != n:
            raise SystemExit(f"{k}: {len(rows[k])} rows, expected {n}")
        if [(r["task_id"], r["task_type"]) for r in rows[k]] != ref:
            raise SystemExit(f"{k}: task order/type mismatch")

    Y = {k: [int(float(r["won"])) for r in rows[k]] for k in CELL_ORDER}
    routed = [routed_group(r) for r in rows["full"]]
    # The router must have seen the same group in every cell that injects TS.
    got_ts = [routed_group(r) for r in rows["ts_only"]]
    if got_ts != routed:
        raise SystemExit("ts_only routed groups differ from full")
    for k, allowed in (("selective", {"cle", "coo"}),
                       ("selective_cle", {"cle"})):
        got = [routed_group(r) for r in rows[k]]
        for i, (a, b) in enumerate(zip(got, routed)):
            expected = b if b in allowed else "none"
            if a != expected:
                raise SystemExit(
                    f"{k} row {i}: routed {a}, expected {expected} (full {b})")

    result = {
        "n": n,
        "dataset": rows["full"][0].get("eval_dataset", "unknown"),
        "cells": {},
        "cost": {k: cost(rows[k]) for k in CELL_ORDER},
        "contrasts": {},
        "cost_contrasts": {},
        "by_routed_group": [],
        "by_task_type": [],
    }
    for k in CELL_ORDER:
        result["cells"][k] = {"wins": sum(Y[k]), "sr": sum(Y[k]) / n}

    contrasts = [
        ("selective", "full"),
        ("selective", "general_only"),
        ("selective_cle", "selective"),
        ("ts_only", "full"),
        ("general_only", "full"),
        ("selective_cle", "full"),
    ]
    for a, b in contrasts:
        result["contrasts"][f"{a}_vs_{b}"] = mcnemar(Y[a], Y[b])
        ta = [float(r["prompt_tokens"]) for r in rows[a]]
        tb = [float(r["prompt_tokens"]) for r in rows[b]]
        diff = [x - y for x, y in zip(ta, tb)]
        result["cost_contrasts"][f"{a}_vs_{b}"] = {
            "prompt_diff_mean": mean(diff),
            "prompt_diff_median": median(diff),
            "prompt_diff_ci95_boot": boot_ci(diff, args.boot),
            "prompt_diff_pct": mean(diff) / mean(tb) if mean(tb) else None,
        }

    def stratify(keyfn):
        keys = list(dict.fromkeys(keyfn(i) for i in range(n)))
        out = []
        for key in keys:
            idx = [i for i in range(n) if keyfn(i) == key]
            Yt = {k: [Y[k][i] for i in idx] for k in CELL_ORDER}
            out.append({
                "stratum": key,
                "n": len(idx),
                "cells": {k: {"wins": sum(Yt[k]),
                              "sr": sum(Yt[k]) / len(idx)}
                          for k in CELL_ORDER},
                "selective_vs_full": mcnemar(Yt["selective"], Yt["full"]),
                "full_vs_general_only":
                    mcnemar(Yt["full"], Yt["general_only"]),
                "ts_only_vs_full": mcnemar(Yt["ts_only"], Yt["full"]),
            })
        return out

    result["by_routed_group"] = stratify(lambda i: routed[i])
    result["by_task_type"] = stratify(
        lambda i: rows["full"][i]["task_type"])

    if args.out:
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(result, f, indent=1)

    # ---- report --------------------------------------------------------
    print(f"dataset={result['dataset']} n={n}\n")
    print("cells:")
    for k in CELL_ORDER:
        c, co = result["cells"][k], result["cost"][k]
        tps = ("—" if co["tokens_per_success"] is None
               else f"{co['tokens_per_success']:.0f}")
        print(f"  {k:15s} SR={c['sr']:.4f} ({c['wins']}/{n})  "
              f"prompt/ep={co['prompt_tokens_mean']:7.0f}  "
              f"prompt/step={co['prompt_tokens_per_step']:6.0f}  "
              f"steps={co['steps_mean']:4.1f}  tokens/success={tps}")

    print("\npre-registered / secondary contrasts (exact McNemar):")
    for name, v in result["contrasts"].items():
        ci = v["ci95_exact"]
        cis = "—" if ci is None else f"[{ci[0]:+.3f}, {ci[1]:+.3f}]"
        cc = result["cost_contrasts"][name]
        print(f"  {name:30s} Δ={v['delta']:+.4f} b/c={v['b']:d}/{v['c']:d} "
              f"CI={cis} p2={v['p_two_sided']:.4f} p1={v['p_greater']:.4f}  "
              f"Δtok/ep={cc['prompt_diff_mean']:+.0f} "
              f"({cc['prompt_diff_pct']:+.1%})")

    for title, fam in [("by routed group", "by_routed_group"),
                       ("by nominal task_type", "by_task_type")]:
        print(f"\n{title}:")
        for s in result[fam]:
            c = s["cells"]
            sf = s["selective_vs_full"]
            fg = s["full_vs_general_only"]
            print(f"  {s['stratum']:30s} n={s['n']:3d} "
                  f"full={c['full']['wins']:2d} ts={c['ts_only']['wins']:2d} "
                  f"gen={c['general_only']['wins']:2d} "
                  f"sel={c['selective']['wins']:2d} "
                  f"sel_cle={c['selective_cle']['wins']:2d} | "
                  f"sel-full Δ={sf['delta']:+.3f} p={sf['p_two_sided']:.3f} | "
                  f"full-gen Δ={fg['delta']:+.3f} p={fg['p_two_sided']:.3f}")
    if args.out:
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
