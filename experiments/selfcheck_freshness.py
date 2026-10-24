"""Self-check for the Stage 5.5 Freshness mechanism (design doc section 7).

Pure Python, no model: exercises the formula, the no-op semantics, the
heterogeneous demotion (the one positive case), timeline merging and legacy
backward compatibility.  Exit 0 iff all checks pass.

Usage:
    python experiments/selfcheck_freshness.py
"""

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # SkillRL root

from freshskill import UtilityStore, UtilityTimeline, FreshnessScorer  # noqa: E402
from freshskill.scorer import freshness, freshness_score  # noqa: E402


def check(cond, msg):
    print(f"[{'PASS' if cond else 'FAIL'}] {msg}")
    return bool(cond)


def main():
    ok = True

    # 1. formula: U=0.5, t=20, current=40, lambda=0.05 -> F=e^-1, Score=U*e^-1
    store = UtilityStore()
    store.set("s", "p", 0.5, 1, 0, 0, 1, step=20)
    f = freshness("s", "p", store, current_step=40, lambda_=0.05)
    ok &= check(abs(f - math.exp(-1.0)) < 1e-6, f"F_i == e^-1 ({f:.6f})")
    ok &= check(
        abs(freshness_score("s", "p", store, 40, 0.05) - 0.5 * math.exp(-1.0)) < 1e-6,
        "Score_i == U_i * F_i",
    )

    # 2. no-op semantics: uniform age -> rank == by-U, filter(tau=0) == by-U
    tl = UtilityTimeline()
    for sid, u in {"a": 0.3, "b": -0.1, "c": 0.2}.items():
        tl.set(sid, 20, u, {"help": 1, "harm": 0, "neutral": 0, "usage": 1})
    sc = FreshnessScorer(tl, current_step=40, lambda_=0.05)  # all measured at 20
    skills = [{"skill_id": "a"}, {"skill_id": "b"}, {"skill_id": "c"}]
    by_u = sorted(skills, key=lambda s: tl.utility(s["skill_id"], 20), reverse=True)
    ranked = sc.select(skills, mode="rank")
    ok &= check([s["skill_id"] for s in ranked] == [s["skill_id"] for s in by_u],
                "uniform-age rank == by-U order (no-op)")
    filt = sc.select(skills, mode="filter", threshold=0.0)
    by_u_filt = [s["skill_id"] for s in skills if tl.utility(s["skill_id"], 20) >= 0.0]
    ok &= check([s["skill_id"] for s in filt] == by_u_filt,
                "uniform-age filter(tau=0) == by-U filter (no-op)")

    # 3. heterogeneous demotion: fresh gen_001 (+0.021@40) vs stale gen_005
    #    (+0.029@0, actually harmful now).  lambda>0 must demote gen_005 below
    #    gen_001; lambda=0 (naive) must rank gen_005 first.
    tl2 = UtilityTimeline()
    tl2.set("gen_001", 0, 0.007, {})
    tl2.set("gen_001", 40, 0.021, {})
    tl2.set("gen_005", 0, 0.029, {})
    tl2.set("gen_005", 40, -0.014, {})
    steps = {"*": 40, "gen_005": 0}  # gen_005 stale since SFT, rest fresh
    fresh_sc = FreshnessScorer(tl2, current_step=40, lambda_=0.05, steps=steps)
    s001, s005 = fresh_sc.score("gen_001"), fresh_sc.score("gen_005")
    ok &= check(s001 > s005, f"freshness: score(gen_001)={s001:.4f} > score(gen_005)={s005:.4f}")
    naive_sc = FreshnessScorer(tl2, current_step=40, lambda_=0.0, steps=steps)
    ok &= check(naive_sc.score("gen_005") > naive_sc.score("gen_001"),
                "naive (lambda=0): score(gen_005) > score(gen_001) (order flips)")

    # 4. timeline merge: latest_le / utility lookup across the 3 policy stores
    s_sft = UtilityStore(); s_sft.set("gen_001", "SFT55", 0.007, 0, 0, 0, 1)
    s_20 = UtilityStore();  s_20.set("gen_001", "RL_ckpt20", 0.036, 0, 0, 0, 1)
    s_40 = UtilityStore();  s_40.set("gen_001", "RL_ckpt40", 0.021, 0, 0, 0, 1)
    merged = UtilityTimeline.from_policy_stores(
        [s_sft, s_20, s_40], {"SFT55": 0, "RL_ckpt20": 20, "RL_ckpt40": 40}
    )
    ok &= check(merged.latest_le("gen_001", 25) == 20, "latest_le(gen_001,25) == 20")
    ok &= check(abs(merged.utility("gen_001", 40) - 0.021) < 1e-9,
                "utility(gen_001,40) == 0.021")

    # 5. backward compatibility: a store without a step field -> step()==None
    #    -> freshness_score degenerates to U_i (age treated as 0).
    legacy = UtilityStore()
    legacy.set("s", "p", 0.7, 0, 0, 0, 1)  # no step
    ok &= check(legacy.step("s", "p") is None, "legacy store step() == None")
    ok &= check(freshness_score("s", "p", legacy, 40, 0.05) == 0.7,
                "legacy freshness_score == U_i")

    print()
    print("ALL PASS" if ok else "SOME CHECKS FAILED")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
