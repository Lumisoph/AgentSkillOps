"""Self-check for the Stage 5.6 Skill Lifecycle (design doc section 8).

Pure Python, no model: exercises the confirm-before-retire rule, the
premature-retirement guard, the first-cycle behaviour, re-measure priority and
bank-state transitions.  Exit 0 iff all checks pass.

Usage:
    python experiments/selfcheck_lifecycle.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # SkillRL root

from freshskill import BankState, LifecyclePolicy, UtilityTimeline  # noqa: E402
from freshskill.lifecycle import KEEP, RETIRE  # noqa: E402


def check(cond, msg):
    print(f"[{'PASS' if cond else 'FAIL'}] {msg}")
    return bool(cond)


def main():
    ok = True
    lp = LifecyclePolicy(theta_retire=-0.02)

    # 1. confirm rule: retire only when *both* cycles are below theta.
    tl = UtilityTimeline()
    tl.set("a", 20, -0.05, {"usage": 140})
    tl.set("a", 40, -0.05, {"usage": 140})   # both harmful -> RETIRE
    tl.set("b", 20, -0.05, {"usage": 140})
    tl.set("b", 40, 0.01, {"usage": 140})     # recovered -> KEEP
    tl.set("c", 20, 0.01, {"usage": 140})
    tl.set("c", 40, -0.05, {"usage": 140})    # not confirmed -> KEEP
    tl.set("d", 20, 0.02, {"usage": 140})
    tl.set("d", 40, 0.03, {"usage": 140})     # both helpful -> KEEP
    actions = lp.maintain(tl, 40, prev_step=20)
    ok &= check(actions["a"] == RETIRE, "both-negative -> RETIRE")
    ok &= check(actions["b"] == KEEP, "neg-then-pos -> KEEP (recovered)")
    ok &= check(actions["c"] == KEEP, "pos-then-neg -> KEEP (not confirmed)")
    ok &= check(actions["d"] == KEEP, "both-positive -> KEEP")

    # 2. premature retirement guard: U(20)=-0.05, U(40)=0.00.
    tl2 = UtilityTimeline()
    tl2.set("s", 20, -0.05, {"usage": 140})
    tl2.set("s", 40, 0.00, {"usage": 140})
    single20 = {sid for sid in tl2.skill_ids()
                if lp.is_pending(tl2.utility(sid, 20))}
    ok &= check(single20 == {"s"}, "single-snapshot@20 would retire s")
    ok &= check(lp.maintain(tl2, 40, prev_step=20)["s"] == KEEP,
                "confirm keeps s (U(40)=0.00 >= theta)")

    # 3. first cycle: no previous cycle -> record only, never retire.
    tl3 = UtilityTimeline()
    tl3.set("s", 20, -0.05, {"usage": 140})
    ok &= check(lp.maintain(tl3, 20, prev_step=None)["s"] == KEEP,
                "first cycle -> KEEP (nothing to confirm)")

    # 4. re-measure priority: staleness first, then impact.
    tl4 = UtilityTimeline()
    tl4.set("e", 0, 0.5, {"usage": 140})   # stale (age 40)
    tl4.set("f", 40, 0.5, {"usage": 140})  # fresh, impact 0.5
    tl4.set("g", 40, 1.0, {"usage": 140})  # fresh, impact 1.0
    ok &= check(lp.select_remeasure(["e", "f", "g"], 40, tl4, 1) == ["e"],
                "stale e outranks fresh f/g (budget 1)")
    ok &= check(lp.select_remeasure(["f", "g"], 40, tl4, 1) == ["g"],
                "same age -> higher impact g wins")

    # 5. bank state: retired excluded from the active set.
    tl5 = UtilityTimeline()
    tl5.set("gen_002", 40, -0.029, {"usage": 140})
    tl5.set("gen_001", 40, 0.021, {"usage": 140})
    bs = BankState()
    bs.apply_actions({"gen_002": RETIRE, "gen_001": KEEP}, 40, tl5)
    ok &= check(bs.active_skill_ids() == ["gen_001"], "active excludes retired")
    ok &= check(bs.status("gen_002") == "retired", "retired status recorded")

    print()
    print("ALL PASS" if ok else "SOME CHECKS FAILED")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
