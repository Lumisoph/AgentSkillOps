"""``UtilityTimeline``: per-skill utility keyed by policy step (time).

Stage 5.5 Freshness (experiment_design.md section 16) needs each skill's
utility at several training steps of the *same* policy lineage.  A
``UtilityStore`` keys utility by a policy *label*; a ``UtilityTimeline`` keys it
by the *step* at which that label was measured, so freshness can look up "what
was skill i's utility when last measured at step t".

Produced by ``experiments/merge_timeline.py`` from the per-policy stores of the
measure stage, and consumed by ``freshskill.freshness.FreshnessScorer``.

JSON schema (one entry per skill; step keys are strings because JSON)::

    {
      "gen_001": {
        "0":  {"utility": 0.007, "counts": {"help": .., "harm": .., ...}},
        "20": {"utility": 0.036, "counts": {...}},
        "40": {"utility": 0.021, "counts": {...}}
      }
    }
"""

import json
import os


def _step_key(step):
    """Canonical string key for a step value (0.0 -> "0", 0.5 -> "0.5")."""
    f = float(step)
    return str(int(f)) if f.is_integer() else str(f)


class UtilityTimeline:
    def __init__(self, entries=None):
        self.entries = entries or {}

    # ---- writers --------------------------------------------------------
    def set(self, skill_id, step, utility, counts):
        self.entries.setdefault(skill_id, {})[_step_key(step)] = {
            "utility": float(utility),
            "counts": dict(counts),
        }

    # ---- readers --------------------------------------------------------
    def utility(self, skill_id, step, default=0.0):
        e = self.entries.get(skill_id, {})
        return e.get(_step_key(step), {}).get("utility", default)

    def counts(self, skill_id, step):
        e = self.entries.get(skill_id, {})
        return e.get(_step_key(step), {}).get("counts")

    def steps(self, skill_id):
        """Sorted recorded steps (floats) for a skill."""
        e = self.entries.get(skill_id, {})
        return sorted(float(s) for s in e)

    def skill_ids(self):
        return list(self.entries.keys())

    def latest_le(self, skill_id, t):
        """Largest recorded step <= t, or None if none qualify."""
        steps = [s for s in self.steps(skill_id) if s <= t]
        return steps[-1] if steps else None

    # ---- construction ---------------------------------------------------
    @classmethod
    def from_policy_stores(cls, stores, step_of_policy):
        """Fold several ``UtilityStore``s into one step-keyed timeline.

        ``stores``          : iterable of ``UtilityStore``
        ``step_of_policy``  : {policy_label: measured_step}
        """
        tl = cls()
        for store in stores:
            for skill_id in store.skill_ids():
                for policy in store.policies():
                    if policy not in step_of_policy:
                        continue
                    counts = store.counts(skill_id, policy)
                    if counts is None:
                        continue
                    tl.set(skill_id, step_of_policy[policy],
                           store.utility(skill_id, policy), counts)
        return tl

    # ---- persistence ----------------------------------------------------
    def save(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.entries, f, indent=2, ensure_ascii=False)

    @classmethod
    def load(cls, path):
        with open(path) as f:
            return cls(json.load(f))

    def __repr__(self):
        return f"UtilityTimeline({len(self.entries)} skills)"
