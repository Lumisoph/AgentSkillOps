"""``UtilityStore``: per-skill, per-policy utility estimates.

Stage 4 (experiment_design.md section 14) extends skill metadata with a
``utility`` map keyed by policy.  This module is the in-memory + on-disk
representation of that data: produced by ``experiments/analyze_utility.py``
from the Stage 3B leave-one-out rollouts, and consumed by
``freshskill.scorer`` during utility-aware retrieval.

JSON schema (one entry per skill)::

    {
      "clean_001": {
        "title": "…",
        "utility": {"SFT": 0.42, "RL": -0.08},
        "step":    {"SFT": 0, "RL": 40},
        "counts": {
          "SFT": {"help": 8, "harm": 2, "neutral": 10, "usage": 20},
          "RL":  {"help": 3, "harm": 5, "neutral": 12, "usage": 20}
        }
      }
    }
"""

import json
import os
from typing import Dict, Optional


class UtilityStore:
    def __init__(self, entries: Optional[Dict] = None):
        self.entries: Dict = entries or {}

    # ---- writers --------------------------------------------------------
    def set(self, skill_id, policy, utility, n_help, n_harm, n_neutral,
            n_usage, title=None, step=None):
        e = self.entries.setdefault(skill_id, {"utility": {}, "counts": {}})
        if title is not None:
            e["title"] = title
        e["utility"][policy] = float(utility)
        e["counts"][policy] = {
            "help": int(n_help),
            "harm": int(n_harm),
            "neutral": int(n_neutral),
            "usage": int(n_usage),
        }
        if step is not None:
            # Optional (Stage 5.5 Freshness): the training step at which this
            # measurement was taken.  Kept out of the schema unless provided so
            # legacy stores load unchanged.
            e.setdefault("step", {})[policy] = step

    # ---- readers --------------------------------------------------------
    def utility(self, skill_id: str, policy: str, default: float = 0.0) -> float:
        e = self.entries.get(skill_id)
        if not e:
            return default
        return e.get("utility", {}).get(policy, default)

    def counts(self, skill_id: str, policy: str) -> Optional[Dict]:
        e = self.entries.get(skill_id)
        if not e:
            return None
        return e.get("counts", {}).get(policy)

    def step(self, skill_id: str, policy: str):
        """Training step at which ``policy``'s estimate was measured (or None)."""
        e = self.entries.get(skill_id)
        if not e:
            return None
        return e.get("step", {}).get(policy)

    def skill_ids(self):
        return list(self.entries.keys())

    def policies(self):
        ps = set()
        for e in self.entries.values():
            ps.update(e.get("utility", {}).keys())
        return sorted(ps)

    # ---- persistence ----------------------------------------------------
    def save(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.entries, f, indent=2, ensure_ascii=False)

    @classmethod
    def load(cls, path: str) -> "UtilityStore":
        with open(path) as f:
            return cls(json.load(f))

    def __repr__(self):
        return f"UtilityStore({len(self.entries)} skills)"
