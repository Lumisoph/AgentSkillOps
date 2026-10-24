"""Policy-aware skill selection: ``Relevant -> Useful -> Inject``.

This is the Stage 4 (experiment_design.md section 14) extension of the
upstream ``Relevant -> Inject`` pipeline.  Given the skills retrieved by
``SkillsOnlyMemory`` and a ``UtilityStore``, it keeps only the skills whose
utility (for a given policy) clears a threshold and/or re-ranks them by
descending utility before injection.

Two building blocks mirror the Stage 5 ablation (section 15):

* ``filter_by_utility`` — drop skills with ``utility(policy) < threshold``
  (keeps original order)  →  experiment "E2 Utility Filter".
* ``rank_by_utility``    — sort by ``utility(policy)`` descending, optionally
  truncate to ``top_k`` →  experiment "E3 Utility Rank".
"""


import math


def _utility(skill, policy, store):
    return store.utility(skill.get("skill_id", ""), policy)


def filter_by_utility(skills, policy, store, threshold=0.0):
    """Keep only skills whose policy utility is >= threshold (order preserved)."""
    return [s for s in skills if _utility(s, policy, store) >= threshold]


def rank_by_utility(skills, policy, store, top_k=None):
    """Sort skills by descending policy utility; optionally truncate to top_k."""
    ranked = sorted(skills, key=lambda s: _utility(s, policy, store), reverse=True)
    if top_k is not None:
        ranked = ranked[:top_k]
    return ranked


def select(skills, policy, store, mode="rank", threshold=0.0, top_k=None):
    """Apply filter and/or rank.  ``mode`` is one of:

    * ``"filter"``       — threshold only (E2)
    * ``"rank"``         — sort only (E3)
    * ``"filter_rank"``  — threshold then sort
    """
    if mode not in ("filter", "rank", "filter_rank"):
        raise ValueError(f"unknown mode: {mode}")
    if mode in ("filter", "filter_rank"):
        skills = filter_by_utility(skills, policy, store, threshold)
    if mode in ("rank", "filter_rank"):
        skills = rank_by_utility(skills, policy, store, top_k)
    return skills


# --------------------------------------------------------------------------- #
# Stage 5.5 Freshness (RQ4) — pure formula (unit-testable)                     #
# --------------------------------------------------------------------------- #
# Score_i = U_i * F_i,  F_i = e^{-lambda * age},  age = current_step - t_i.
# F_i shrinks a stale estimate toward 0 but never flips its sign, so this is a
# *confidence prior*, not a sign-flip corrector (docs/stage55_freshness_design.md
# section 2).  The runtime version that selects over heterogeneous ages lives in
# ``freshskill.freshness.FreshnessScorer``.

def freshness(skill_id, policy, store, current_step, lambda_):
    """F_i = e^{-lambda * age}; age = max(0, current_step - measured_step)."""
    t = store.step(skill_id, policy)
    if t is None:
        return 1.0  # no step recorded -> treat as fresh (legacy stores)
    return math.exp(-lambda_ * max(0.0, current_step - t))


def freshness_score(skill_id, policy, store, current_step, lambda_):
    """Score_i = U_i * F_i."""
    return store.utility(skill_id, policy) * freshness(
        skill_id, policy, store, current_step, lambda_
    )
