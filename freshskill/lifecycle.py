"""Skill Lifecycle (Stage 5.6 / RQ4): RE-MEASURE + KEEP/RETIRE with
confirm-before-retire.

Freshness (Stage 5.5) only demotes stale estimates at *retrieval* time; it
cannot remove a skill whose utility has turned persistently negative.  The
Maintenance layer decides *what the bank contains*, on a periodic offline
schedule:

    retire_i = (U_i(t) < theta_retire) AND (U_i(t - Delta) < theta_retire)

i.e. a skill is retired only if it is harmful at two consecutive maintenance
cycles.  A single negative snapshot is not enough — the skill may recover
(gen_004/005 in the real data).  See docs/stage56_lifecycle_design.md.

v0.1 scope is pruning only (RE-MEASURE + KEEP + RETIRE, "自然选择" half of
evolution).  REWRITE/MERGE and the closed loop (train with the cleaner bank)
are future work (design doc section 4 / experiment_design.md part 4).
"""

KEEP = "KEEP"
RETIRE = "RETIRE"


class LifecyclePolicy:
    def __init__(self, theta_retire=-0.02, theta_keep=0.02):
        self.theta_retire = theta_retire
        self.theta_keep = theta_keep  # only used for re-measure priority (doc 4.2)

    # ---- confirm-before-retire -----------------------------------------
    def is_pending(self, utility):
        """pending_i = (U_i < theta_retire) — the per-cycle harmful flag."""
        return utility < self.theta_retire

    def maintain(self, timeline, current_step, prev_step=None, budget=None,
                 theta_retire=None, skill_ids=None):
        """Run one maintenance cycle at ``current_step``.

        Returns ``{skill_id: action}`` with action in {KEEP, RETIRE}, for the
        skills selected for re-measurement this cycle (all of them when
        ``budget`` is None — the L1/L2 setting).

        ``prev_step`` is the previous maintenance cycle's step, used to evaluate
        the second half of the confirm rule.  When None this is the *first*
        cycle: nothing can be confirmed, so every re-measured skill is KEEP (the
        caller records ``pending`` and retires only on a later cycle).

        A skill with no utility recorded at ``prev_step``/``current_step`` reads
        as 0.0, which is never below a negative threshold — so an unmeasured
        skill is conservatively kept, never retired on a single snapshot.
        """
        theta = self.theta_retire if theta_retire is None else theta_retire
        ids = list(timeline.skill_ids() if skill_ids is None else skill_ids)
        if budget is not None:
            ids = self.select_remeasure(ids, current_step, timeline, budget)

        actions = {}
        for sid in ids:
            u_now = timeline.utility(sid, current_step)
            if prev_step is None:
                actions[sid] = KEEP          # first cycle: record, don't retire
                continue
            u_prev = timeline.utility(sid, prev_step)
            if u_now < theta and u_prev < theta:
                actions[sid] = RETIRE
            else:
                actions[sid] = KEEP
        return actions

    # ---- re-measure budget / priority (design doc 4.3) -----------------
    def remeasure_priority(self, skill_id, current_step, timeline, n_tasks=140):
        """priority_i = staleness_i * impact_i.

        staleness_i = current_step - last_eval_step  (floored at 1)
        impact_i    = (usage / n_tasks) * |U_i(last_eval_step)|
        """
        t_i = timeline.latest_le(skill_id, current_step)
        if t_i is None:
            t_i = current_step                    # never measured -> treat as fresh
        staleness = max(1.0, current_step - t_i)
        u = timeline.utility(skill_id, t_i)
        counts = timeline.counts(skill_id, t_i) or {}
        impact = (counts.get("usage", 0) / n_tasks) * abs(u)
        return staleness * impact

    def select_remeasure(self, skill_ids, current_step, timeline, budget):
        """The ``budget`` skills with the highest re-measure priority."""
        scored = sorted(
            ((self.remeasure_priority(sid, current_step, timeline), sid)
             for sid in skill_ids),
            reverse=True,
        )
        return [sid for _, sid in scored[:budget]]
