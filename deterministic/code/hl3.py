"""High level for L lanes (used with L = 3): actions 0 = stay, 1 = change left (lane - 1), 2 = change right
(lane + 1). Same option semantics, filter, failure charges and returns as hl.py (two lanes).
"""
import numpy as np
from nn import MLP
from ppo import u_from_a
from hl import metrics, model_clone   # noqa: F401  (re-exported)
from env import LEAD, FOLL, ovm_V

STAY, LEFT, RIGHT = 0, 1, 2
DIRS = np.array([0, -1, 1])


def masks3(env):
    """(B, 3) admissible actions: stay iff I^Delta nonempty; change iff entry guard (+ dwell, horizon)."""
    return np.stack([env.stay_ok(), env.entry_ok(env.lane - 1), env.entry_ok(env.lane + 1)], 1)


def run3(env, actor, decide, c, collect=None, forced_first=None, t_end=None, max_vec_steps=100000):
    """decide(env, rows, M) -> actions in {0,1,2} for rows (M: (n,3) admissible mask)."""
    B = env.B
    live = ~env.done.copy()
    opt_left = np.zeros(B, int); R = np.zeros(B); G = np.zeros(B); a_cur = np.zeros(B, int)
    first = np.ones(B, bool)
    n_dec = np.zeros(B, int); n_choice = np.zeros(B, int)
    fail_mask_empty = np.zeros(B, bool); fail_interval = np.zeros(B, bool)
    y0 = np.zeros((B, env.critic_obs().shape[1])) if collect is not None else None
    for _ in range(max_vec_steps):
        if not live.any():
            break
        need = live & (opt_left == 0)
        if need.any():
            rows = np.where(need)[0]
            M = masks3(env)[rows]
            acts = np.asarray(decide(env, rows, M)).astype(int)
            if forced_first is not None:
                ff = forced_first[rows]; use = first[rows] & (ff >= 0)
                acts = np.where(use, ff, acts)
            # inadmissible choice -> stay if admissible, else the first admissible change
            ok = M[np.arange(len(rows)), acts]
            fallback = np.where(M[:, 0], 0, np.where(M[:, 1], 1, 2))
            acts = np.where(ok, acts, fallback)
            first[rows] = False
            n_dec[rows] += 1; n_choice[rows] += (M.sum(1) >= 2)
            if collect is not None:
                y0[rows] = env.critic_obs()[rows]
            chg = np.zeros(B, bool); chg[rows[acts > 0]] = True
            tgt = env.lane.copy(); tgt[rows] = env.lane[rows] + DIRS[acts]
            env.change_lane(chg, tgt)
            a_cur[rows] = acts
            opt_left[rows] = np.where(acts > 0, c.N, 1)
            R[rows] = np.where(acts > 0, -c.c0, 0.0)
        uc = u_from_a(actor.forward(env.actor_obs())[:, 0], c)
        u, feas, active = env.filt_cur(uc)
        newfail = live & ~feas
        r = env.step(u, active_mask=live & feas)
        env.n_filter_active = env.n_filter_active + (live & feas & active)
        R += r; R -= np.where(newfail, c.C_fail, 0.0); fail_interval |= newfail
        opt_left = np.where(live, opt_left - 1, opt_left)
        if t_end is not None:
            trunc = live & ~newfail & (env.t >= t_end - 1e-9) & (env.t < c.Tf - 1e-9)
            if trunc.any():
                G[trunc] += R[trunc]; R[trunc] = 0.0; live[trunc] = False; env.done[trunc] = True
        ended = live & ((opt_left <= 0) | newfail)
        at_T = env.t >= c.Tf - 1e-9
        if ended.any():
            rows = np.where(ended)[0]
            term = newfail[rows] | at_T[rows]
            M2 = masks3(env)[rows]
            empty = ~M2.any(1) & ~term
            R[rows] -= np.where(empty, c.C_fail, 0.0); fail_mask_empty[rows] |= empty
            z = term | empty
            if collect is not None:
                collect.append((y0[rows].copy(), a_cur[rows].copy(), R[rows].copy(), env.critic_obs()[rows], z.copy(), M2))
            G[rows] += R[rows]; R[rows] = 0.0
            fin = rows[z]; live[fin] = False; env.done[fin] = True
            env.failed[fin] = newfail[fin] | empty[z]
    return dict(G=G, fail=fail_interval | fail_mask_empty, fail_mask=fail_mask_empty, fail_int=fail_interval,
                n_dec=n_dec, n_both=n_choice)


# ---------------------------------------------------------------- rules
def rule_stay3(env, rows, M):
    return np.zeros(len(rows), int)


def make_rule_greedy3(qnet):
    """pi_b: masked argmax of the critic, ties to stay."""
    def rule(env, rows, M):
        qq = qnet.forward(env.critic_obs()[rows])
        qq = np.where(M, qq, -np.inf)
        best = np.argmax(qq, 1)
        return np.where(qq[np.arange(len(rows)), 0] >= qq[np.arange(len(rows)), best], 0, best)
    return rule


def make_rule_stay_then(base):
    """pi_s: stay whenever admissible, otherwise act as base."""
    def rule(env, rows, M):
        return np.where(M[:, 0], 0, base(env, rows, M))
    return rule


def make_rule_mobil3(actor, c, p=0.25, a_th=0.2, b_safe=4.0):
    """MOBIL for both directions with the filtered actor for the ego and the OVM for the followers."""
    def ovm_acc(env, k, xlead):
        i = np.arange(env.B); fi = FOLL[k]
        d = xlead - env.x[i, fi] - c.L_veh
        a = env.f_alpha[np.arange(env.B), k] * (ovm_V(d, env.fvmax[np.arange(env.B), k], c.ovm_dc, c.ovm_ell) - env.v[i, fi])
        return np.clip(a, -c.f_bmax, c.f_amax)

    def rule(env, rows, M):
        i = np.arange(env.B); q = env.lane; xe = env.x[:, 0]
        uc_cur = u_from_a(actor.forward(env.actor_obs(q))[:, 0], c)
        u_cur, _, _ = env.filt_cur(uc_cur)
        ao_now = ovm_acc(env, q, xe); ao_after = ovm_acc(env, q, env.x[i, LEAD[q]])
        inc = np.full((env.B, 3), -np.inf)
        for a_, dk in [(1, -1), (2, 1)]:
            t = np.clip(q + dk, 0, env.L - 1)
            uc_t = u_from_a(actor.forward(env.actor_obs(t))[:, 0], c)
            res = [env.filt(uc_t, kk) for kk in range(env.L)]
            ut = np.choose(t, [r_[0] for r_ in res]); ft = np.choose(t, [r_[1] for r_ in res])
            at_e = np.where(ft, ut, -np.inf)
            an_now = ovm_acc(env, t, env.x[i, LEAD[t]]); an_after = ovm_acc(env, t, xe)
            v = (at_e - u_cur) + p * ((an_after - an_now) + (ao_after - ao_now))
            inc[:, a_] = np.where(an_after >= -b_safe, v, -np.inf)
        inc = inc[rows]; inc = np.where(M, inc, -np.inf); inc[:, 0] = -np.inf
        best = np.argmax(inc, 1)
        return np.where(inc[np.arange(len(rows)), best] > a_th, best, 0)
    return rule


def rollout_pair3(env, r, acts, actor, base, c, model, horizon=None):
    """Advantage g of 'take acts (a change), then base' over 'stay, then pi_s' from rows r.
    Returns g, change-rollout failed flag, simulated steps."""
    sc = model_clone(env, r, 1, model); ss = model_clone(env, r, 1, model)
    n0 = sc.n_steps.copy(); m0 = ss.n_steps.copy()
    te = None if horizon is None else sc.t + horizon
    rc = run3(sc, actor, base, c, forced_first=acts, t_end=te)
    rs = run3(ss, actor, make_rule_stay_then(base), c, forced_first=np.zeros(len(r), int), t_end=te)
    steps = int((sc.n_steps - n0).sum() + (ss.n_steps - m0).sum())
    return rc['G'] - rs['G'], rc['fail'], steps


class Stats:
    def __init__(self):
        self.props = 0; self.rolled = 0; self.sim_steps = 0; self.rollouts = 0


def feats3(env):
    """Observable student features: critic features without disturbance parameters (first 26 of 41) plus
    measured accelerations of the current/left/right leaders (zero if the lane does not exist)."""
    y = env.critic_obs()[:, :26]
    _, a = env.leader_speed_acc(env.t)
    i = np.arange(env.B); q = env.lane
    acc = []
    for dk in (0, -1, 1):
        k = q + dk; ex = (k >= 0) & (k < env.L)
        acc.append(np.where(ex, a[i, np.clip(k, 0, env.L - 1)], 0.0) / 3.0)
    return np.concatenate([y, np.stack(acc, 1)], 1)


class Gate3:
    """Rollout gate for L lanes. mode: 'teacher' (exact rollouts, records labels), 'rollout' (model, horizon),
    'student' (accept iff g_hat > tau), 'cascade' (accept/reject outside +-delta, else rollouts with model).
    Proposal: pi_b proposes a change and stay is admissible. Features include the proposed direction."""
    def __init__(self, q, actor, c, mode='rollout', model='true', horizon=None, student=None, mu=None, sd=None,
                 tau=0.0, delta=None, shadow=False, record=False):
        self.base = make_rule_greedy3(q); self.actor, self.c = actor, c
        self.mode, self.model, self.horizon = mode, model, horizon
        self.student, self.mu, self.sd, self.tau, self.delta = student, mu, sd, tau, delta
        self.shadow, self.record = shadow, record
        self.st = Stats(); self.X, self.Y, self.rec = [], [], []

    def ghat(self, env, r, acts):
        f = np.concatenate([feats3(env)[r], (DIRS[acts])[:, None].astype(float)], 1)
        return self.student.forward((f - self.mu) / self.sd)[:, 0] * 10.0

    def __call__(self, env, rows, M):
        a = self.base(env, rows, M)
        cand = np.where((a > 0) & M[:, 0])[0]
        if len(cand) == 0:
            return a
        r = rows[cand]; acts = a[cand]; self.st.props += len(r)
        if self.mode in ('teacher', 'rollout'):
            model = 'true' if self.mode == 'teacher' else self.model
            g, fail, steps = rollout_pair3(env, r, acts, self.actor, self.base, self.c, model, self.horizon)
            self.st.rolled += len(r); self.st.sim_steps += steps; self.st.rollouts += 2 * len(r)
            ok = ~fail & (g >= 0.0)
            if self.record:
                self.X.append(np.concatenate([feats3(env)[r], (DIRS[acts])[:, None].astype(float)], 1)); self.Y.append(g)
        else:
            gh = self.ghat(env, r, acts)
            if self.mode == 'student':
                ok = gh > self.tau
            else:
                ok = gh >= self.delta
                band = np.abs(gh) < self.delta
                if band.any():
                    g, fail, steps = rollout_pair3(env, r[band], acts[band], self.actor, self.base, self.c, self.model)
                    ok[band] = ~fail & (g >= 0.0)
                    self.st.rolled += int(band.sum()); self.st.sim_steps += steps; self.st.rollouts += 2 * int(band.sum())
            if self.shadow:
                g_ex, f_ex, _ = rollout_pair3(env, r, acts, self.actor, self.base, self.c, 'true')
                self.rec.append(np.stack([gh, g_ex, f_ex.astype(float)], 1))
        a[cand[~ok]] = 0
        return a


def make_rule_predictive3(actor, c, model='ca', horizon=None, period=None):
    """Critic-free: compare 'stay' with 'change to each admissible adjacent lane, then stay' by rollouts
    (one rollout per admissible option); counts evaluated decisions and rollouts."""
    def rule(env, rows, M):
        out = np.zeros(len(rows), int)
        ok = M[:, 0] & (M[:, 1] | M[:, 2])
        if period is not None:
            if getattr(env, '_pred_last', None) is None or len(env._pred_last) != env.B:
                env._pred_last = np.full(env.B, -np.inf)
            ok &= env.t[rows] - env._pred_last[rows] >= period - 1e-9
        cand = np.where(ok)[0]
        rule.n_eval += len(cand)
        if len(cand) == 0:
            return out
        if period is not None:
            env._pred_last[rows[cand]] = env.t[rows[cand]]
        stay_r = make_rule_stay_then(rule_stay3)
        Gs = {}
        for a_ in (0, 1, 2):
            sel = cand if a_ == 0 else cand[M[cand, a_]]
            if len(sel) == 0:
                continue
            sub = model_clone(env, rows[sel], 1, model)
            te = None if horizon is None else sub.t + horizon
            res = run3(sub, actor, stay_r, c, forced_first=np.full(len(sel), a_), t_end=te)
            rule.n_roll += len(sel)
            Gs[a_] = (sel, res['G'])
        best = np.full(len(rows), -np.inf); choice = np.zeros(len(rows), int)
        for a_, (sel, g) in Gs.items():
            better = g > best[sel]
            best[sel] = np.where(better, g, best[sel]); choice[sel] = np.where(better, a_, choice[sel])
        out[cand] = choice[cand]
        return out
    rule.n_eval = 0; rule.n_roll = 0
    return rule
