"""High level of Algorithm 1: masked macro-option execution, Double DQN, and deployment rules.

Options: stay = 1 physics step, change = lane reset + N physics steps, both driven by the same
frozen low-level actor through the same filter. Rewards are executed integrals; c0 is charged once
per initiation; a failure (empty I^Delta inside an option, or an empty decision mask before T_f)
terminates the episode and charges C_fail once to the transition that led to it.
"""
import numpy as np
from env import Env
from nn import MLP, Adam
from ppo import u_from_a

STAY, CHANGE = 0, 1


# ============================================================ masks and helper quantities
def masks(env):
    ms = env.stay_ok()
    mc = env.entry_ok()
    return ms, mc


def target_S(env, H_star=10.0, t_star=1.0):
    """S = min_{+,-} H^±/(H* + t* [-D^±]_+) in the candidate target lane (eq:theta)."""
    k = 1 - env.lane
    out = np.full(env.B, np.inf)
    for kk in (0, 1):
        g = env.lane_view(kk)
        Sp = g['Hp'] / (H_star + t_star * np.maximum(0, -g['Dp']))
        Sm = g['Hm'] / (H_star + t_star * np.maximum(0, -g['Dm']))
        S = np.where(g['valid'], np.minimum(Sp, Sm), np.inf)
        out = np.where(k == kk, S, out)
    return np.maximum(out, 0.0)


# ============================================================ generic episode runner
def run(env, actor, decide, c, collect=None, max_vec_steps=100000, forced_first=None,
        on_decision=None, t_end=None, cf_w=0.0, cf_h=None, cf_q=None):
    """Run all rows of env (already reset) to T_f. decide(env, rows, ms, mc) -> actions for rows.
    forced_first: optional array of first actions (-1 = no forcing) applied at the first decision.
    collect: list to which (y, a, R, y2, z, m2) transitions are appended (numpy arrays).
    Returns dict of per-row metrics and returns."""
    B = env.B
    live = ~env.done.copy()
    opt_left = np.zeros(B, int)
    R = np.zeros(B); G = np.zeros(B)
    a_cur = np.zeros(B, int)
    y0 = None
    first = np.ones(B, bool)
    n_dec = np.zeros(B, int); n_both = np.zeros(B, int)
    fail_mask_empty = np.zeros(B, bool); fail_interval = np.zeros(B, bool)
    if collect is not None:
        y0 = np.zeros((B, env.critic_obs().shape[1]))
    # CF ablation (training only): counterfactual stay progress over the change-option window
    x_dec = np.zeros(B); dx_stay = np.zeros(B); cf_on = (collect is not None) and cf_w > 0.0
    for _ in range(max_vec_steps):
        if not live.any():
            break
        need = live & (opt_left == 0)
        if need.any():
            rows = np.where(need)[0]
            ms, mc = masks(env)
            ms, mc = ms[rows], mc[rows]
            acts = decide(env, rows, ms, mc)
            if forced_first is not None:
                ff = forced_first[rows]
                use = first[rows] & (ff >= 0)
                acts = np.where(use, ff, acts)
            acts = np.where(ms & ~mc, STAY, np.where(mc & ~ms, CHANGE, acts))
            first[rows] = False
            n_dec[rows] += 1; n_both[rows] += (ms & mc)
            if on_decision is not None:
                on_decision(env, rows, acts, ms, mc)
            if collect is not None:
                y0[rows] = env.critic_obs()[rows]
            chg = np.zeros(B, bool); chg[rows[acts == CHANGE]] = True
            if cf_on and chg.any() and cf_h is not None:
                # fair CF (training only): same comparison as the online gate over horizon cf_h:
                # change now then pi_b (current critic) vs stay in lane, both simulated to t + cf_h
                rc_ = np.where(chg)[0]
                pib = make_rule_margin([cf_q], c, 0.0, 0.0)
                s1 = env.subset(rc_); s1.done[:] = False; s2 = env.subset(rc_); s2.done[:] = False
                te = s1.t + cf_h
                g1 = run(s1, actor, pib, c, forced_first=np.full(len(rc_), CHANGE), t_end=te)['G']
                g2 = run(s2, actor, rule_stay_only, c, forced_first=np.full(len(rc_), STAY), t_end=te)['G']
                dx_stay[rc_] = g2; x_dec[rc_] = g1           # reuse arrays: bonus = cf_w * (g1 - g2)
            elif cf_on and chg.any():
                rc_ = np.where(chg)[0]
                cl = env.subset(rc_); cl.done[:] = False      # clone BEFORE the lane reassignment
                x0 = cl.x[:, 0].copy(); alive = np.ones(len(rc_), bool)
                for _k in range(c.N):                          # frozen actor stays N steps, same filter
                    ucl = u_from_a(actor.forward(cl.actor_obs())[:, 0], c)
                    ul, fl, _ = cl.filt_cur(ucl)
                    alive &= fl
                    cl.step(ul, active_mask=alive)
                x_dec[rc_] = env.x[rc_, 0]; dx_stay[rc_] = cl.x[:, 0] - x0
            env.change_lane(chg)
            a_cur[rows] = acts
            opt_left[rows] = np.where(acts == CHANGE, c.N, 1)
            R[rows] = np.where(acts == CHANGE, -c.c0, 0.0)
        # physics step
        uc = u_from_a(actor.forward(env.actor_obs())[:, 0], c)
        u, feas, active = env.filt_cur(uc)
        newfail = live & ~feas
        r = env.step(u, active_mask=live & feas)
        env.n_filter_active = env.n_filter_active + (live & feas & active)
        R += r
        R -= np.where(newfail, c.C_fail, 0.0)
        fail_interval |= newfail
        opt_left = np.where(live, opt_left - 1, opt_left)
        if t_end is not None:
            # truncated rollout horizon (online variant): stop at t_end, even inside an option, no charge
            trunc = live & ~newfail & (env.t >= t_end - 1e-9) & (env.t < c.Tf - 1e-9)
            if trunc.any():
                G[trunc] += R[trunc]; R[trunc] = 0.0
                live[trunc] = False; env.done[trunc] = True
        ended = live & ((opt_left <= 0) | newfail)
        at_T = env.t >= c.Tf - 1e-9
        if ended.any():
            rows = np.where(ended)[0]
            term = newfail[rows] | at_T[rows]
            ms2, mc2 = masks(env)
            empty = ~(ms2[rows] | mc2[rows]) & ~term
            R[rows] -= np.where(empty, c.C_fail, 0.0)
            fail_mask_empty[rows] |= empty
            z = term | empty
            if cf_on:
                isc = a_cur[rows] == CHANGE
                if cf_h is not None:
                    R[rows] += np.where(isc, cf_w * (x_dec[rows] - dx_stay[rows]), 0.0)
                else:
                    R[rows] += np.where(isc, cf_w * ((env.x[rows, 0] - x_dec[rows]) - dx_stay[rows]), 0.0)
            if collect is not None:
                y2 = env.critic_obs()[rows]
                m2 = np.stack([ms2[rows], mc2[rows]], 1)
                collect.append((y0[rows].copy(), a_cur[rows].copy(), R[rows].copy(), y2, z.copy(), m2))
            G[rows] += R[rows]
            R[rows] = 0.0
            fin = rows[z]
            live[fin] = False
            env.done[fin] = True
            env.failed[fin] = newfail[fin] | empty[z]
    return dict(G=G, fail=fail_interval | fail_mask_empty, fail_mask=fail_mask_empty, fail_int=fail_interval,
                n_dec=n_dec, n_both=n_both)


def metrics(env, res):
    c = env.c
    env.close_cut(np.ones(env.B, bool))          # attribute any open post-cut-in window
    prog = env.x[:, 0] - env.x_start
    T = np.maximum(env.n_steps * c.dt, c.dt)
    return dict(cut_sum=env.cut_sum.copy(), n_cut=env.n_cut.copy(), n_cut_hard=env.n_cut_hard.copy(),
                hb_time=env.hb_time.copy(), entry_gap=env.entry_gap.copy(), entry_ttc=env.entry_ttc.copy(),
                jerk_rms=np.sqrt(env.jerk2 / T), dual_fallback=env.dual_fallback.copy(),
                G=res['G'], score=prog * c.w_d - c.w_u * env.effort - c.c0 * env.n_sw,
                progress=prog, effort=env.effort, n_sw=env.n_sw.astype(float),
                min_front=env.min_front, min_rear=env.min_rear, collided=env.collided.astype(float),
                fail=res['fail'].astype(float), f_brake=env.f_brake,
                filter_active=env.n_filter_active / np.maximum(env.n_steps, 1))


# ============================================================ Q network
class QNet:
    def __init__(self, in_dim, rng, hidden=128):
        self.net = MLP([in_dim, hidden, hidden, 2], rng, out_scale=0.1)

    def q(self, y):
        """Action values in reward units (unscaled)."""
        return self.net.forward(y)


def q_values(qnet, env, rows, c):
    y = env.critic_obs()[rows]
    return qnet.forward(y) / c.rew_scale


# ============================================================ decision rules
def rule_stay_only(env, rows, ms, mc):
    return np.zeros(len(rows), int)


def make_rule_margin(qnets, c, theta0=0.0, theta1=0.0, H_star=10.0, t_star=1.0):
    """Change iff Qc - Qs > theta0 + theta1 exp(-S) (eq:rule, eq:theta). qnets: list -> mean (ensemble)."""
    def rule(env, rows, ms, mc):
        y = env.critic_obs()[rows]
        q = np.mean([qn.forward(y) for qn in qnets], 0) / c.rew_scale
        th = theta0
        if theta1 != 0.0:
            S = target_S(env, H_star, t_star)[rows]
            th = theta0 + theta1 * np.exp(-S)
        return np.where(q[:, 1] - q[:, 0] > th, CHANGE, STAY)
    return rule


def make_rule_mobil(actor, c, p=0.25, a_th=0.2, b_safe=4.0):
    """MOBIL (Kesting et al.) with the same longitudinal controller (filtered learned actor) for the
    ego and the OVM for the followers; hard guard/dwell/horizon enforced by the runner's mask."""
    from env import LEAD, FOLL, ovm_V
    def ovm_acc(env, k, xlead):
        i = np.arange(env.B); fi = FOLL[k]
        d = xlead - env.x[i, fi] - c.L_veh
        a = env.f_alpha[:, k] * (ovm_V(d, env.fvmax[:, k], c.ovm_dc, c.ovm_ell) - env.v[i, fi])
        return np.clip(a, -c.f_bmax, c.f_amax)
    def rule(env, rows, ms, mc):
        i = np.arange(env.B)
        q = env.lane; t = 1 - q
        uc_cur = u_from_a(actor.forward(env.actor_obs(q))[:, 0], c)
        uc_tgt = u_from_a(actor.forward(env.actor_obs(t))[:, 0], c)
        a_e = np.zeros(env.B); at_e = np.zeros(env.B)
        for kk in (0, 1):
            u_k, f_k, _ = env.filt(uc_cur if True else None, kk)
            a_e = np.where(q == kk, u_k, a_e)
            u_t, f_t, _ = env.filt(uc_tgt, kk)
            at_e = np.where(t == kk, np.where(f_t, u_t, -np.inf), at_e)
        xe = env.x[:, 0]
        a_n = np.zeros(env.B); at_n = np.zeros(env.B); a_o = np.zeros(env.B); at_o = np.zeros(env.B)
        for kk in (0, 1):
            lead_x = env.x[i, LEAD[kk]]
            # new follower n in target lane kk
            an_now = ovm_acc(env, kk, lead_x); an_after = ovm_acc(env, kk, xe)
            # old follower o in current lane kk
            ao_now = ovm_acc(env, kk, xe); ao_after = ovm_acc(env, kk, lead_x)
            a_n = np.where(t == kk, an_now, a_n); at_n = np.where(t == kk, an_after, at_n)
            a_o = np.where(q == kk, ao_now, a_o); at_o = np.where(q == kk, ao_after, at_o)
        inc = (at_e - a_e) + p * ((at_n - a_n) + (at_o - a_o))
        ok = (inc > a_th) & (at_n >= -b_safe)
        return np.where(ok[rows], CHANGE, STAY)
    return rule


def clone_rows(env, rows, rep):
    return env.subset(rows, repeat=rep)


def make_rule_predictive(actor, c, model='true', horizon=None, period=None):
    """Deterministic common-horizon predictive selector: compare 'stay in lane to T_f' with
    'change now, then stay to T_f' by full rollouts (same filter and mask), under the given model.
    horizon: truncate both rollouts at t + horizon. period: only evaluate at decisions with
    t in {0, period, 2 period, ...} (matched-compute variant); otherwise stay."""
    def rule(env, rows, ms, mc):
        out = np.zeros(len(rows), int)
        ok = mc & ms
        if period is not None:
            # at most one evaluation per period: first admissible decision >= period after the last one
            if getattr(env, '_pred_last', None) is None or len(env._pred_last) != env.B:
                env._pred_last = np.full(env.B, -np.inf)
            ok &= env.t[rows] - env._pred_last[rows] >= period - 1e-9
        cand = np.where(ok)[0]
        if period is not None:
            env._pred_last[rows[cand]] = env.t[rows[cand]]
        rule.n_eval += len(cand)
        if len(cand) == 0:
            return out
        sub = model_clone(env, rows[cand], 2, model)
        ff = np.tile([STAY, CHANGE], len(cand))
        te = None if horizon is None else sub.t + horizon
        res = run(sub, actor, rule_stay_only, c, forced_first=ff, t_end=te)
        Gs = res['G'][0::2]; Gc = res['G'][1::2]
        out[cand] = np.where(Gc > Gs, CHANGE, STAY)
        return out
    rule.n_eval = 0
    return rule


ROBUST_CASES = [(0.0, 0.0), (-1.0, -1.0), (-1.0, 1.0), (1.0, -1.0), (1.0, 1.0)]


def model_clone(env, rows, rep, model, cases=None):
    """Clone rows (each repeated rep times) and, unless model == 'true', replace the other vehicles'
    future by a prediction model: 'pred' = constant-speed leaders + nominal OVM followers;
    'robust' = constant leader accelerations at the declared bounds (cases repeat within rep)."""
    sub = env.subset(rows, repeat=rep); sub.done[:] = False
    if model == 'pred':
        sub.to_model('pred')
    elif model == 'ca':
        sub.to_model('ca')
    elif model == 'robust':
        K = len(cases); per = rep // K
        ac = np.tile(np.repeat([cc[0] for cc in cases], per), len(rows))
        at = np.tile(np.repeat([cc[1] for cc in cases], per), len(rows))
        sub.to_model('case', ac, at)
    return sub


class BudgetGovernor:
    """Baseline-relative budget governor (Sec. IV-C) with the two-rollout certificate.
    Baseline pi_b = frozen unthresholded masked-greedy critic (ties -> stay).
    model: 'true'   rollouts use the exact declared model, incl. the realised leader profiles;
           'pred'   rollouts use a prediction (constant-speed leaders, nominal OVM followers);
           'robust' price = worst case over leader accelerations at the declared bounds."""
    def __init__(self, qnets, actor, c, B0=10.0, beta0=0.0, beta1=10.0, H_star=10.0, t_star=1.0, model='true',
                 horizon=None):
        self.horizon = horizon            # None = to T_f; else rollouts truncated to t + horizon
        self.base = make_rule_margin(qnets, c, 0.0, 0.0)
        self.actor, self.c = actor, c
        self.B0, self.beta0, self.beta1, self.Hs, self.ts = B0, beta0, beta1, H_star, t_star
        self.model = model
        self.cases = ROBUST_CASES if model == 'robust' else [(0.0, 0.0)]
        self.budget = None
        self.stats = dict(proposals=0, accepted=0, debit=0.0, base_infeasible=0)

    def reset(self, B):
        self.budget = np.full(B, float(self.B0))
        self.base_ok = np.ones(B, bool)

    def check_initial(self, env):
        """Verify a complete feasible baseline rollout from the initial state (per row, all cases)."""
        K = len(self.cases)
        sub = model_clone(env, np.arange(env.B), K, self.model, self.cases)
        te = None if self.horizon is None else sub.t + self.horizon
        res = run(sub, self.actor, self.base, self.c, t_end=te)
        self.base_ok = ~res['fail'].reshape(env.B, K).any(1)
        self.stats['base_infeasible'] += int((~self.base_ok).sum())
        return res

    def __call__(self, env, rows, ms, mc):
        c = self.c
        a = self.base(env, rows, ms, mc)
        a = np.where(ms & ~mc, STAY, np.where(mc & ~ms, CHANGE, a))
        cand = np.where((a == CHANGE) & ms & self.base_ok[rows])[0]
        if len(cand) == 0:
            return a
        S = target_S(env, self.Hs, self.ts)[rows[cand]]
        allow = self.beta0 + self.beta1 * np.exp(-S)
        cand2 = cand[allow > 1e-12]
        if len(cand2) == 0:
            return a
        K = len(self.cases)
        sub = model_clone(env, rows[cand2], 2 * K, self.model, self.cases)
        ff = np.tile([CHANGE, STAY], len(cand2) * K)
        te = None if self.horizon is None else sub.t + self.horizon
        res = run(sub, self.actor, self.base, c, forced_first=ff, t_end=te)
        Gb = res['G'][0::2].reshape(-1, K); Gs = res['G'][1::2].reshape(-1, K)
        fb = res['fail'][0::2].reshape(-1, K); fs = res['fail'][1::2].reshape(-1, K)
        cert = np.maximum(0.0, Gb - Gs).max(1)          # worst case over the cases (same w per pair)
        feas = ~(fb.any(1) | fs.any(1))
        r = rows[cand2]
        allow2 = (self.beta0 + self.beta1 * np.exp(-target_S(env, self.Hs, self.ts)[r]))
        acc = feas & (cert <= np.minimum(self.budget[r], allow2))
        self.stats['proposals'] += len(cand2); self.stats['accepted'] += int(acc.sum())
        self.stats['debit'] += float(cert[acc].sum())
        self.budget[r[acc]] -= cert[acc]
        a[cand2[acc]] = STAY
        return a


class PredictionGate:
    """Rollout gate (eq:gate): when pi_b proposes a change and stay is admissible, simulate (i) change
    now then pi_b and (ii) stay in lane to T_f; change iff G_i - G_ii >= rho * max(G_ii, 0).
    Improvement guarantee: Prop. (rollout improvement). model='robust' requires the inequality in every
    disturbance case (common case in both branches), i.e. the guarantee then holds over the case set."""
    def __init__(self, qnets, actor, c, rho=0.02, model='true', horizon=None, shadow=False):
        self.base = make_rule_margin(qnets, c, 0.0, 0.0)
        self.actor, self.c, self.rho, self.model, self.horizon = actor, c, rho, model, horizon
        self.cases = ROBUST_CASES if model == 'robust' else [(0.0, 0.0)]
        self.stats = dict(proposals=0, accepted=0)
        # shadow: also run exact full-horizon rollouts and accumulate, per row, the error eps_k of the
        # compared difference (Prop. forecast error) and the rho-term; diagnostics only, decisions unchanged
        self.shadow = shadow; self.eps = None; self.rho_term = None

    def __call__(self, env, rows, ms, mc):
        c = self.c
        a = self.base(env, rows, ms, mc)
        a = np.where(ms & ~mc, STAY, np.where(mc & ~ms, CHANGE, a))
        cand = np.where((a == CHANGE) & ms)[0]
        if len(cand) == 0:
            return a
        r = rows[cand]; K = len(self.cases)
        sc = model_clone(env, r, K, self.model, self.cases); ss = model_clone(env, r, K, self.model, self.cases)
        te = None if self.horizon is None else sc.t + self.horizon
        rc = run(sc, self.actor, self.base, c, forced_first=np.full(len(r) * K, CHANGE), t_end=te)
        rs = run(ss, self.actor, rule_stay_only, c, forced_first=np.full(len(r) * K, STAY), t_end=te)
        gain = (rc['G'] - rs['G']).reshape(-1, K)
        thr = (self.rho * np.maximum(rs['G'], 0.0)).reshape(-1, K)
        ok = ~rc['fail'].reshape(-1, K).any(1) & (gain >= thr).all(1)
        self.stats['proposals'] += len(r); self.stats['accepted'] += int(ok.sum())
        if self.shadow and K == 1:
            if self.eps is None or len(self.eps) != env.B:
                self.eps = np.zeros(env.B); self.rho_term = np.zeros(env.B)
            xc = model_clone(env, r, 1, 'true'); xs = model_clone(env, r, 1, 'true')
            gc = run(xc, self.actor, self.base, c, forced_first=np.full(len(r), CHANGE))['G']
            gs = run(xs, self.actor, rule_stay_only, c, forced_first=np.full(len(r), STAY))['G']
            self.eps[r] += np.abs(gain[:, 0] - (gc - gs))
            self.rho_term[r] += self.rho * np.maximum(rs['G'], 0.0)
        a[cand[~ok]] = STAY
        return a
