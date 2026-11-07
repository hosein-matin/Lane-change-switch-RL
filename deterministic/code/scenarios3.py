"""Scenario families for L = 3 lanes (lanes 0, 1, 2; the ego may start in any lane).
Same leader family (eq:leader) and OVM followers as the two-lane model; lane-specific parameters.
Families: random, equal, faster_adjacent, faster_far (best lane two changes away, slower lane between),
leader_braking, squeeze, late.
"""
import numpy as np
from env import LEAD, FOLL, Env
from scenarios import in_domain

FAMILIES3 = ['random', 'equal', 'faster_adjacent', 'faster_far', 'leader_braking', 'squeeze', 'late']
TRAIN_MIX3 = {'random': 0.4, 'equal': 0.1, 'faster_adjacent': 0.1, 'faster_far': 0.1,
              'leader_braking': 0.1, 'squeeze': 0.1, 'late': 0.1}


def _amp(vbar, vmax, cap, rng):
    hi = min(vbar, vmax - vbar, cap)
    return rng.uniform(0.0, max(hi, 0.0))


def sample_params3(family, rng, cfg):
    U = rng.uniform
    vmax = cfg.vmax
    ve = U(8, 15)

    def leader(vbar, A=None, w=None, phi=None, Acap=3.0):
        vbar = float(np.clip(vbar, 4.0, vmax - 1.0))
        A = _amp(vbar, vmax, Acap, rng) if A is None else min(A, vbar, vmax - vbar)
        return dict(vbar=vbar, A=A, w=U(0.2, 0.7) if w is None else w, phi=U(0, 2 * np.pi) if phi is None else phi)

    def lane(ld, dp, dm, vf, fv=None):
        return dict(ld=ld, dp=dp, dm=dm, vf=float(np.clip(vf, 0.5, vmax - 1)), fv=U(15, 20) if fv is None else fv)

    q0 = int(rng.integers(3))
    lanes = [None] * 3
    if family == 'random':
        lanes = [lane(leader(U(6, 18)), U(6, 45), U(6, 40), U(6, 16)) for _ in range(3)]
    elif family == 'equal':
        ld = leader(U(8, 16)); dp = U(10, 40); dm = U(10, 35); vf = ve + U(-1, 1); fv = U(15, 20)
        lanes = [lane(dict(ld), dp, dm, vf, fv) for _ in range(3)]
    elif family == 'faster_adjacent':
        vb = U(7, 12)
        adj = [p for p in (q0 - 1, q0 + 1) if 0 <= p < 3]
        fast = adj[int(rng.integers(len(adj)))]
        for k in range(3):
            if k == fast:
                lanes[k] = lane(leader(vb + U(3, 6), Acap=1.0), U(25, 50), U(20, 40), ve + U(-1, 1))
            else:
                lanes[k] = lane(leader(vb + U(-1, 0.5), Acap=1.0), U(10, 30), U(10, 30), ve + U(-1, 1))
    elif family == 'faster_far':
        q0 = int(rng.choice([0, 2])); far = 2 - q0; vb = U(8, 12)
        for k in range(3):
            if k == far:
                lanes[k] = lane(leader(vb + U(4, 7), Acap=1.0), U(25, 50), U(20, 40), ve + U(-1, 1))
            elif k == 1:
                lanes[k] = lane(leader(vb - U(0, 1.5), Acap=1.0), U(20, 40), U(15, 35), ve + U(-1, 1))
            else:
                lanes[k] = lane(leader(vb, Acap=1.0), U(10, 30), U(10, 30), ve + U(-1, 1))
    elif family in ('leader_braking', 'squeeze'):
        vb = U(9, 14); w = U(0.5, 0.8)
        for k in range(3):
            if k == q0:
                if family == 'leader_braking':
                    lanes[k] = lane(leader(vb, A=U(3, 5), w=w, phi=U(0.0, 1.2)), U(12, 30), U(12, 30), ve + U(-1, 1))
                else:
                    lanes[k] = lane(leader(vb, A=U(3, 5), w=w, phi=U(0.0, 1.2)), U(10, 22), U(5, 10), ve + U(0.5, 2.5))
            else:
                lanes[k] = lane(leader(vb + U(-1, 2), Acap=1.0), U(15, 45), U(8, 30) if family == 'squeeze' else U(10, 35), ve + U(-1, 1))
    elif family == 'late':
        vb = U(9, 13)
        adj = [p for p in (q0 - 1, q0 + 1) if 0 <= p < 3]
        lt = adj[int(rng.integers(len(adj)))]
        for k in range(3):
            if k == lt:
                lanes[k] = lane(leader(vb + U(0, 1.5), A=U(3, 4), w=np.pi / U(11, 14), phi=-np.pi / 2), U(15, 35), U(12, 30), ve + U(-1, 1))
            else:
                lanes[k] = lane(leader(vb, Acap=0.8), U(12, 30), U(12, 30), ve + U(-1, 1))
    else:
        raise ValueError(family)
    return dict(family=family, ve=ve, q0=q0, lanes=lanes)


def write_row3(env, r, p):
    c = env.c; L = c.L_veh
    env.x[r] = 0.0; env.v[r] = 0.0; env.v[r, 0] = p['ve']
    for k, ln in enumerate(p['lanes']):
        ld = ln['ld']
        env.vbar[r, k] = ld['vbar']; env.A[r, k] = ld['A']; env.w[r, k] = ld['w']; env.phi[r, k] = ld['phi']
        env.x[r, LEAD[k]] = ln['dp'] + L; env.xl0[r, k] = ln['dp'] + L
        env.v[r, LEAD[k]] = ld['vbar'] + ld['A'] * np.sin(ld['phi'])
        env.x[r, FOLL[k]] = -(ln['dm'] + L); env.v[r, FOLL[k]] = ln['vf']
        env.fvmax[r, k] = max(ln['fv'], ln['vf'] + 1.0)
    env.t[r] = 0.0; env.lane[r] = p['q0']; env.eta[r] = 0.0
    env.pulse_t[r] = np.inf; env.pulse_a[r] = 0; env.pulse_T[r] = 0; env.pulse_v[r] = 0; env.pulse_x[r] = 0
    env.f_alpha[r] = c.ovm_alpha
    env.done[r] = False; env.failed[r] = False
    env.min_front[r] = np.inf; env.min_rear[r] = np.inf; env.collided[r] = False; env.n_sw[r] = 0
    env.effort[r] = 0; env.x_start[r] = 0.0; env.f_brake[r] = 0; env.n_filter_active[r] = 0; env.n_steps[r] = 0
    env.lmode[r] = 0; env.lacc[r] = 0; env.lacc_until[r] = np.inf; env.sch_t[r] = np.inf; env.sch_a[r] = 0
    env.dual_left[r] = 0; env.old_lane[r] = 0; env.dual_fallback[r] = 0; env.u_hold[r] = 0; env.hold_on[r] = False
    env.cut_open[r] = False; env.cut_since[r] = 0; env.cut_peak[r] = 0; env.cut_sum[r] = 0
    env.n_cut[r] = 0; env.n_cut_hard[r] = 0; env.hb_time[r] = 0
    env.entry_gap[r] = np.inf; env.entry_ttc[r] = np.inf; env.u_prev[r] = np.nan; env.jerk2[r] = 0


class Sampler3:
    def __init__(self, cfg, seed, mix=None):
        self.cfg = cfg; self.rng = np.random.default_rng(seed); self.mix = mix or TRAIN_MIX3
        self.n_reject = 0; self.n_accept = 0

    def reset_rows(self, env, rows, family=None):
        fams = list(self.mix); pr = np.array([self.mix[f] for f in fams]); pr = pr / pr.sum()
        for r in np.atleast_1d(rows):
            fam = family or fams[self.rng.choice(len(fams), p=pr)]
            while True:
                write_row3(env, r, sample_params3(fam, self.rng, self.cfg))
                if in_domain(env, r):
                    self.n_accept += 1; break
                self.n_reject += 1


def make_test_set3(cfg, family, n, seed):
    rng = np.random.default_rng(seed); out = []; rej = 0; e = Env(cfg, 1)
    while len(out) < n:
        p = sample_params3(family, rng, cfg); write_row3(e, 0, p)
        if in_domain(e, 0):
            out.append(p)
        else:
            rej += 1
    return out, rej


def load_set3(env, scen):
    for r, p in enumerate(scen):
        write_row3(env, r, p)
