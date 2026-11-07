"""Initial-state / disturbance generators. Each family fills rows of an Env at t = 0.

All families use the sinusoid leader family (eq:leader) with 0 <= A <= min(vbar, vmax - vbar),
so the declared leader bound -A|w| is valid. Initial states outside the occupied-lane domain
(H>=0, psi>=0, speed bounds, nonempty I^Delta) are rejected and resampled; rejections are counted.
"""
import numpy as np
from env import LEAD, FOLL

FAMILIES = ['random', 'equal', 'faster_target', 'fast_follower', 'leader_braking', 'squeeze', 'late']
TRAIN_MIX = {'random': 0.4, 'equal': 0.1, 'faster_target': 0.1, 'fast_follower': 0.1,
             'leader_braking': 0.1, 'squeeze': 0.1, 'late': 0.1}


def _amp(vbar, vmax, cap, rng, lo=0.0):
    hi = np.minimum(np.minimum(vbar, vmax - vbar), cap)
    return rng.uniform(np.minimum(lo, hi), hi)


def sample_params(family, rng, cfg):
    """Return a dict of scalars for one scenario (lane-relative: 'c' current, 't' target)."""
    U = rng.uniform
    L = cfg.L_veh
    p = {}
    ve = U(8, 15)
    p['ve'] = ve
    def leader(vbar, A=None, w=None, phi=None, Acap=3.0):
        vbar = float(np.clip(vbar, 4.0, cfg.vmax - 1.0))
        A = _amp(vbar, cfg.vmax, Acap, rng) if A is None else min(A, vbar, cfg.vmax - vbar)
        return dict(vbar=vbar, A=A, w=U(0.2, 0.7) if w is None else w,
                    phi=U(0, 2 * np.pi) if phi is None else phi)
    if family in ('random',):
        p['cl'] = leader(U(6, 18)); p['tl'] = leader(U(6, 18))
        p['cdp'] = U(6, 45); p['cdm'] = U(6, 40); p['cvf'] = U(6, 16)
        p['tdp'] = U(3, 50); p['tdm'] = U(3, 40); p['tvf'] = U(6, 18)
    elif family == 'equal':
        cl = leader(U(8, 16))
        p['cl'] = cl; p['tl'] = dict(cl)
        p['cdp'] = p['tdp'] = U(10, 40); p['cdm'] = p['tdm'] = U(10, 35)
        p['cvf'] = p['tvf'] = ve + U(-1, 1)
    elif family == 'faster_target':
        vb = U(7, 12)
        p['cl'] = leader(vb, Acap=1.0); p['tl'] = leader(vb + U(3, 6), Acap=1.0)
        p['cdp'] = U(10, 30); p['cdm'] = U(10, 30); p['cvf'] = ve + U(-1, 1)
        p['tdp'] = U(25, 50); p['tdm'] = U(20, 40); p['tvf'] = ve + U(-1, 1)
    elif family == 'fast_follower':
        vb = U(7, 12)
        p['cl'] = leader(vb, Acap=1.0); p['tl'] = leader(vb + U(3, 6), Acap=1.0)
        p['cdp'] = U(10, 30); p['cdm'] = U(10, 30); p['cvf'] = ve + U(-1, 1)
        p['tdp'] = U(25, 50); p['tdm'] = U(5, 14); p['tvf'] = ve + U(2, 5)
    elif family == 'leader_braking':
        vb = U(9, 14)
        w = U(0.5, 0.8)
        # phase chosen so that the leader is decelerating within the first ~2 s
        p['cl'] = leader(vb, A=U(3, 5), w=w, phi=U(0.0, 1.2))
        p['tl'] = leader(vb + U(-1, 2), Acap=1.0)
        p['cdp'] = U(12, 30); p['cdm'] = U(12, 30); p['cvf'] = ve + U(-1, 1)
        p['tdp'] = U(15, 45); p['tdm'] = U(10, 35); p['tvf'] = ve + U(-1, 1)
    elif family == 'squeeze':
        vb = U(9, 14)
        w = U(0.5, 0.8)
        p['cl'] = leader(vb, A=U(3, 5), w=w, phi=U(0.0, 1.2))
        p['tl'] = leader(vb + U(-1, 2), Acap=1.0)
        p['cdp'] = U(10, 22); p['cdm'] = U(5, 10); p['cvf'] = ve + U(0.5, 2.5)
        p['tdp'] = U(15, 45); p['tdm'] = U(8, 30); p['tvf'] = ve + U(-1, 1)
    elif family == 'late':
        vb = U(9, 13)
        w = np.pi / U(11, 14)            # target leader slowest at t=0, fastest at t ~ 11-14 s
        p['cl'] = leader(vb, Acap=0.8)
        p['tl'] = leader(vb + U(0, 1.5), A=U(3, 4), w=w, phi=-np.pi / 2)
        p['cdp'] = U(12, 30); p['cdm'] = U(12, 30); p['cvf'] = ve + U(-1, 1)
        p['tdp'] = U(15, 35); p['tdm'] = U(12, 30); p['tvf'] = ve + U(-1, 1)
    else:
        raise ValueError(family)
    p['cvf'] = float(np.clip(p['cvf'], 0.5, cfg.vmax - 1)); p['tvf'] = float(np.clip(p['tvf'], 0.5, cfg.vmax - 1))
    p['cfvmax'] = U(15, 20); p['tfvmax'] = U(15, 20)
    if family == 'equal':
        p['tfvmax'] = p['cfvmax']
    if family == 'fast_follower':
        p['tfvmax'] = max(p['tfvmax'], p['tvf'] + 1.0)
    p['family'] = family
    return p


def write_row(env, r, p, q0):
    c = env.c
    L = c.L_veh
    k_c, k_t = q0, 1 - q0
    env.x[r] = 0.0; env.v[r] = 0.0
    env.x[r, 0] = 0.0; env.v[r, 0] = p['ve']
    for k, key, dp, dm, vf, fv in ((k_c, 'cl', p['cdp'], p['cdm'], p['cvf'], p['cfvmax']),
                                   (k_t, 'tl', p['tdp'], p['tdm'], p['tvf'], p['tfvmax'])):
        ld = p[key]
        env.vbar[r, k] = ld['vbar']; env.A[r, k] = ld['A']; env.w[r, k] = ld['w']; env.phi[r, k] = ld['phi']
        env.x[r, LEAD[k]] = dp + L; env.xl0[r, k] = dp + L
        env.v[r, LEAD[k]] = ld['vbar'] + ld['A'] * np.sin(ld['phi'])
        env.x[r, FOLL[k]] = -(dm + L); env.v[r, FOLL[k]] = vf
        env.fvmax[r, k] = fv
    env.t[r] = 0.0; env.lane[r] = q0; env.eta[r] = 0.0
    env.pulse_t[r] = np.inf; env.pulse_a[r] = 0; env.pulse_T[r] = 0; env.pulse_v[r] = 0; env.pulse_x[r] = 0
    env.f_alpha[r] = c.ovm_alpha
    env.done[r] = False; env.failed[r] = False
    env.min_front[r] = np.inf; env.min_rear[r] = np.inf; env.collided[r] = False; env.n_sw[r] = 0
    env.effort[r] = 0; env.x_start[r] = 0.0; env.f_brake[r] = 0; env.n_filter_active[r] = 0; env.n_steps[r] = 0
    env.lmode[r] = 0; env.lacc[r] = 0; env.lacc_until[r] = np.inf
    env.dual_left[r] = 0; env.old_lane[r] = 0; env.dual_fallback[r] = 0; env.u_hold[r] = 0; env.hold_on[r] = False
    env.cut_open[r] = False; env.cut_since[r] = 0; env.cut_peak[r] = 0; env.cut_sum[r] = 0
    env.n_cut[r] = 0; env.n_cut_hard[r] = 0; env.hb_time[r] = 0
    env.entry_gap[r] = np.inf; env.entry_ttc[r] = np.inf; env.u_prev[r] = np.nan; env.jerk2[r] = 0


def in_domain(env, r):
    """Occupied-lane domain test for row r (H>=0, psi>=0, speeds, nonempty I^Delta)."""
    c = env.c
    sub = env.subset([r])
    lo_s, hi_s, _, _, g = sub.intervals(int(sub.lane[0]))
    psi_p = g['Dp'] + c.lam_f * g['Hp']; psi_m = g['Dm'] + c.lam_r * g['Hm']
    ok = g['valid'][0] and g['Hp'][0] >= 0 and g['Hm'][0] >= 0 and psi_p[0] >= 0 and psi_m[0] >= 0 \
        and lo_s[0] <= hi_s[0] and 0 <= sub.v[0, 0] <= c.vmax
    return bool(ok)


class Sampler:
    def __init__(self, cfg, seed, mix=None):
        self.cfg = cfg
        self.rng = np.random.default_rng(seed)
        self.mix = mix or TRAIN_MIX
        self.n_reject = 0; self.n_accept = 0

    def draw_family(self):
        fams = list(self.mix); pr = np.array([self.mix[f] for f in fams]); pr = pr / pr.sum()
        return fams[self.rng.choice(len(fams), p=pr)]

    def reset_rows(self, env, rows, family=None):
        for r in np.atleast_1d(rows):
            fam = family or self.draw_family()
            while True:
                p = sample_params(fam, self.rng, self.cfg)
                q0 = int(self.rng.integers(2))
                write_row(env, r, p, q0)
                if in_domain(env, r):
                    self.n_accept += 1
                    break
                self.n_reject += 1


def make_test_set(cfg, family, n, seed):
    """Fixed list of (params, q0) for held-out evaluation."""
    from env import Env
    rng = np.random.default_rng(seed)
    out = []; rej = 0
    e = Env(cfg, 1)
    while len(out) < n:
        p = sample_params(family, rng, cfg); q0 = int(rng.integers(2))
        write_row(e, 0, p, q0)
        if in_domain(e, 0):
            out.append((p, q0))
        else:
            rej += 1
    return out, rej


def load_set(env, scen):
    for r, (p, q0) in enumerate(scen):
        write_row(env, r, p, q0)
