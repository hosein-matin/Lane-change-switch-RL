"""Vectorised two-lane, five-vehicle longitudinal hybrid model (paper Secs. II-III).

Vehicle indices: 0 ego, 1 leader lane 0, 2 follower lane 0, 3 leader lane 1, 4 follower lane 1.
Lane k has leader index LEAD[k] and follower index FOLL[k].

Ego: exact held-input update  v+ = v + u dt,  x+ = x + v dt + u dt^2/2.
Leaders: analytic sinusoid  v = vbar + A sin(w t + phi)   (eq:leader), optional braking pulses
         (used only in the model-mismatch test; then the declared bound is violated on purpose).
Followers: saturated OVM, explicit substeps with exact position update inside each substep.
Each follower follows the nearest vehicle ahead in its lane (ego if the ego occupies the lane).

Safety (Sec. III): sampled held-input interval I^Delta (Prop. 2) intersected, when compatible, with
the sample-time HOCBF interval I_q (eq:front/rear). Executed input is always in I^Delta, so Prop. 2
applies; if I^Delta is empty the step is a feasibility failure.
"""
import numpy as np

# lane k has leader 1 + 2k and follower 2 + 2k (up to 3 lanes)
LEAD = np.array([1, 3, 5])
FOLL = np.array([2, 4, 6])


def ovm_V(d, vmaxf, dc, ell):
    d = np.maximum(d, 0.0)
    t0 = np.tanh(dc / ell)
    return vmaxf * (np.tanh((d - dc) / ell) + t0) / (1.0 + t0)


def m_fun(H, D, dt):
    """m(H,D,Delta) = inf_{0<s<=Delta} 2(H+Ds)/s^2  (eq:mdef), H >= 0, vectorised."""
    H = np.asarray(H, float); D = np.asarray(D, float)
    out = 2.0 * (H + D * dt) / dt**2
    with np.errstate(divide='ignore', invalid='ignore'):
        inner = (H > 0) & (D < 0) & (-2.0 * H / np.where(D < 0, D, -1.0) <= dt)
        out = np.where(inner, -D**2 / (2.0 * np.where(H > 0, H, 1.0)), out)
    out = np.where((H <= 0) & (D < 0), -np.inf, out)
    return out


class Env:
    def __init__(self, cfg, B, rng=None):
        self.c = cfg
        self.B = B
        self.rng = rng if rng is not None else np.random.default_rng(0)
        z = lambda *s: np.zeros((B,) + s)
        L = self.L = cfg.n_lanes
        self.x = z(1 + 2 * L); self.v = z(1 + 2 * L)
        self.t = z(); self.lane = np.zeros(B, int)
        self.eta = z()                  # elapsed-since-change clock
        self.vbar = z(L); self.A = z(L); self.w = z(L); self.phi = z(L); self.xl0 = z(L)
        self.fvmax = z(L)               # OVM desired-speed caps of followers (per lane)
        self.f_alpha = np.full((B, L), cfg.ovm_alpha)   # TRUE follower gains (mismatch test changes)
        self.pulse_t = np.full((B, L), np.inf); self.pulse_a = z(L); self.pulse_T = z(L)
        self.pulse_v = z(L)             # accumulated speed offset from pulses
        self.pulse_x = z(L)
        self.done = np.zeros(B, bool)
        self.failed = np.zeros(B, bool)
        # episode metrics
        self.min_front = np.full(B, np.inf); self.min_rear = np.full(B, np.inf)
        self.collided = np.zeros(B, bool); self.n_sw = np.zeros(B, int)
        self.effort = z(); self.x_start = z(); self.f_brake = z()
        self.n_filter_active = z(); self.n_steps = z()
        # leader model: 0 = sinusoid (eq:leader, the "true" traffic); 1 = constant acceleration
        # (used only inside governor rollouts as a predictive / worst-case model)
        self.lmode = np.zeros((B, L), int); self.lacc = z(L); self.lacc_until = np.full((B, L), np.inf)
        # mismatch test traffic (lmode 2): piecewise-constant leader accelerations within the declared bound,
        # segment k active on [sch_t[..., k], sch_t[..., k+1])
        self.sch_t = np.full((B, L, self.SCH_K), np.inf); self.sch_a = np.zeros((B, L, self.SCH_K))
        self.dual_left = z(); self.old_lane = np.zeros(B, int); self.dual_fallback = z(); self.u_hold = z(); self.hold_on = np.zeros(B, bool)
        # disruption metrics ("what restraint buys")
        self.cut_open = np.zeros(B, bool); self.cut_since = z(); self.cut_peak = z()
        self.cut_sum = z(); self.n_cut = z(); self.n_cut_hard = z()
        self.hb_time = z()                       # time any human follower brakes harder than 3 m/s^2
        self.entry_gap = np.full(B, np.inf); self.entry_ttc = np.full(B, np.inf)
        self.u_prev = np.full(B, np.nan); self.jerk2 = z()

    # ------------------------------------------------------------------ state copy
    FIELDS = ['x', 'v', 't', 'lane', 'eta', 'vbar', 'A', 'w', 'phi', 'xl0', 'fvmax', 'f_alpha',
              'pulse_t', 'pulse_a', 'pulse_T', 'pulse_v', 'pulse_x', 'done', 'failed', 'min_front',
              'min_rear', 'collided', 'n_sw', 'effort', 'x_start', 'f_brake', 'n_filter_active', 'n_steps',
              'lmode', 'lacc', 'lacc_until', 'sch_t', 'sch_a', 'dual_left', 'old_lane', 'dual_fallback', 'u_hold', 'hold_on', 'cut_open', 'cut_since', 'cut_peak', 'cut_sum', 'n_cut', 'n_cut_hard',
              'hb_time', 'entry_gap', 'entry_ttc', 'u_prev', 'jerk2']
    HARD = 3.0        # hard-braking threshold [m/s^2]
    SCH_K = 24        # segments of the piecewise-constant leader schedule (mismatch traffic)
    CA_HOLD = 2.0     # horizon of the constant-acceleration forecast [s]
    CUT_WIN = 3.0     # window after a cut-in in which the new follower's braking is attributed [s]

    def close_cut(self, mask):
        m = mask & self.cut_open
        self.cut_sum = np.where(m, self.cut_sum + self.cut_peak, self.cut_sum)
        self.n_cut = self.n_cut + m
        self.n_cut_hard = self.n_cut_hard + (m & (self.cut_peak > self.HARD))
        self.cut_open = self.cut_open & ~m

    def to_model(self, kind, acc_cur=0.0, acc_tgt=0.0):
        """Turn this (cloned) env into a prediction model of the other vehicles, from the current state.
        kind='pred': leaders keep their current speed; followers use the declared nominal OVM.
        kind='case': leaders hold constant accelerations acc_cur/acc_tgt (in units of their declared
                     bound A|w|, i.e. in [-1,1]) for the current and the other lane.
        kind='ca':   leaders hold their currently measured acceleration (clipped to the declared
                     bound) for CA_HOLD seconds, then keep constant speed."""
        i = np.arange(self.B)
        q = self.lane
        bound = self.A * np.abs(self.w)
        _, acc_meas = self.leader_speed_acc(self.t)     # measured now (before the model switch)
        self.lmode[:] = 1
        a = np.zeros((self.B, self.L))
        until = np.full((self.B, self.L), np.inf)
        if kind == 'case':
            assert self.L == 2
            a[i, q] = acc_cur * bound[i, q]
            a[i, 1 - q] = acc_tgt * bound[i, 1 - q]
        elif kind == 'ca':
            a = np.clip(acc_meas, -bound, bound)
            until = self.t[:, None] + self.CA_HOLD + np.zeros((self.B, self.L))
        self.lacc = a; self.lacc_until = until
        self.fvmax[:] = 17.5                     # nominal desired speed (centre of the declared range)
        self.f_alpha[:] = self.c.ovm_alpha       # declared gain
        self.pulse_t[:] = np.inf; self.pulse_v[:] = 0; self.pulse_x[:] = 0
        return self

    def subset(self, idx, repeat=1):
        idx = np.repeat(np.asarray(idx), repeat)
        e = Env(self.c, len(idx), self.rng)
        for f in self.FIELDS:
            setattr(e, f, getattr(self, f)[idx].copy())
        return e

    def set_rows(self, idx, other):
        for f in self.FIELDS:
            getattr(self, f)[idx] = getattr(other, f)

    # ------------------------------------------------------------------ leaders
    def sched_acc(self, t):
        """Acceleration of schedule-driven leaders (lmode 2) at times t (B,), shape (B, L)."""
        idx = np.maximum((self.sch_t <= t[:, None, None]).sum(2) - 1, 0)
        return np.take_along_axis(self.sch_a, idx[:, :, None], 2)[:, :, 0]

    def leader_speed_acc(self, t):
        """t: (B,) -> v,a of the lane leaders, (B,L). Schedule-driven leaders (lmode 2) report their
        current speed and the scheduled acceleration (valid at the current time only)."""
        ph = self.w * t[:, None] + self.phi
        v = self.vbar + self.A * np.sin(ph)
        a = self.A * self.w * np.cos(ph)
        on = (t[:, None] >= self.pulse_t) & (t[:, None] < self.pulse_t + self.pulse_T)
        a = a + np.where(on, self.pulse_a, 0.0)
        v = v + self.pulse_v
        m2 = self.lmode == 2
        if m2.any():
            L = self.L
            v = np.where(m2, self.v[:, LEAD[:L]], v)
            a = np.where(m2, self.sched_acc(t), a)
        return v, a

    def leader_bound(self):
        """Declared lower acceleration bound of the lane leaders, (B,2): -A|w| (eq:disturbance)."""
        if self.c.leader_generic > 0:
            return np.full((self.B, self.L), -self.c.leader_generic)
        return -self.A * np.abs(self.w)

    # ------------------------------------------------------------------ neighbours
    def lane_view(self, k):
        """Ego-relative quantities in lane k (array (B,)), with ego treated as occupying lane k.
        Returns dict with slot validity, clearances d+/d-, H, D and declared bounds."""
        c = self.c
        i = np.arange(self.B)
        li, fi = LEAD[k], FOLL[k]
        xe, ve = self.x[:, 0], self.v[:, 0]
        xl, vl = self.x[i, li], self.v[i, li]
        xf, vf = self.x[i, fi], self.v[i, fi]
        valid = (xf < xe) & (xe < xl)
        dp = xl - xe - c.L_veh
        dm = xe - xf - c.L_veh
        Hp, Hm = dp - c.d0, dm - c.d0
        Dp, Dm = vl - ve, ve - vf
        a_lo_p = self.leader_bound()[i, k]
        a_hi_m = self.follower_bound(k)
        return dict(valid=valid, dp=dp, dm=dm, Hp=Hp, Hm=Hm, Dp=Dp, Dm=Dm,
                    a_lo_p=a_lo_p, a_hi_m=a_hi_m, vl=vl, vf=vf)

    def follower_bound(self, k, T=None):
        """Declared upper bound on the lane-k follower's acceleration over the next sample,
        valid for the declared OVM model with the ego as its leader (derivation in paper App./Sec. V-A).
        a_f <= min(a_max, max(-b, alpha (V(d_up) - v_low))),
        d_up = d + max(0, v_e - v_f) dt + (u_hi + b) dt^2 / 2,  v_low = v_f (1 - alpha h)^n_sub."""
        c = self.c
        if c.follower_bound == 'max':
            # model-free: only the follower's acceleration saturation is assumed
            return np.full(self.B, c.f_amax)
        i = np.arange(self.B)
        fi = FOLL[k]
        xe, ve = self.x[:, 0], self.v[:, 0]
        xf, vf = self.x[i, fi], self.v[i, fi]
        d = xe - xf - c.L_veh
        T = c.dt if T is None else T        # certification window (one sample, or the commitment T_m)
        n = int(round(c.n_sub * T / c.dt))
        d_up = d + np.maximum(0.0, ve - vf) * T + 0.5 * (c.u_hi + c.f_bmax) * T**2
        h = c.dt / c.n_sub
        a_decl = c.ovm_alpha               # DECLARED gain (true gain may differ in mismatch test)
        v_low = vf * (1.0 - a_decl * h) ** n
        ub = a_decl * (ovm_V(d_up, self.fvmax[:, k], c.ovm_dc, c.ovm_ell) - v_low)
        return np.minimum(c.f_amax, np.maximum(-c.f_bmax, ub))

    # ------------------------------------------------------------------ intervals
    def intervals(self, k, view=None):
        """Sampled interval I^Delta and sample-time HOCBF interval I_q for lane k. Returns
        (lo_s, hi_s, lo_c, hi_c, view)."""
        c = self.c
        g = self.lane_view(k) if view is None else view
        v = self.v[:, 0]
        Hp = np.maximum(g['Hp'], 0.0); Hm = np.maximum(g['Hm'], 0.0)
        mp = m_fun(Hp, g['Dp'], c.dt); mm = m_fun(Hm, g['Dm'], c.dt)
        lo_s = np.maximum.reduce([np.full(self.B, c.u_lo), -v / c.dt, g['a_hi_m'] - mm])
        hi_s = np.minimum.reduce([np.full(self.B, c.u_hi), (c.vmax - v) / c.dt, g['a_lo_p'] + mp])
        # negative clearance => certificate cannot be issued
        bad = (g['Hp'] < 0) | (g['Hm'] < 0) | ~g['valid']
        lo_s = np.where(bad, np.inf, lo_s); hi_s = np.where(bad, -np.inf, hi_s)
        U = g['a_lo_p'] + (c.lam_f + c.kap_f) * g['Dp'] + c.kap_f * c.lam_f * g['Hp']
        Lr = g['a_hi_m'] - (c.lam_r + c.kap_r) * g['Dm'] - c.kap_r * c.lam_r * g['Hm']
        lo_c = np.maximum.reduce([np.full(self.B, c.u_lo), Lr, -c.kap_v * v])
        hi_c = np.minimum.reduce([np.full(self.B, c.u_hi), U, c.kap_v * (c.vmax - v)])
        return lo_s, hi_s, lo_c, hi_c, g

    def window_interval(self, k, T):
        """Held-input interval over a window T (Prop. sampled with Delta -> T) for lane k: any constant
        u in [lo, hi] keeps lane k's clearances >= d0 and 0 <= v <= vmax for all times in the window."""
        c = self.c
        g = self.lane_view(k)
        v = self.v[:, 0]
        mp = m_fun(np.maximum(g['Hp'], 0.0), g['Dp'], T); mm = m_fun(np.maximum(g['Hm'], 0.0), g['Dm'], T)
        a_hi_m = self.follower_bound(k, T)
        lo = np.maximum.reduce([np.full(self.B, c.u_lo), -v / T, a_hi_m - mm])
        hi = np.minimum.reduce([np.full(self.B, c.u_hi), (c.vmax - v) / T, g['a_lo_p'] + mp])
        bad = (g['Hp'] < 0) | (g['Hm'] < 0) | ~g['valid']
        return np.where(bad, np.inf, lo), np.where(bad, -np.inf, hi)

    def commit_interval(self, k_old, k_new):
        """Intersection of both lanes' held-input intervals over the commitment T_m (per row lanes)."""
        T = self.c.Tm
        w = [self.window_interval(k, T) for k in range(self.L)]
        k_old = np.clip(k_old, 0, self.L - 1); k_new = np.clip(k_new, 0, self.L - 1)
        lo_o = np.choose(k_old, [x[0] for x in w]); hi_o = np.choose(k_old, [x[1] for x in w])
        lo_n = np.choose(k_new, [x[0] for x in w]); hi_n = np.choose(k_new, [x[1] for x in w])
        return np.maximum(lo_o, lo_n), np.minimum(hi_o, hi_n)

    def filt(self, u_c, k):
        """Executed input: clip onto I^Delta ∩ I_q when nonempty, else onto I^Delta.
        Returns u, feasible flag, active flag."""
        lo_s, hi_s, lo_c, hi_c, _ = self.intervals(k)
        feas = lo_s <= hi_s
        lo_b, hi_b = np.maximum(lo_s, lo_c), np.minimum(hi_s, hi_c)
        both = lo_b <= hi_b
        lo = np.where(both, lo_b, lo_s); hi = np.where(both, hi_b, hi_s)
        u = np.where(feas, np.clip(u_c, np.where(feas, lo, 0), np.where(feas, hi, 0)), 0.0)
        active = feas & (np.abs(u - u_c) > 1e-9)
        return u, feas, active

    def entry_ok(self, tgt=None):
        """Sampled lane-entry guard for the target lane tgt (default: the other lane of two; target H>=0,
        psi>=0 conservatively, speeds, nonempty I^Delta), plus dwell and horizon conditions.
        Rows whose target lane does not exist are inadmissible."""
        c = self.c
        if tgt is None:
            assert self.L == 2
            tgt = 1 - self.lane
        exists = (tgt >= 0) & (tgt < self.L)
        k = np.clip(tgt, 0, self.L - 1)
        ok = np.zeros(self.B, bool)
        for kk in range(self.L):
            sel = k == kk
            if not sel.any():
                continue
            lo_s, hi_s, _, _, g = self.intervals(kk)
            psi_p = g['Dp'] + c.lam_f * g['Hp']; psi_m = g['Dm'] + c.lam_r * g['Hm']
            okk = g['valid'] & (g['Hp'] >= 0) & (g['Hm'] >= 0) & (psi_p >= 0) & (psi_m >= 0) & (lo_s <= hi_s)
            ok = np.where(sel, okk, ok)
        if c.dual_lane and c.dual_mode == 'hold':
            # one input certified over the whole commitment for both lanes must exist
            lo_w, hi_w = self.commit_interval(self.lane, k)
            ok &= lo_w <= hi_w
        elif c.dual_lane:
            # the two lanes' sampled intervals must overlap at entry
            cur = self.intervals_cur(); tg = [self.intervals(kk) for kk in range(self.L)]
            lo_t = np.choose(k, [x[0] for x in tg]); hi_t = np.choose(k, [x[1] for x in tg])
            ok &= np.maximum(cur[0], lo_t) <= np.minimum(cur[1], hi_t)
        ok &= exists
        ok &= self.eta >= c.tau_d - 1e-9
        ok &= self.t + c.Tm <= c.Tf + 1e-9
        return ok

    def stay_ok(self):
        lo_s, hi_s, _, _, _ = self.intervals_cur()
        return lo_s <= hi_s

    def intervals_cur(self):
        out = [None] * 5
        res = [self.intervals(k) for k in range(self.L)]
        for j in range(4):
            out[j] = np.choose(self.lane, [r[j] for r in res])
        out[4] = None
        return out

    def filt_cur(self, u_c):
        res = [self.filt(u_c, k) for k in range(self.L)]
        u, f, a = (np.choose(self.lane, [r[j] for r in res]) for j in range(3))
        s = self.lane == 1
        if self.c.dual_lane and self.c.dual_mode == 'hold' and (self.dual_left > 1e-9).any():
            # hold mode: at the first step of the commitment choose u = clip(u_c) onto the commitment
            # interval of both lanes (state unchanged since entry) and hold it for T_m
            dl = self.dual_left > 1e-9
            first = dl & (self.dual_left > self.c.Tm - 0.5 * self.c.dt)
            if first.any():
                lo_w, hi_w = self.commit_interval(self.old_lane, self.lane)
                okw = lo_w <= hi_w
                uh = np.clip(u_c, np.where(okw, lo_w, 0.0), np.where(okw, hi_w, 0.0))
                self.u_hold = np.where(first & okw, uh, self.u_hold)
                self.dual_fallback = self.dual_fallback + (first & ~okw)
                # no certified hold (cannot happen after the guard): keep the per-sample filter
                self.hold_on = np.where(first, okw, self.hold_on)
            use = dl & self.hold_on
            a = np.where(use, np.abs(self.u_hold - u_c) > 1e-9, a)
            u = np.where(use, self.u_hold, u)
            f = np.where(use, True, f)
            return u, f, a
        if self.c.dual_lane and (self.dual_left > 1e-9).any():
            assert self.L == 2, 'intersect mode is implemented for two lanes'
            # during the commitment the input must also lie in the OLD lane's sampled interval
            dl = self.dual_left > 1e-9
            o = [self.intervals(0), self.intervals(1)]
            lo_o = np.where(self.old_lane == 1, o[1][0], o[0][0]); hi_o = np.where(self.old_lane == 1, o[1][1], o[0][1])
            lo_n = np.where(s, o[1][0], o[0][0]); hi_n = np.where(s, o[1][1], o[0][1])
            # new-lane executed interval: keep the anticipative choice when compatible
            lo = np.maximum(lo_o, lo_n); hi = np.minimum(hi_o, hi_n)
            both = dl & f & (lo <= hi)
            u_d = np.clip(u_c, np.where(both, lo, 0), np.where(both, hi, 0))
            # prefer the new lane's anticipative choice if it already satisfies the old lane
            ok_old = (u >= lo_o - 1e-12) & (u <= hi_o + 1e-12)
            u = np.where(both & ~ok_old, u_d, u)
            self.dual_fallback = self.dual_fallback + (dl & f & ~(lo <= hi))
            a = a | (both & ~ok_old)
        return u, f, a

    # ------------------------------------------------------------------ dynamics
    def change_lane(self, mask, tgt=None):
        """Reassign the ego to lane tgt (default: the other lane of two) where mask is set."""
        if tgt is None:
            tgt = 1 - self.lane
        self.close_cut(mask)
        if self.c.dual_lane:
            self.old_lane = np.where(mask, self.lane, self.old_lane)
            self.dual_left = np.where(mask, self.c.Tm, self.dual_left)
        self.lane = np.where(mask, tgt, self.lane)
        self.eta = np.where(mask, 0.0, self.eta)
        self.n_sw = self.n_sw + mask.astype(int)
        # disruption bookkeeping at entry: gap and time-to-collision to the NEW follower
        if mask.any():
            i = np.arange(self.B); fi = FOLL[self.lane]
            dm = self.x[:, 0] - self.x[i, fi] - self.c.L_veh
            close = self.v[i, fi] - self.v[:, 0]
            ttc = np.where(close > 1e-6, dm / np.maximum(close, 1e-6), np.inf)
            self.entry_gap = np.where(mask, np.minimum(self.entry_gap, dm), self.entry_gap)
            self.entry_ttc = np.where(mask, np.minimum(self.entry_ttc, ttc), self.entry_ttc)
            self.cut_open = self.cut_open | mask
            self.cut_since = np.where(mask, 0.0, self.cut_since)
            self.cut_peak = np.where(mask, 0.0, self.cut_peak)

    def step(self, u, active_mask=None):
        """Advance one sample with ego input u held constant. Only rows with active_mask advance.
        Returns per-row executed reward r_n = w_d (x_{n+1}-x_n) - w_u u^2 dt."""
        c = self.c
        B = self.B
        act = np.ones(B, bool) if active_mask is None else active_mask
        i = np.arange(B)
        h = c.dt / c.n_sub
        x0e, v0e = self.x[:, 0].copy(), self.v[:, 0].copy()
        min_f = np.full(B, np.inf); min_r = np.full(B, np.inf); fbrk = np.zeros(B)
        newf_brk = np.zeros(B); hb = np.zeros(B)
        lm1 = self.lmode == 1
        lm2 = self.lmode == 2
        for s in range(c.n_sub):
            ts = self.t + s * h
            # follower accelerations at substep start
            af = np.zeros((B, self.L))
            for k in range(self.L):
                fi = FOLL[k]
                in_k = (self.lane == k) | ((self.dual_left > 1e-9) & (self.old_lane == k))
                ego_here = in_k & (self.x[:, 0] > self.x[i, fi])
                xlead = np.where(ego_here, self.x[:, 0], self.x[i, LEAD[k]])
                dfol = xlead - self.x[i, fi] - c.L_veh
                a = self.f_alpha[:, k] * (ovm_V(dfol, self.fvmax[:, k], c.ovm_dc, c.ovm_ell) - self.v[i, fi])
                af[:, k] = np.clip(a, -c.f_bmax, c.f_amax)
            fbrk = np.maximum(fbrk, np.where(act, -af.min(axis=1), 0))
            newf_brk = np.maximum(newf_brk, -af[i, self.lane])          # follower in the ego's lane
            hb = hb + h * (af.min(axis=1) < -self.HARD)
            # leaders: exact at substep end
            t1 = ts + h
            ph = self.w * t1[:, None] + self.phi
            xl = self.xl0 + self.vbar * t1[:, None] - np.where(self.w != 0, self.A / np.where(self.w != 0, self.w, 1), 0) * (np.cos(ph) - np.cos(self.phi))
            # pulses (mismatch test): integrate piecewise-constant extra acceleration
            on = (ts[:, None] >= self.pulse_t) & (ts[:, None] < self.pulse_t + self.pulse_T)
            pa = np.where(on, self.pulse_a, 0.0)
            self.pulse_x = self.pulse_x + np.where(act[:, None], self.pulse_v * h + 0.5 * pa * h * h, 0)
            self.pulse_v = self.pulse_v + np.where(act[:, None], pa * h, 0)
            vl, _ = self.leader_speed_acc(t1)
            sch = self.sched_acc(ts) if lm2.any() else None
            for k in range(self.L):
                # constant-acceleration model leaders (rollouts only), speed kept in [0, vmax]
                v_old = self.v[:, LEAD[k]]
                acc_k = np.where(ts < self.lacc_until[:, k], self.lacc[:, k], 0.0)
                if lm2.any():
                    acc_k = np.where(lm2[:, k], sch[:, k], acc_k)
                v_new = np.clip(v_old + acc_k * h, 0.0, c.vmax)
                x_m = self.x[:, LEAD[k]] + 0.5 * (v_old + v_new) * h
                integ = lm1[:, k] | lm2[:, k]
                xk = np.where(integ, x_m, xl[:, k] + self.pulse_x[:, k])
                vk = np.where(integ, v_new, vl[:, k])
                self.x[:, LEAD[k]] = np.where(act, xk, self.x[:, LEAD[k]])
                self.v[:, LEAD[k]] = np.where(act, vk, self.v[:, LEAD[k]])
                fi = FOLL[k]
                self.x[:, fi] = np.where(act, self.x[:, fi] + self.v[:, fi] * h + 0.5 * af[:, k] * h * h, self.x[:, fi])
                self.v[:, fi] = np.where(act, np.maximum(self.v[:, fi] + af[:, k] * h, 0.0), self.v[:, fi])
            # ego exact
            tau = (s + 1) * h
            self.x[:, 0] = np.where(act, x0e + v0e * tau + 0.5 * u * tau**2, self.x[:, 0])
            self.v[:, 0] = np.where(act, v0e + u * tau, self.v[:, 0])
            # monitor occupied-lane clearances at substep end
            k = self.lane
            dp = self.x[i, LEAD[k]] - self.x[:, 0] - c.L_veh
            dm = self.x[:, 0] - self.x[i, FOLL[k]] - c.L_veh
            if c.dual_lane:
                # during the commitment the old lane's neighbours are monitored as well
                dlm = self.dual_left > 1e-9; ko = self.old_lane
                dpo = self.x[i, LEAD[ko]] - self.x[:, 0] - c.L_veh
                dmo = self.x[:, 0] - self.x[i, FOLL[ko]] - c.L_veh
                dp = np.where(dlm, np.minimum(dp, dpo), dp); dm = np.where(dlm, np.minimum(dm, dmo), dm)
            min_f = np.minimum(min_f, dp); min_r = np.minimum(min_r, dm)
        self.t = np.where(act, self.t + c.dt, self.t)
        self.eta = np.where(act, self.eta + c.dt, self.eta)
        self.dual_left = np.where(act, np.maximum(self.dual_left - c.dt, 0.0), self.dual_left)
        self.min_front = np.where(act, np.minimum(self.min_front, min_f), self.min_front)
        self.min_rear = np.where(act, np.minimum(self.min_rear, min_r), self.min_rear)
        self.collided |= act & ((min_f <= 0) | (min_r <= 0))
        self.f_brake = np.where(act, np.maximum(self.f_brake, fbrk), self.f_brake)
        # disruption: new follower's peak braking within CUT_WIN after a cut-in; hard-braking time
        self.cut_peak = np.where(act & self.cut_open, np.maximum(self.cut_peak, newf_brk), self.cut_peak)
        self.cut_since = np.where(act & self.cut_open, self.cut_since + c.dt, self.cut_since)
        self.close_cut(act & (self.cut_since >= self.CUT_WIN - 1e-9))
        self.hb_time = np.where(act, self.hb_time + hb, self.hb_time)
        du = np.where(np.isnan(self.u_prev), 0.0, u - self.u_prev)
        self.jerk2 = np.where(act, self.jerk2 + (du / c.dt) ** 2 * c.dt, self.jerk2)
        self.u_prev = np.where(act, u, self.u_prev)
        self.effort = np.where(act, self.effort + u**2 * c.dt, self.effort)
        self.n_steps = self.n_steps + act
        dx = self.x[:, 0] - x0e
        return np.where(act, c.w_d * dx - c.w_u * u**2 * c.dt, 0.0)

    # ------------------------------------------------------------------ observations
    def actor_obs(self, k=None):
        """Low-level actor input for lane k (default: current lane). 10 dims, normalised."""
        c = self.c
        if k is None:
            k = self.lane
        k = np.broadcast_to(k, (self.B,))
        cols = []
        res = [self.intervals(kk) for kk in range(self.L)]
        def pick(j, key=None):
            return np.choose(k, [r[j] if key is None else r[4][key] for r in res])
        lo_s, hi_s, lo_c, hi_c = pick(0), pick(1), pick(2), pick(3)
        lo = np.maximum(lo_s, lo_c); hi = np.minimum(hi_s, hi_c)
        lo = np.clip(np.where(np.isfinite(lo), lo, c.u_hi), c.u_lo, c.u_hi)
        hi = np.clip(np.where(np.isfinite(hi), hi, c.u_lo), c.u_lo, c.u_hi)
        v = self.v[:, 0]
        o = np.stack([
            v / c.vmax,
            np.clip(pick(None, 'dp'), -5, 100) / 50.0,
            pick(None, 'Dp') / 10.0,
            np.clip(pick(None, 'dm'), -5, 100) / 50.0,
            pick(None, 'Dm') / 10.0,
            pick(None, 'a_lo_p') / 3.0,
            pick(None, 'a_hi_m') / 3.0,
            lo / 6.0, hi / 6.0,
            (c.Tf - self.t) / c.Tf,
        ], axis=1)
        return o

    def critic_obs(self):
        """High-level critic input y (full state incl. disturbance parameters).
        L=2: 27 dims [time, speed, cooldown, current lane (7), other lane (7), disturbance (2x5)].
        L=3: 41 dims [time, speed, cooldown, current/left/right lane (3x7, zeros if absent),
             leftmost/rightmost flags, disturbance (3x5)]."""
        c = self.c
        q = self.lane
        i = np.arange(self.B)
        cool = np.maximum(0.0, c.tau_d - self.eta) / c.tau_d
        head = np.stack([(c.Tf - self.t) / c.Tf, self.v[:, 0] / c.vmax, cool], 1)
        def dist_of(kk, present):
            kc = np.clip(kk, 0, self.L - 1)
            d = [self.vbar[i, kc] / c.vmax, self.A[i, kc] / 5.0, self.w[i, kc],
                 np.sin(self.w[i, kc] * self.t + self.phi[i, kc]), np.cos(self.w[i, kc] * self.t + self.phi[i, kc])]
            return [x * present for x in d]
        scale = 0.0 if c.critic_nodist else 1.0
        if self.L == 2:
            dist = dist_of(q, 1.0) + dist_of(1 - q, 1.0)
            return np.concatenate([head, self.lane_feats(q), self.lane_feats(1 - q), np.stack(dist, 1) * scale], axis=1)
        lanes = [q, q - 1, q + 1]
        pres = [np.ones(self.B), (q - 1 >= 0).astype(float), (q + 1 < self.L).astype(float)]
        feats = [self.lane_feats(np.clip(k, 0, self.L - 1)) * p[:, None] for k, p in zip(lanes, pres)]
        dist = sum((dist_of(k, p) for k, p in zip(lanes, pres)), [])
        flags = np.stack([1.0 - pres[1], 1.0 - pres[2]], 1)
        return np.concatenate([head] + feats + [flags, np.stack(dist, 1) * scale], axis=1)

    def lane_feats(self, k):
        views = [self.lane_view(kk) for kk in range(self.L)]
        k = np.broadcast_to(k, (self.B,))
        def p(key):
            return np.choose(k, [g[key] for g in views])
        valid = p('valid').astype(float)
        return np.stack([np.clip(p('dp'), -50, 100) / 50.0, p('Dp') / 10.0,
                         np.clip(p('dm'), -50, 100) / 50.0, p('Dm') / 10.0,
                         p('a_lo_p') / 3.0, p('a_hi_m') / 3.0, valid], 1)

    def rel_gaps(self, k):
        g = self.lane_view(k)
        return g
