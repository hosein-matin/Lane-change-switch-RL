"""Amortized rollout gate: a student network learns the exact gate's advantage g(y) = Q_b(y,c) - V_s(y).

Usage: python amortize.py SEED [CRITIC_TAG]   (CRITIC_TAG default 'c010' -> results/dqn_c010_s{SEED}/q.pkl)
Writes results/eval8_{CRITIC_TAG}_s{SEED}.pkl (test metrics in the same format as the other rounds) and
results/amort_{CRITIC_TAG}_s{SEED}/ (labels, student, tuning log).

Pipeline
  1. Labels: run the exact gate (rho = 0, exact full-horizon rollouts = training simulator) on training-mixture
     episodes (scenario seeds disjoint from validation and test). At every proposal (pi_b = change, stay
     admissible) record the student features and the label g = G(change, then pi_b) - G(stay).
  2. Student: MLP regression on clipped labels (Huber). Features are observable only: the critic's traffic
     features (no leader disturbance parameters) plus the measured leader accelerations.
  3. Calibration: on held-out labels, delta_alpha = (1 - alpha) quantile of |g_hat - g| (split conformal).
  4. Tuning on the 300-scenario validation set (episode return): tau for the student-only rule, delta for the
     cascade (grid incl. the conformal values). Test sets are used once.
  5. Test: student-only, cascade with deployable forecast fallback ('ca') and with exact fallback ('true'),
     and the exact teacher gate itself.
"""
import sys, os, time, pickle, json
import numpy as np
from config import Cfg
from env import Env
from scenarios import FAMILIES, Sampler
from nn import MLP, Adam
from hl import run, make_rule_margin, model_clone, rule_stay_only, STAY, CHANGE, PredictionGate
import evaluate as E

R0 = os.environ.get('RESULTS', os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, 'results'))
N_LABEL_EP = int(os.environ.get('N_LABEL_EP', 2048))
GCLIP = 30.0      # label clip for training (failures carry -200; only the sign/magnitude near 0 matters)


def feats(env):
    """Observable student features: critic features without disturbance parameters + measured leader accels."""
    y = env.critic_obs()[:, :17]
    _, a = env.leader_speed_acc(env.t)
    i = np.arange(env.B); q = env.lane
    return np.concatenate([y, (a[i, q] / 3.0)[:, None], (a[i, 1 - q] / 3.0)[:, None]], 1)


class Stats:
    def __init__(self):
        self.props = 0; self.rolled = 0; self.sim_steps = 0; self.accepted = 0


def rollout_gain(env, r, actor, base, c, model):
    """Two rollouts (change then pi_b / stay) from rows r; returns gain, change-failed flag, simulated steps."""
    sc = model_clone(env, r, 1, model); ss = model_clone(env, r, 1, model)
    n0 = sc.n_steps.copy(); m0 = ss.n_steps.copy()
    rc = run(sc, actor, base, c, forced_first=np.full(len(r), CHANGE))
    rs = run(ss, actor, rule_stay_only, c, forced_first=np.full(len(r), STAY))
    steps = int((sc.n_steps - n0).sum() + (ss.n_steps - m0).sum())
    return rc['G'] - rs['G'], rc['fail'], steps


class TeacherGate:
    """Exact gate (rho = 0) that records (features, label) at every proposal."""
    def __init__(self, q, actor, c):
        self.base = make_rule_margin([q], c, 0.0, 0.0); self.actor, self.c = actor, c
        self.X, self.Y = [], []; self.st = Stats()

    def __call__(self, env, rows, ms, mc):
        a = self.base(env, rows, ms, mc)
        a = np.where(ms & ~mc, STAY, np.where(mc & ~ms, CHANGE, a))
        cand = np.where((a == CHANGE) & ms)[0]
        if len(cand) == 0:
            return a
        r = rows[cand]
        g, fail, steps = rollout_gain(env, r, self.actor, self.base, self.c, 'true')
        self.X.append(feats(env)[r]); self.Y.append(g)
        self.st.props += len(r); self.st.rolled += len(r); self.st.sim_steps += steps
        ok = ~fail & (g >= 0.0)
        a[cand[~ok]] = STAY
        return a


class AmortGate:
    """Student-only (delta=None: change iff g_hat > tau) or cascade (accept g_hat >= delta, reject g_hat <= -delta,
    otherwise two rollouts with the fallback model and the gate's rule g >= 0)."""
    def __init__(self, q, actor, c, student, mu, sd, tau=0.0, delta=None, fallback='ca', shadow=False):
        self.base = make_rule_margin([q], c, 0.0, 0.0); self.actor, self.c = actor, c
        self.student, self.mu, self.sd = student, mu, sd
        self.tau, self.delta, self.fallback = tau, delta, fallback
        self.st = Stats()
        # shadow: also compute the exact advantage g at every proposal (diagnostics only; decisions unchanged)
        self.shadow = shadow; self.rec = []

    def ghat(self, env, r):
        return self.student.forward((feats(env)[r] - self.mu) / self.sd)[:, 0] * 10.0

    def __call__(self, env, rows, ms, mc):
        a = self.base(env, rows, ms, mc)
        a = np.where(ms & ~mc, STAY, np.where(mc & ~ms, CHANGE, a))
        cand = np.where((a == CHANGE) & ms)[0]
        if len(cand) == 0:
            return a
        r = rows[cand]; gh = self.ghat(env, r); self.st.props += len(r)
        if self.shadow:
            g_ex, f_ex, _ = rollout_gain(env, r, self.actor, self.base, self.c, 'true')
            self.rec.append(np.stack([gh, g_ex, f_ex.astype(float)], 1))
        if self.delta is None:
            ok = gh > self.tau
        else:
            ok = gh >= self.delta
            band = np.abs(gh) < self.delta
            if band.any():
                g, fail, steps = rollout_gain(env, r[band], self.actor, self.base, self.c, self.fallback)
                ok[band] = ~fail & (g >= 0.0)
                self.st.rolled += int(band.sum()); self.st.sim_steps += steps
        self.st.accepted += int(ok.sum())
        a[cand[~ok]] = STAY
        return a


def collect_labels(q, actor, c, seed, out_dir):
    p = f'{out_dir}/labels.npz'
    if os.path.exists(p):
        d = np.load(p); return d['X'], d['Y']
    B = 64; env = Env(c, B); sampler = Sampler(c, 70_000 + seed)
    t = TeacherGate(q, actor, c); t0 = time.time()
    for k in range(N_LABEL_EP // B):
        sampler.reset_rows(env, range(B)); env.done[:] = False
        run(env, actor, t, c)
        if k % 8 == 0:
            print(f'  labels: {(k+1)*B} episodes, {sum(len(y) for y in t.Y)} proposals, {time.time()-t0:.0f}s', flush=True)
    if not t.Y:      # no proposals at all (only possible with an undertrained critic, e.g. the smoke test)
        t.X.append(np.zeros((0, feats(env).shape[1]))); t.Y.append(np.zeros(0))
    X = np.concatenate(t.X); Y = np.concatenate(t.Y)
    np.savez(p, X=X, Y=Y)
    return X, Y


def train_student(X, Y, seed, epochs=300):
    rng = np.random.default_rng(seed + 31)
    if len(Y) < 10:  # degenerate (smoke test): untrained student, conformal band covers everything
        net = MLP([X.shape[1], 128, 128, 1], rng, out_scale=0.0)
        return net, np.zeros(X.shape[1]), np.ones(X.shape[1]), dict(n=int(len(Y)), n_cal=0, mae_cal=None,
               sign_agree_cal=None, delta={str(al): 1e6 for al in [0.05, 0.1, 0.2, 0.3, 0.5]}, frac_pos=None)
    n = len(Y); idx = rng.permutation(n); ncal = n // 5
    cal, tr = idx[:ncal], idx[ncal:]
    mu = X[tr].mean(0); sd = X[tr].std(0) + 1e-6
    Xt = (X - mu) / sd; Yt = np.clip(Y, -GCLIP, GCLIP) / 10.0
    net = MLP([X.shape[1], 128, 128, 1], rng, out_scale=0.1); opt = Adam(net.params(), lr=1e-3, clip=5.0)
    bs = 256
    for ep in range(epochs):
        p = rng.permutation(tr)
        for j in range(0, len(p), bs):
            b = p[j:j + bs]
            out, hs = net.forward(Xt[b], cache=True)
            err = out[:, 0] - Yt[b]
            dout = np.clip(err, -1.0, 1.0)[:, None] / len(b)
            opt.step(net.backward(hs, dout))
    gh = net.forward(Xt)[:, 0] * 10.0
    res = np.abs(gh[cal] - Y[cal])            # unclipped labels: the band refers to the deployed g
    res_clip = np.abs(gh[cal] - np.clip(Y[cal], -GCLIP, GCLIP))
    sign_agree = float(np.mean(np.sign(gh[cal]) == np.sign(Y[cal])))
    cal_info = dict(n=int(n), n_cal=int(ncal), mae_cal=float(res.mean()), sign_agree_cal=sign_agree,
                    delta={str(al): float(np.quantile(res, 1 - al)) for al in [0.05, 0.1, 0.2, 0.3, 0.5]},
                    delta_clipped={str(al): float(np.quantile(res_clip, 1 - al)) for al in [0.1]},
                    frac_pos=float(np.mean(Y > 0)))
    return net, mu, sd, cal_info


def ev(c, actor, rule, scen):
    m, _ = E.evaluate(c, actor, rule, scen)
    return m


def main(seed, tag='c010'):
    c = Cfg(); t0 = time.time()
    out_dir = f'{R0}/amort_{tag}_s{seed}'; os.makedirs(out_dir, exist_ok=True)
    actor = MLP.load(f'{R0}/ppo_s{seed}/actor.pkl'); q = MLP.load(f'{R0}/dqn_{tag}_s{seed}/q.pkl')
    X, Y = collect_labels(q, actor, c, seed, out_dir)
    print(f'[{time.time()-t0:.0f}s] labels {len(Y)}', flush=True)
    net, mu, sd, cal = train_student(X, Y, seed)
    net.save(f'{out_dir}/student.pkl'); np.savez(f'{out_dir}/norm.npz', mu=mu, sd=sd)
    print(f'[{time.time()-t0:.0f}s] student', json.dumps(cal), flush=True)
    test, val = E.build_sets(c); T = {f: test[f][0] for f in test}
    # ---- validation tuning (episode return)
    tune = dict(tau={}, delta={})
    for tau in [-2.0, -1.0, 0.0, 1.0, 2.0]:
        r = AmortGate(q, actor, c, net, mu, sd, tau=tau)
        tune['tau'][tau] = float(ev(c, actor, r, val)['G'].mean())
    dgrid = sorted(set([0.5, 1.0, 2.0, 4.0] + [round(v, 2) for v in cal['delta'].values()]))
    for d in dgrid:
        r = AmortGate(q, actor, c, net, mu, sd, delta=d, fallback='ca')
        m = ev(c, actor, r, val)
        tune['delta'][d] = (float(m['G'].mean()), r.st.rolled / len(val))
    tau_b = max(tune['tau'], key=tune['tau'].get)
    d_b = max(tune['delta'], key=lambda k: tune['delta'][k][0])
    d_conf = cal['delta']['0.1']
    print(f'[{time.time()-t0:.0f}s] tuned tau={tau_b} delta={d_b} conformal(0.1)={d_conf:.2f}', json.dumps(tune), flush=True)
    # ---- test (used once)
    meth = {f'{tag}_student': AmortGate(q, actor, c, net, mu, sd, tau=tau_b),
            f'{tag}_student_tau0': AmortGate(q, actor, c, net, mu, sd, tau=0.0),
            f'{tag}_cascade_ca': AmortGate(q, actor, c, net, mu, sd, delta=d_b, fallback='ca'),
            f'{tag}_cascade_conf_ca': AmortGate(q, actor, c, net, mu, sd, delta=d_conf, fallback='ca'),
            f'{tag}_cascade_true': AmortGate(q, actor, c, net, mu, sd, delta=d_b, fallback='true'),
            f'{tag}_teacher_true': TeacherGate(q, actor, c),
            f'{tag}_student_cov': AmortGate(q, actor, c, net, mu, sd, tau=tau_b, shadow=True)}
    fname = f'{R0}/eval8_{tag}_s{seed}.pkl'
    out = dict(seed=seed, test={}, rollouts={}, tuned=dict(tau=tau_b, delta=d_b, delta_conf=d_conf), tune=tune, cal=cal)
    for name, rule in meth.items():
        out['test'][name] = {}; out['rollouts'][name] = {}
        for f in FAMILIES:
            rule.st = Stats()
            if getattr(rule, 'shadow', False):
                rule.rec = []
            out['test'][name][f] = ev(c, actor, rule, T[f])
            if getattr(rule, 'shadow', False):
                out.setdefault('coverage', {})[f] = np.concatenate(rule.rec) if rule.rec else np.zeros((0, 3))
            n = len(T[f])
            out['rollouts'][name][f] = dict(props=rule.st.props / n, rolled=rule.st.rolled / n, sim_steps=rule.st.sim_steps / n)
        G = np.concatenate([out['test'][name][f]['G'] for f in FAMILIES])
        ro = np.mean([out['rollouts'][name][f]['rolled'] for f in FAMILIES])
        print(f'[{time.time()-t0:.0f}s] {name}: G={G.mean():.2f} rolled/ep={ro:.2f}', flush=True)
        pickle.dump(out, open(fname, 'wb'))
    print(f'[{time.time()-t0:.0f}s] saved {fname}', flush=True)


if __name__ == '__main__':
    main(int(sys.argv[1]), sys.argv[2] if len(sys.argv) > 2 else 'c010')
