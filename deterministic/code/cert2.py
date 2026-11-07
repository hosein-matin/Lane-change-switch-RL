"""On-policy relabelling (DAgger) of the student and an episode-level conformal certificate, two lanes.

Usage: DUAL_LANE=1 DUAL_MODE=hold CRITIC_NODIST=1 RESULTS=dir python cert2.py SEED [TAG]
Needs amortize.py outputs for SEED (results/amort_{TAG}_s{SEED}/labels.npz, eval8_{TAG}_s{SEED}.pkl) and the
critic results/dqn_{TAG}_s{SEED}. Writes results/cert2_{TAG}_s{SEED}.pkl. Same steps as cert3.py, plus the
mismatch check: the nominal calibration quantile is tested on mismatched traffic (exact advantages there are
computed in the realised mismatched traffic, so the pathwise bound still holds; coverage is not guaranteed).
"""
import sys, os, time, pickle
import numpy as np
from config import Cfg
from env import Env
from scenarios import FAMILIES, Sampler, load_set
from nn import MLP
from hl import run, metrics, make_rule_margin, rule_stay_only, STAY, CHANGE
from amortize import feats, rollout_gain, train_student, AmortGate
from mismatch import apply_mismatch
from cert3 import charges, cq, ALPHAS
import evaluate as E

R0 = os.environ.get('RESULTS', os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, 'results'))
ROUNDS = int(os.environ.get('ROUNDS', 2)); N_DAG = int(os.environ.get('N_DAG', 2048))
N_CAL = int(os.environ.get('N_CAL', 500)); N_MIX = int(os.environ.get('N_MIX', 500))


class Shadow2:
    """Student decides (change iff g_hat > tau); exact advantage at every proposal.
    rec rows: [row, g_hat, g, change-rollout failed, accepted]."""
    def __init__(self, q, actor, c, net, mu, sd, tau):
        self.base = make_rule_margin([q], c, 0.0, 0.0); self.actor, self.c = actor, c
        self.net, self.mu, self.sd, self.tau = net, mu, sd, tau
        self.X, self.Y, self.rec = [], [], []

    def __call__(self, env, rows, ms, mc):
        a = self.base(env, rows, ms, mc)
        a = np.where(ms & ~mc, STAY, np.where(mc & ~ms, CHANGE, a))
        cand = np.where((a == CHANGE) & ms)[0]
        if len(cand) == 0:
            return a
        r = rows[cand]; f = feats(env)[r]
        gh = self.net.forward((f - self.mu) / self.sd)[:, 0] * 10.0
        g, fl, _ = rollout_gain(env, r, self.actor, self.base, self.c, 'true')
        ok = gh > self.tau
        self.X.append(f); self.Y.append(g)
        self.rec.append(np.stack([r.astype(float), gh, g, fl.astype(float), ok.astype(float)], 1))
        a[cand[~ok]] = STAY
        return a


def main(seed, tag='best'):
    c = Cfg(); t0 = time.time(); log = lambda *a: print(f'[{time.time()-t0:7.1f}s]', *a, flush=True)
    fname = f'{R0}/cert2_{tag}_s{seed}.pkl'
    out = pickle.load(open(fname, 'rb')) if os.path.exists(fname) else dict(seed=seed)
    e8 = pickle.load(open(f'{R0}/eval8_{tag}_s{seed}.pkl', 'rb'))
    actor = MLP.load(f'{R0}/ppo_s{seed}/actor.pkl'); qb = MLP.load(f'{R0}/dqn_{tag}_s{seed}/q.pkl')
    base = make_rule_margin([qb], c, 0.0, 0.0)
    d = f'{R0}/amort_{tag}_s{seed}'
    z = np.load(f'{d}/labels.npz'); X, Y = z['X'], z['Y']
    test, val = E.build_sets(c); T = {f: test[f][0] for f in test}

    def env_of(scen, mis=False):
        e = Env(c, len(scen)); load_set(e, scen); e.done[:] = False
        if mis:
            apply_mismatch(e, 0)
        return e

    def env_from_sampler(seed_s, n, mis=False):
        e = Env(c, n); Sampler(c, seed_s).reset_rows(e, range(n)); e.done[:] = False
        if mis:
            apply_mismatch(e, 0)
        return e

    def tune_tau(net, mu, sd):
        tv = {}
        for tau in [-2.0, -1.0, 0.0, 1.0, 2.0]:
            tv[tau] = float(run(env_of(val), actor, AmortGate(qb, actor, c, net, mu, sd, tau=tau), c)['G'].mean())
        return max(tv, key=tv.get), tv

    out.setdefault('rounds', {})
    net, mu, sd, cal = train_student(X, Y, seed); tau = e8['tuned']['tau']
    out['rounds'][0] = dict(n_labels=int(len(Y)), tau=tau, cal=cal)
    for k in range(1, ROUNDS + 1):
        p = f'{d}/dagger_r{k}.npz'
        if not os.path.exists(p):
            B = 64; sh = Shadow2(qb, actor, c, net, mu, sd, tau); smp = Sampler(c, 90_000 + 1000 * k + seed)
            for j in range(max(1, N_DAG // B)):
                env = Env(c, B); smp.reset_rows(env, range(B)); env.done[:] = False; run(env, actor, sh, c)
            Xk = np.concatenate(sh.X) if sh.X else np.zeros((0, X.shape[1])); Yk = np.concatenate(sh.Y) if sh.Y else np.zeros(0)
            rec = np.concatenate(sh.rec) if sh.rec else np.zeros((0, 5))
            wrong = float(np.mean((rec[:, 4] > 0.5) != (rec[:, 2] >= 0))) if len(rec) else 0.0
            np.savez(p, X=Xk, Y=Yk, wrong=wrong)
        zz = np.load(p); Xk, Yk, wr = zz['X'], zz['Y'], float(zz['wrong'])
        X = np.concatenate([X, Xk]); Y = np.concatenate([Y, Yk])
        net, mu, sd, cal = train_student(X, Y, seed)
        tau, tv = tune_tau(net, mu, sd)
        out['rounds'][k] = dict(n_labels=int(len(Y)), new=int(len(Yk)), wrong_prev_student=wr, tau=tau, tau_val=tv, cal=cal)
        log(f'round {k}: +{len(Yk)} labels, wrong decisions of the previous student on its own proposals {wr:.3f}, tau={tau}')
    net.save(f'{d}/student_final.pkl'); np.savez(f'{d}/norm_final.npz', mu=mu, sd=sd)
    out['tau'] = tau

    def episode_stats(e):
        e_b = e.subset(np.arange(e.B)); e_b.done[:] = False
        e_s = e.subset(np.arange(e.B)); e_s.done[:] = False
        sh = Shadow2(qb, actor, c, net, mu, sd, tau)
        res = run(e, actor, sh, c); m = metrics(e, res)
        rec = np.concatenate(sh.rec) if sh.rec else np.zeros((0, 5))
        C, W, P = charges(rec, e.B)
        Gb = run(e_b, actor, base, c)['G']; Gs = run(e_s, actor, rule_stay_only, c)['G']
        return dict(G=res['G'], Gb=Gb, Gs=Gs, C=C, wrong=W, props=P, rec=rec, n_sw=m['n_sw'],
                    n_cut_hard=m['n_cut_hard'], min_front=m['min_front'], min_rear=m['min_rear'], fail=m['fail'])

    def check(st, q):
        Wv = np.maximum(st['Gb'], st['Gs'])
        o = {a: dict(cov_charge=float(np.mean(st['C'] <= qa)), cov_return=float(np.mean(st['G'] >= Wv - qa - 1e-6)))
             for a, qa in q.items()}
        o['pathwise_bound_holds'] = float(np.mean(st['G'] >= Wv - st['C'] - 1e-6))
        return o

    if 'cal' not in out:
        out['cal'] = episode_stats(env_from_sampler(123_000 + seed, N_CAL))
        out['q'] = {str(a): cq(out['cal']['C'], a) for a in ALPHAS}
        log('calibration q', out['q'], 'frac zero charge', float(np.mean(out['cal']['C'] == 0)))
        pickle.dump(out, open(fname, 'wb'))
    for key, mis in [('mix', False), ('mix_mis', True)]:
        if key not in out:
            out[key] = episode_stats(env_from_sampler(124_000 + seed, N_MIX, mis)); out[key + '_check'] = check(out[key], out['q'])
            log(key, out[key + '_check']); pickle.dump(out, open(fname, 'wb'))
    for key, mis in [('test', False), ('test_mis', True)]:
        out.setdefault(key, {})
        for f in FAMILIES:
            if f not in out[key]:
                out[key][f] = episode_stats(env_of(T[f], mis)); pickle.dump(out, open(fname, 'wb'))
        G = np.concatenate([out[key][f]['G'] for f in FAMILIES])
        log(key, 'final student G', round(float(G.mean()), 2))
    if 'cascade' not in out:
        dv = {}
        for dl in [0.5, 1.0, 2.0, 4.0]:
            g2 = AmortGate(qb, actor, c, net, mu, sd, delta=dl, fallback='ca')
            dv[dl] = (float(run(env_of(val), actor, g2, c)['G'].mean()), g2.st.rolled / len(val))
        db = max(dv, key=lambda k: dv[k][0]); cas = {}
        for key, mis in [('test', False), ('test_mis', True)]:
            cas[key] = {}
            for f in FAMILIES:
                g2 = AmortGate(qb, actor, c, net, mu, sd, delta=db, fallback='ca'); e = env_of(T[f], mis)
                res = run(e, actor, g2, c); m = metrics(e, res)
                cas[key][f] = dict(G=res['G'], n_sw=m['n_sw'], n_cut_hard=m['n_cut_hard'], decisions=g2.st.rolled / len(T[f]),
                                   min_front=m['min_front'], min_rear=m['min_rear'])
        out['cascade'] = dict(delta=db, val=dv, **cas)
        log('cascade delta', db, 'G', round(float(np.concatenate([cas['test'][f]['G'] for f in FAMILIES]).mean()), 2))
    pickle.dump(out, open(fname, 'wb')); log('saved', fname)


if __name__ == '__main__':
    main(int(sys.argv[1]), sys.argv[2] if len(sys.argv) > 2 else 'best')
