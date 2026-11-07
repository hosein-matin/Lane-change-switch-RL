"""On-policy relabelling (DAgger) of the student and an episode-level conformal certificate, L = 3 lanes.

Usage: N_LANES=3 DUAL_LANE=1 DUAL_MODE=hold CRITIC_NODIST=1 RESULTS3=dir python cert3.py SEED
Needs the outputs of eval3l.py for SEED (actor, critics, teacher labels, tuned tau). Writes RESULTS3/cert3_s{SEED}.pkl.

1. Round 0: the student trained on teacher-driven labels (identical to eval3l.py).
2. Rounds 1..ROUNDS: run the current student (decisions g_hat > tau) on fresh training-mixture scenarios, compute the
   exact advantage g at every proposal it visits, aggregate the labels, retrain, re-tune tau on validation.
3. Certificate (Prop. gate): on one scenario, the deployed rule's charge C = sum over proposals where it decides
   differently from g >= 0 of |g| satisfies G_pi >= max(G_b, G_s) - C. Run the final student on N_CAL fresh
   training-mixture scenarios, q_alpha = ceil((n+1)(1-alpha))-th smallest charge. For a new scenario from the same
   distribution, P(G_pi >= max(G_b, G_s) - q_alpha) >= 1 - alpha (split conformal; scenarios i.i.d., rule fixed).
4. Checks on N_MIX fresh training-mixture scenarios (coverage of the charge and of the return inequality) and the
   per-family test sets (same as eval3l.py; no guarantee there: the families are not the calibration distribution).
5. Cascade with the final student (validated delta, CA fallback) on the test families.
"""
import sys, os, time, json, pickle
import numpy as np
from config import Cfg
from env import Env
from nn import MLP
from scenarios3 import FAMILIES3, Sampler3, make_test_set3, load_set3
from hl3 import run3, Gate3, DIRS, feats3, rollout_pair3, metrics, make_rule_greedy3, make_rule_stay_then, Stats
from amortize import train_student

R3 = os.environ.get('RESULTS3', os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, 'results3'))
ROUNDS = int(os.environ.get('ROUNDS', 2)); N_DAG = int(os.environ.get('N_DAG', 2048))
N_CAL = int(os.environ.get('N_CAL', 500)); N_MIX = int(os.environ.get('N_MIX', 500))
N_TEST = int(os.environ.get('N_TEST', 100)); N_VAL = int(os.environ.get('N_VAL', 300))
ALPHAS = [0.05, 0.1, 0.2]
TAG = {2: '', 5: 'c05_', 10: 'c010_', 14: 'c014_', 20: 'c020_'}


class Shadow(Gate3):
    """Student decides (accept iff g_hat > tau); the exact advantage is computed at every proposal.
    rec rows: [row, g_hat, g, change-rollout failed, accepted]."""
    def __call__(self, env, rows, M):
        a = self.base(env, rows, M); cand = np.where((a > 0) & M[:, 0])[0]
        if len(cand) == 0:
            return a
        r = rows[cand]; acts = a[cand]
        f = np.concatenate([feats3(env)[r], DIRS[acts][:, None].astype(float)], 1)
        gh = self.student.forward((f - self.mu) / self.sd)[:, 0] * 10.0
        g, fl, _ = rollout_pair3(env, r, acts, self.actor, self.base, self.c, 'true')
        ok = gh > self.tau
        self.X.append(f); self.Y.append(g)
        self.rec.append(np.stack([r.astype(float), gh, g, fl.astype(float), ok.astype(float)], 1))
        a[cand[~ok]] = 0
        return a


def charges(rec, n):
    """Per-episode charge of Prop. gate and number of wrong decisions (wrong: accept != (g >= 0))."""
    C = np.zeros(n); W = np.zeros(n); P = np.zeros(n)
    if len(rec):
        r = rec[:, 0].astype(int); wrong = (rec[:, 4] > 0.5) != (rec[:, 2] >= 0)
        np.add.at(C, r, np.where(wrong, np.abs(rec[:, 2]), 0.0)); np.add.at(W, r, wrong.astype(float)); np.add.at(P, r, 1.0)
    return C, W, P


def cq(C, alpha):
    n = len(C); k = int(np.ceil((n + 1) * (1 - alpha)))
    return float(np.sort(C)[k - 1]) if k <= n else float('inf')


def main(seed):
    c = Cfg(); assert c.n_lanes == 3
    t0 = time.time(); log = lambda *a: print(f'[{time.time()-t0:7.1f}s]', *a, flush=True)
    fname = f'{R3}/cert3_s{seed}.pkl'
    out = pickle.load(open(fname, 'rb')) if os.path.exists(fname) else dict(seed=seed)
    e3 = pickle.load(open(f'{R3}/eval3l_s{seed}.pkl', 'rb'))
    actor = MLP.load(f'{R3}/ppo_s{seed}/actor.pkl'); qb = MLP.load(f'{R3}/dqn3_{TAG[e3["best_c0"]]}s{seed}/q.pkl')
    base = make_rule_greedy3(qb)
    d = f'{R3}/amort3_s{seed}'
    z = np.load(f'{d}/labels.npz'); X, Y = z['X'], z['Y']
    vs = Sampler3(c, 55_555); venv = Env(c, N_VAL); vs.reset_rows(venv, range(N_VAL))

    def run_on(rule, env_src):
        e = env_src.subset(np.arange(env_src.B)); e.done[:] = False
        res = run3(e, actor, rule, c); return e, res

    def tune_tau(net, mu, sd):
        tv = {}
        for tau in [-2.0, -1.0, 0.0, 1.0, 2.0]:
            _, res = run_on(Gate3(qb, actor, c, mode='student', student=net, mu=mu, sd=sd, tau=tau), venv)
            tv[tau] = float(res['G'].mean())
        return max(tv, key=tv.get), tv

    # ---- rounds
    out.setdefault('rounds', {})
    net, mu, sd, cal = train_student(X, Y, seed); tau = e3['tuned']['tau']
    out['rounds'][0] = dict(n_labels=int(len(Y)), tau=tau, cal=cal)
    for k in range(1, ROUNDS + 1):
        p = f'{d}/dagger_r{k}.npz'
        if os.path.exists(p):
            zz = np.load(p); Xk, Yk = zz['X'], zz['Y']
        else:
            B = 64; env = Env(c, B); smp = Sampler3(c, 90_000 + 1000 * k + seed)
            sh = Shadow(qb, actor, c, mode='student', student=net, mu=mu, sd=sd, tau=tau)
            for j in range(max(1, N_DAG // B)):
                smp.reset_rows(env, range(B)); env.done[:] = False; run3(env, actor, sh, c)
            Xk = np.concatenate(sh.X) if sh.X else np.zeros((0, X.shape[1])); Yk = np.concatenate(sh.Y) if sh.Y else np.zeros(0)
            rec = np.concatenate(sh.rec) if sh.rec else np.zeros((0, 5))
            wrong = float(np.mean((rec[:, 4] > 0.5) != (rec[:, 2] >= 0))) if len(rec) else 0.0
            np.savez(p, X=Xk, Y=Yk, wrong=wrong)
        wr = float(np.load(p)['wrong'])
        X = np.concatenate([X, Xk]); Y = np.concatenate([Y, Yk])
        net, mu, sd, cal = train_student(X, Y, seed)
        tau, tv = tune_tau(net, mu, sd)
        out['rounds'][k] = dict(n_labels=int(len(Y)), new=int(len(Yk)), wrong_prev_student=wr, tau=tau, tau_val=tv, cal=cal)
        log(f'round {k}: +{len(Yk)} labels, wrong decisions of the previous student on its own proposals {wr:.3f}, tau={tau}')
    net.save(f'{d}/student_final.pkl'); np.savez(f'{d}/norm_final.npz', mu=mu, sd=sd)
    out['tau'] = tau

    def episode_stats(env_src):
        """Final student with exact shadow + pi_b and pi_s from the same initial states."""
        sh = Shadow(qb, actor, c, mode='student', student=net, mu=mu, sd=sd, tau=tau)
        e, res = run_on(sh, env_src); m = metrics(e, res)
        rec = np.concatenate(sh.rec) if sh.rec else np.zeros((0, 5))
        C, W, P = charges(rec, env_src.B)
        _, rb = run_on(base, env_src); _, rs = run_on(make_rule_stay_then(base), env_src)
        return dict(G=res['G'], Gb=rb['G'], Gs=rs['G'], C=C, wrong=W, props=P, rec=rec, n_sw=m['n_sw'],
                    n_cut_hard=m['n_cut_hard'], min_front=m['min_front'], min_rear=m['min_rear'], fail=m['fail'])

    def env_from_sampler(seed_s, n):
        e = Env(c, n); Sampler3(c, seed_s).reset_rows(e, range(n)); return e

    # ---- calibration and the training-mixture check
    if 'cal' not in out:
        out['cal'] = episode_stats(env_from_sampler(123_000 + seed, N_CAL))
        out['q'] = {str(a): cq(out['cal']['C'], a) for a in ALPHAS}
        log('calibration q', out['q'], 'frac zero charge', float(np.mean(out['cal']['C'] == 0)))
        pickle.dump(out, open(fname, 'wb'))
    if 'mix' not in out:
        st = episode_stats(env_from_sampler(124_000 + seed, N_MIX)); out['mix'] = st
        Wv = np.maximum(st['Gb'], st['Gs'])
        out['mix_check'] = {a: dict(cov_charge=float(np.mean(st['C'] <= q)), cov_return=float(np.mean(st['G'] >= Wv - q - 1e-6)))
                            for a, q in out['q'].items()}
        out['mix_check']['pathwise_bound_holds'] = float(np.mean(st['G'] >= Wv - st['C'] - 1e-6))
        log('mix check', out['mix_check'])
        pickle.dump(out, open(fname, 'wb'))
    # ---- test families
    test = {f: make_test_set3(c, f, N_TEST, seed=20_000 + 17 * i)[0] for i, f in enumerate(FAMILIES3)}
    out.setdefault('test', {})
    for f in FAMILIES3:
        if f in out['test']:
            continue
        e = Env(c, len(test[f])); load_set3(e, test[f]); e.done[:] = False
        out['test'][f] = episode_stats(e)
        pickle.dump(out, open(fname, 'wb'))
    G = np.concatenate([out['test'][f]['G'] for f in FAMILIES3])
    log('final student test G', round(float(G.mean()), 2), 'round-0 student (eval3l)',
        round(float(np.concatenate([e3['test']['best_student'][f]['G'] for f in FAMILIES3]).mean()), 2))
    # ---- cascade with the final student (validated delta, CA fallback)
    if 'cascade' not in out:
        dv = {}
        for dl in [0.5, 1.0, 2.0, 4.0]:
            g3 = Gate3(qb, actor, c, mode='cascade', model='ca', student=net, mu=mu, sd=sd, delta=dl)
            _, res = run_on(g3, venv); dv[dl] = (float(res['G'].mean()), g3.st.rolled / N_VAL)
        db = max(dv, key=lambda k: dv[k][0])
        cas = {}
        for f in FAMILIES3:
            g3 = Gate3(qb, actor, c, mode='cascade', model='ca', student=net, mu=mu, sd=sd, delta=db)
            e = Env(c, len(test[f])); load_set3(e, test[f]); e.done[:] = False
            res = run3(e, actor, g3, c); m = metrics(e, res)
            cas[f] = dict(G=res['G'], n_sw=m['n_sw'], n_cut_hard=m['n_cut_hard'], decisions=g3.st.rolled / len(test[f]),
                          min_front=m['min_front'], min_rear=m['min_rear'])
        out['cascade'] = dict(delta=db, val=dv, test=cas)
        log('cascade delta', db, 'G', round(float(np.concatenate([cas[f]['G'] for f in FAMILIES3]).mean()), 2),
            'sim decisions/ep', round(float(np.mean([cas[f]['decisions'] for f in FAMILIES3])), 2))
    # ---- wall clock (seed 0)
    if seed == 0 and 'timing' not in out:
        e1 = env_from_sampler(4242, 1); r = np.array([0]); acts = np.array([1 if e1.lane[0] > 0 else 2])
        g3 = Gate3(qb, actor, c, mode='student', student=net, mu=mu, sd=sd, tau=tau)
        t1 = time.perf_counter()
        for _ in range(200):
            g3.ghat(e1, r, acts); qb.forward(e1.critic_obs())
        out['timing'] = dict(student_ms=1e3 * (time.perf_counter() - t1) / 200)
    pickle.dump(out, open(fname, 'wb')); log('saved', fname)


if __name__ == '__main__':
    main(int(sys.argv[1]))
