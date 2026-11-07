"""Three-lane experiment (L = 3), one seed, end to end after the critics are trained.
Usage: N_LANES=3 DUAL_LANE=1 DUAL_MODE=hold CRITIC_NODIST=1 RESULTS3=dir python eval3l.py SEED
Steps: select the training-time switching cost on the critics' validation score; tune MOBIL on validation;
label proposals with the exact gate and train the student; tune tau on validation and set the conformal band;
evaluate all rules once on the held-out test families. Output RESULTS3/eval3l_s{SEED}.pkl (resumable by method).
"""
import sys, os, time, json, pickle, shutil, itertools
import numpy as np
from config import Cfg
from env import Env
from nn import MLP
from scenarios3 import FAMILIES3, Sampler3, make_test_set3, load_set3
from hl3 import (run3, metrics, rule_stay3, make_rule_greedy3, make_rule_mobil3, make_rule_predictive3, Gate3,
                 Stats, model_clone)
from amortize import train_student

R3 = os.environ.get('RESULTS3', os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, 'results3'))
N_TEST = int(os.environ.get('N_TEST', 100)); N_VAL = int(os.environ.get('N_VAL', 300))
N_LABEL_EP = int(os.environ.get('N_LABEL_EP', 4096))
C0S = {'': 2, 'c05_': 5, 'c010_': 10, 'c014_': 14, 'c020_': 20}


def ev(c, actor, rule, scen):
    env = Env(c, len(scen)); load_set3(env, scen); env.done[:] = False
    res = run3(env, actor, rule, c)
    m = metrics(env, res); m['fail_mask'] = res['fail_mask'].astype(float)
    return m


def main(seed):
    c = Cfg(); assert c.n_lanes == 3
    t0 = time.time(); log = lambda *a: print(f'[{time.time()-t0:7.1f}s]', *a, flush=True)
    actor = MLP.load(f'{R3}/ppo_s{seed}/actor.pkl')
    fname = f'{R3}/eval3l_s{seed}.pkl'
    out = pickle.load(open(fname, 'rb')) if os.path.exists(fname) else dict(seed=seed, test={}, rollouts={}, extra={})
    # ---- validated training cost
    sc = {c0: json.load(open(f'{R3}/dqn3_{t}s{seed}/dqn_log.json'))['best_val'] for t, c0 in C0S.items()}
    best_c0 = max(sc, key=sc.get); tag = [t for t, v in C0S.items() if v == best_c0][0]
    q2 = MLP.load(f'{R3}/dqn3_s{seed}/q.pkl'); qb = MLP.load(f'{R3}/dqn3_{tag}s{seed}/q.pkl')
    out['c0_val'] = sc; out['best_c0'] = best_c0
    # ---- sets
    test = {f: make_test_set3(c, f, N_TEST, seed=20_000 + 17 * i) for i, f in enumerate(FAMILIES3)}
    T = {f: test[f][0] for f in FAMILIES3}; out['rejects'] = {f: test[f][1] for f in FAMILIES3}
    vs = Sampler3(c, 55_555); val = []
    venv = Env(c, N_VAL); vs.reset_rows(venv, range(N_VAL))
    def vG(rule):
        e = venv.subset(np.arange(N_VAL)); e.done[:] = False
        return float(run3(e, actor, rule, c)['G'].mean())
    # ---- MOBIL tuning
    if 'mobil' not in out.get('tuned', {}):
        mb = {g: vG(make_rule_mobil3(actor, c, p=g[0], a_th=g[1])) for g in itertools.product([0.0, 0.25, 0.5, 1.0], [0.1, 0.3, 0.6, 1.0])}
        out.setdefault('tuned', {})['mobil'] = max(mb, key=mb.get); log('mobil', out['tuned']['mobil'])
    # ---- labels and student
    d = f'{R3}/amort3_s{seed}'; os.makedirs(d, exist_ok=True)
    if os.path.exists(f'{d}/labels.npz'):
        z = np.load(f'{d}/labels.npz'); X, Y = z['X'], z['Y']
    else:
        B = 64; env = Env(c, B); smp = Sampler3(c, 70_000 + seed); tg = Gate3(qb, actor, c, mode='teacher', record=True)
        for k in range(max(1, N_LABEL_EP // B)):
            smp.reset_rows(env, range(B)); env.done[:] = False; run3(env, actor, tg, c)
        X = np.concatenate(tg.X) if tg.X else np.zeros((0, 30)); Y = np.concatenate(tg.Y) if tg.Y else np.zeros(0)
        np.savez(f'{d}/labels.npz', X=X, Y=Y)
    net, mu, sd, cal = train_student(X, Y, seed)
    out['cal'] = cal; log('student', json.dumps(cal))
    if 'tau' not in out['tuned']:
        tv = {tau: vG(Gate3(qb, actor, c, mode='student', student=net, mu=mu, sd=sd, tau=tau)) for tau in [-2.0, -1.0, 0.0, 1.0, 2.0]}
        out['tuned']['tau'] = max(tv, key=tv.get); out['tuned']['tau_val'] = tv
    tau = out['tuned']['tau']; dconf = cal['delta']['0.1']; out['tuned']['delta_conf'] = dconf
    pm = out['tuned']['mobil']
    methods = {'stay_only': rule_stay3, 'mobil': make_rule_mobil3(actor, c, p=pm[0], a_th=pm[1]),
               'greedy': make_rule_greedy3(q2), 'best_pib': make_rule_greedy3(qb),
               'best_gate_ca': Gate3(qb, actor, c, mode='rollout', model='ca'),
               'best_teacher_true': Gate3(qb, actor, c, mode='teacher'),
               'best_student': Gate3(qb, actor, c, mode='student', student=net, mu=mu, sd=sd, tau=tau),
               'best_cascade_conf_ca': Gate3(qb, actor, c, mode='cascade', model='ca', student=net, mu=mu, sd=sd, delta=dconf),
               'best_student_cov': Gate3(qb, actor, c, mode='student', student=net, mu=mu, sd=sd, tau=tau, shadow=True),
               'predictive_ca': make_rule_predictive3(actor, c, 'ca'),
               'pred_ca_T5_P3': make_rule_predictive3(actor, c, 'ca', horizon=5.0, period=3.0)}
    for name, rule in methods.items():
        if name in out['test'] and len(out['test'][name]) == len(FAMILIES3):
            continue
        out['test'][name] = {}; out['rollouts'][name] = {}
        for f in FAMILIES3:
            if isinstance(rule, Gate3):
                rule.st = Stats(); rule.rec = []
            if hasattr(rule, 'n_eval'):
                rule.n_eval = 0; rule.n_roll = 0
            out['test'][name][f] = ev(c, actor, rule, T[f]); n = len(T[f])
            if isinstance(rule, Gate3):
                out['rollouts'][name][f] = dict(decisions=rule.st.rolled / n, rollouts=rule.st.rollouts / n, steps=rule.st.sim_steps / n)
                if rule.shadow:
                    out.setdefault('coverage', {})[f] = np.concatenate(rule.rec) if rule.rec else np.zeros((0, 3))
            elif hasattr(rule, 'n_eval'):
                out['rollouts'][name][f] = dict(decisions=rule.n_eval / n, rollouts=rule.n_roll / n)
        G = np.concatenate([out['test'][name][f]['G'] for f in FAMILIES3])
        log(name, f'G={G.mean():.2f}', 'LC', round(float(np.concatenate([out['test'][name][f]['n_sw'] for f in FAMILIES3]).mean()), 2))
        pickle.dump(out, open(fname, 'wb'))
    # ---- wall clock per decision (seed 0 only)
    if seed == 0 and 'timing' not in out['extra']:
        smp = Sampler3(c, 4242); e1 = Env(c, 1); tt = {'student': [], 'gate_exact': [], 'predictive_ca': []}
        st = Gate3(qb, actor, c, mode='student', student=net, mu=mu, sd=sd, tau=tau)
        for k in range(12):
            smp.reset_rows(e1, [0]); e1.done[:] = False; e1.eta[:] = 5.0
            r = np.array([0]); acts = np.array([1 if e1.lane[0] > 0 else 2])
            t1 = time.perf_counter()
            for _ in range(100): st.ghat(e1, r, acts); qb.forward(e1.critic_obs())
            tt['student'].append((time.perf_counter() - t1) / 100)
            from hl3 import rollout_pair3, make_rule_greedy3 as g3
            t1 = time.perf_counter(); rollout_pair3(e1, r, acts, actor, g3(qb), c, 'true'); tt['gate_exact'].append(time.perf_counter() - t1)
            pr = make_rule_predictive3(actor, c, 'ca')
            from hl3 import masks3
            t1 = time.perf_counter(); pr(e1, r, masks3(e1)[r]); tt['predictive_ca'].append(time.perf_counter() - t1)
        out['extra']['timing'] = {k: 1e3 * float(np.median(v)) for k, v in tt.items()}
    pickle.dump(out, open(fname, 'wb')); log('saved', fname)


if __name__ == '__main__':
    main(int(sys.argv[1]))
