"""Model-mismatch evaluation: all models, thresholds and forecasts from the nominal traffic, tested on
the same held-out scenarios with mismatched traffic (mismatch.py). Nothing is retrained or retuned.
Usage: python evaluate_mis.py SEED [TAG]  -> RESULTS/evalmis_s{SEED}.pkl (resumable)
Methods: stay, MOBIL, critic c0=2, critic with validated c0 (TAG='best'), forecast gate on it, student,
conformal cascade (forecast fallback), predictive (forecast) at every decision and at most once per 3 s,
and the exact gate as an oracle reference (its rollouts know the realised mismatched traffic).
"""
import sys, os, time, pickle
import numpy as np
from config import Cfg
from env import Env
from scenarios import FAMILIES, load_set
from nn import MLP
from hl import run, metrics, make_rule_margin, make_rule_mobil, rule_stay_only, make_rule_predictive, PredictionGate
from mismatch import apply_mismatch
from amortize import AmortGate, TeacherGate, Stats
import evaluate as E

R0 = os.environ.get('RESULTS', os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, 'results'))


def ev(c, actor, rule, scen):
    env = Env(c, len(scen)); load_set(env, scen); env.done[:] = False
    apply_mismatch(env, 0)
    res = run(env, actor, rule, c)
    m = metrics(env, res)
    m['fail_mask'] = res['fail_mask'].astype(float)
    return m


def main(seed, tag='best'):
    c = Cfg(); t0 = time.time()
    actor = MLP.load(f'{R0}/ppo_s{seed}/actor.pkl')
    q2 = MLP.load(f'{R0}/dqn_s{seed}/q.pkl'); qb = MLP.load(f'{R0}/dqn_{tag}_s{seed}/q.pkl')
    p_mob = pickle.load(open(f'{R0}/eval_s{seed}.pkl', 'rb'))['tuned']['mobil']
    rho = pickle.load(open(f'{R0}/eval2_s{seed}.pkl', 'rb'))['tuned']['gate_rho']
    e8 = pickle.load(open(f'{R0}/eval8_{tag}_s{seed}.pkl', 'rb'))
    d = f'{R0}/amort_{tag}_s{seed}'; net = MLP.load(f'{d}/student.pkl'); nz = np.load(f'{d}/norm.npz')
    tau, dconf = e8['tuned']['tau'], e8['tuned']['delta_conf']
    test, _ = E.build_sets(c); T = {f: test[f][0] for f in test}
    fname = f'{R0}/evalmis_s{seed}.pkl'
    out = pickle.load(open(fname, 'rb')) if os.path.exists(fname) else dict(seed=seed, test={}, rollouts={})
    methods = {'stay_only': rule_stay_only, 'mobil': make_rule_mobil(actor, c, p=p_mob[0], a_th=p_mob[1]),
               'greedy': make_rule_margin([q2], c, 0.0, 0.0), 'best_pib': make_rule_margin([qb], c, 0.0, 0.0),
               'best_gate_ca': PredictionGate([qb], actor, c, rho=rho, model='ca'),
               'best_student': AmortGate(qb, actor, c, net, nz['mu'], nz['sd'], tau=tau),
               'best_cascade_conf_ca': AmortGate(qb, actor, c, net, nz['mu'], nz['sd'], delta=dconf, fallback='ca'),
               'best_teacher_oracle': TeacherGate(qb, actor, c),
               'pred_ca_T5_P3': make_rule_predictive(actor, c, 'ca', horizon=5.0, period=3.0),
               'predictive_ca': make_rule_predictive(actor, c, 'ca')}
    for name, rule in methods.items():
        if name in out['test'] and len(out['test'][name]) == len(FAMILIES):
            continue
        out['test'][name] = {}; out['rollouts'][name] = {}
        for f in FAMILIES:
            if isinstance(rule, (AmortGate, TeacherGate)):
                rule.st = Stats()
            if isinstance(rule, PredictionGate):
                rule.stats = dict(proposals=0, accepted=0)
            if hasattr(rule, 'n_eval'):
                rule.n_eval = 0
            out['test'][name][f] = ev(c, actor, rule, T[f])
            n = len(T[f])
            if isinstance(rule, (AmortGate, TeacherGate)):
                out['rollouts'][name][f] = rule.st.rolled / n
            elif hasattr(rule, 'n_eval'):
                out['rollouts'][name][f] = rule.n_eval / n
            elif isinstance(rule, PredictionGate):
                out['rollouts'][name][f] = rule.stats['proposals'] / n
        G = np.concatenate([out['test'][name][f]['G'] for f in FAMILIES])
        print(f'[{time.time()-t0:7.1f}s] {name}: G={G.mean():.2f}', flush=True)
        pickle.dump(out, open(fname, 'wb'))
    print(f'[{time.time()-t0:7.1f}s] saved {fname}', flush=True)


if __name__ == '__main__':
    main(int(sys.argv[1]), sys.argv[2] if len(sys.argv) > 2 else 'best')
