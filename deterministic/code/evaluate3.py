"""Third evaluation round. Usage:
  python evaluate3.py SEED forecast   -> forecast variants with the original models
  CRITIC_NODIST=1 python evaluate3.py SEED nodist -> critic trained without disturbance parameters
Tuned parameters (B0, beta1, rho) are reused from rounds 1-2 of the same seed.
"""
import sys, os, time, pickle
import numpy as np
from config import Cfg
from scenarios import FAMILIES
from nn import MLP
from hl import make_rule_margin, make_rule_predictive, BudgetGovernor, PredictionGate
import evaluate as E

R0 = os.environ.get('RESULTS', os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, 'results')); RN = os.environ.get('RESULTS_ND', os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, 'results_nd'))


def main(seed, mode):
    c = Cfg()
    if mode == 'nodist':
        assert c.critic_nodist
    t0 = time.time()
    actor = MLP.load(f'{R0}/ppo_s{seed}/actor.pkl')
    q = MLP.load(f'{RN if mode == "nodist" else R0}/dqn_s{seed}/q.pkl')
    tn1 = pickle.load(open(f'{R0}/eval_s{seed}.pkl', 'rb'))['tuned']
    rho = pickle.load(open(f'{R0}/eval2_s{seed}.pkl', 'rb'))['tuned']['gate_rho']
    B0, b1 = tn1['governor']
    test, _ = E.build_sets(c)
    T = {f: test[f][0] for f in test}
    out = dict(seed=seed, mode=mode, test={}, gov={}, tuned=dict(governor=(B0, b1), gate_rho=rho))
    fname = f'{RN if mode == "nodist" else R0}/eval3_s{seed}.pkl'
    import os
    if os.path.exists(fname):
        prev = pickle.load(open(fname, 'rb'))
        out['test'].update(prev['test']); out['gov'].update(prev['gov'])
    if mode == 'forecast':
        methods = {'predictive_pred': (make_rule_predictive(actor, c, 'pred'), None),
                   'predictive_ca': (make_rule_predictive(actor, c, 'ca'), None)}
        g = BudgetGovernor([q], actor, c, B0=B0, beta0=1.0, beta1=b1, model='ca'); methods['gov_ca'] = (g, g)
        methods['gate_ca'] = (PredictionGate([q], actor, c, rho=rho, model='ca'), None)
        methods['gate_robust'] = (PredictionGate([q], actor, c, rho=rho, model='robust'), None)
    else:
        methods = {'greedy': (make_rule_margin([q], c, 0.0, 0.0), None),
                   'gate_true': (PredictionGate([q], actor, c, rho=rho, model='true'), None)}
        g = BudgetGovernor([q], actor, c, B0=B0, beta0=1.0, beta1=b1, model='true'); methods['gov_true'] = (g, g)
    for name, (rule, gv) in methods.items():
        if name in out['test'] and len(out['test'][name]) == len(FAMILIES):
            continue
        out['test'][name] = {}; out['gov'][name] = {}
        for f in FAMILIES:
            m, st = E.evaluate(c, actor, rule, T[f], gv)
            out['test'][name][f] = m
            if st is not None:
                out['gov'][name][f] = st
        print(f'[{time.time()-t0:7.1f}s] done', name,
              {f: (round(float(out['test'][name][f]['G'].mean()), 1), round(float(out['test'][name][f]['n_sw'].mean()), 2))
               for f in FAMILIES}, flush=True)
        pickle.dump(out, open(fname, 'wb'))
    print(f'[{time.time()-t0:7.1f}s] saved {fname}', flush=True)


if __name__ == '__main__':
    main(int(sys.argv[1]), sys.argv[2])
