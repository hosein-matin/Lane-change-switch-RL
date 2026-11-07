"""Sixth evaluation round: critic-free predictive selector at matched horizon / compute.
Usage: python evaluate6.py SEED -> results/eval6_s{SEED}.pkl (resumable)
All variants use the constant-acceleration forecast ('ca'), as the gate and governor do online.
  pred_ca_T5      evaluates every admissible decision, rollouts truncated at t + 5 s
  pred_ca_T5_P5   evaluates at most once per 5 s (first admissible decision after the previous evaluation + 5 s)
  pred_ca_P5      as above with full-horizon rollouts
The actor is the seed's actor (the selector does not use a critic).
"""
import sys, os, time, pickle
import numpy as np
from config import Cfg
from scenarios import FAMILIES
from nn import MLP
from hl import make_rule_predictive
import evaluate as E

R0 = os.environ.get('RESULTS', os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, 'results'))


def main(seed):
    c = Cfg(); t0 = time.time()
    actor = MLP.load(f'{R0}/ppo_s{seed}/actor.pkl')
    test, _ = E.build_sets(c); T = {f: test[f][0] for f in test}
    fname = f'{R0}/eval6_s{seed}.pkl'
    out = dict(seed=seed, test={}, n_eval={})
    if os.path.exists(fname):
        prev = pickle.load(open(fname, 'rb')); out['test'].update(prev['test']); out['n_eval'].update(prev['n_eval'])
    methods = {'pred_ca_T5_P5': dict(horizon=5.0, period=5.0),
               'pred_ca_T5_P3': dict(horizon=5.0, period=3.0),
               'pred_ca_P5': dict(horizon=None, period=5.0),
               'pred_ca_T5': dict(horizon=5.0, period=None)}
    for name, kw in methods.items():
        if name in out['test'] and len(out['test'][name]) == len(FAMILIES):
            continue
        out['test'][name] = {}; out['n_eval'][name] = {}
        for f in FAMILIES:
            rule = make_rule_predictive(actor, c, 'ca', **kw)
            m, _ = E.evaluate(c, actor, rule, T[f])
            out['test'][name][f] = m; out['n_eval'][name][f] = rule.n_eval / len(T[f])
        print(f'[{time.time()-t0:7.1f}s] done', name,
              {f: (round(float(out['test'][name][f]['G'].mean()), 1), round(float(out['test'][name][f]['n_sw'].mean()), 2),
                   round(out['n_eval'][name][f], 2)) for f in FAMILIES}, flush=True)
        pickle.dump(out, open(fname, 'wb'))
    print(f'[{time.time()-t0:7.1f}s] saved {fname}', flush=True)


if __name__ == '__main__':
    main(int(sys.argv[1]))
