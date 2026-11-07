"""Second evaluation round. Usage: python evaluate2.py SEED [mf]
Re-runs every method with the disruption metrics, and adds the governor without exact knowledge of the
future ('pred', 'robust') and the prediction-gated rule ('gate_true', 'gate_pred').
Tuned parameters of the first round (eval_s{SEED}.pkl) are reused; rho of the gate is tuned here on
the validation set. With 'mf' it evaluates the models trained with the model-free follower bound.
"""
import sys, os, time, pickle, itertools
import numpy as np
from config import Cfg
from scenarios import FAMILIES
from nn import MLP
from hl import (make_rule_margin, make_rule_mobil, make_rule_predictive, rule_stay_only,
                BudgetGovernor, PredictionGate)
import evaluate as E

RHO_GRID = [0.0, 0.01, 0.03, 0.1]


def main(seed, mf=False):
    c = Cfg()
    assert (c.follower_bound == 'max') == mf, 'set FOLLOWER_BOUND=max for the mf run'
    t0 = time.time()
    R0 = os.environ.get('RESULTS', os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, 'results'))
    R = os.environ.get('RESULTS_MF', os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, 'results_mf')) if mf else R0
    actor = MLP.load(f'{R}/ppo_s{seed}/actor.pkl'); q = MLP.load(f'{R}/dqn_s{seed}/q.pkl')
    tuned_prev = pickle.load(open(f'{R0}/eval_s{seed}.pkl', 'rb'))['tuned']
    test, val = E.build_sets(c)
    T = {f: test[f][0] for f in test}
    out = dict(seed=seed, mf=mf, cfg=c.as_dict(), test={}, gov={}, val={}, tuned=dict(tuned_prev))
    fname = f'{R}/eval2_s{seed}.pkl'

    def log(*a):
        print(f'[{time.time()-t0:7.1f}s]', *a, flush=True)

    def vG(rule, gov=None):
        m, _ = E.evaluate(c, actor, rule, val, gov)
        return float(m['G'].mean()), float(m['n_sw'].mean())

    if mf:
        # re-tune MOBIL and the governor for the retrained models (small grids, as in round 1)
        mb = {g: vG(make_rule_mobil(actor, c, p=g[0], a_th=g[1]))[0]
              for g in itertools.product([0.0, 0.25, 0.5, 1.0], [0.1, 0.3, 0.6, 1.0])}
        gv = {}
        for B0, b1 in itertools.product([5.0, 15.0], [4.0, 16.0]):
            g = BudgetGovernor([q], actor, c, B0=B0, beta0=1.0, beta1=b1)
            gv[(B0, b1)] = vG(g, g)[0]
        out['tuned']['mobil'] = max(mb, key=mb.get); out['tuned']['governor'] = max(gv, key=gv.get)
        out['val'].update(mobil=mb, governor=gv)
        log('mf tuned', out['tuned'])
    tn = out['tuned']
    # tune the gate threshold rho (exact-model gate); the predicted-model gate reuses it
    gt = {rho: vG(PredictionGate([q], actor, c, rho=rho, model='true')) for rho in RHO_GRID}
    out['val']['gate'] = gt
    rho = max(gt, key=lambda r: gt[r][0]); out['tuned']['gate_rho'] = rho
    log('gate tuned', gt, '->', rho)

    B0, b1 = tn['governor']
    methods = {
        'stay_only': (rule_stay_only, None),
        'mobil': (make_rule_mobil(actor, c, p=tn['mobil'][0], a_th=tn['mobil'][1]), None),
        'greedy': (make_rule_margin([q], c, 0.0, 0.0), None),
    }
    if not mf:
        methods['fixed'] = (make_rule_margin([q], c, tn['fixed'], 0.0), None)
        methods['statedep'] = (make_rule_margin([q], c, tn['statedep'][0], tn['statedep'][1]), None)
    for model in (['true', 'pred', 'robust'] if not mf else ['true', 'pred']):
        g = BudgetGovernor([q], actor, c, B0=B0, beta0=1.0, beta1=b1, model=model)
        methods['gov_' + model] = (g, g)
    for model in ['true', 'pred']:
        methods['gate_' + model] = (PredictionGate([q], actor, c, rho=rho, model=model), None)
    if not mf:
        methods['predictive'] = (make_rule_predictive(actor, c), None)

    for name, (rule, g) in methods.items():
        out['test'][name] = {}; out['gov'][name] = {}
        for f in FAMILIES:
            m, st = E.evaluate(c, actor, rule, T[f], g)
            out['test'][name][f] = m
            if st is not None:
                out['gov'][name][f] = st
        log('done', name, {f: (round(float(out['test'][name][f]['G'].mean()), 1),
                               round(float(out['test'][name][f]['n_sw'].mean()), 2)) for f in FAMILIES})
        pickle.dump(out, open(fname, 'wb'))

    if mf:
        # safety without the human model: nominal and with a mismatched follower gain
        from env import Env
        from scenarios import load_set
        from hl import run, metrics
        allT = sum([T[f] for f in FAMILIES], [])
        out['mismatch'] = {}
        for kind in ['nominal', 'follower_alpha']:
            env = Env(c, len(allT)); load_set(env, allT); env.done[:] = False
            if kind == 'follower_alpha':
                env.f_alpha[:] = 0.3
            res = run(env, actor, make_rule_margin([q], c, 0.0, 0.0), c)
            out['mismatch'][kind] = metrics(env, res)
        log('mf mismatch done')
    pickle.dump(out, open(fname, 'wb'))
    log('saved', fname)


if __name__ == '__main__':
    main(int(sys.argv[1]), len(sys.argv) > 2 and sys.argv[2] == 'mf')
