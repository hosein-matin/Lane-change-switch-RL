"""Fifth evaluation round: critics trained under modified objectives / dynamics.
Usage: [DUAL_LANE=1] python evaluate5.py SEED VARIANT  -> results/eval5_{VARIANT}_s{SEED}.pkl (resumable)
VARIANT is the critic directory tag (results/dqn_{VARIANT}_s{SEED}/q.pkl), e.g.
  c05, c010   critic trained with a larger switching cost c0 (deployment still uses c0 = 2)
  cf1, cf5    counterfactual-shaped critic (1 s window / 5 s rollout comparison)
  dual        critic trained with the two-lane commitment (requires DUAL_LANE=1; baselines re-run)
Tuned rho, B0, beta1 and MOBIL parameters are reused from rounds 1-2 of the same seed.
"""
import sys, os, time, pickle
from config import Cfg
from scenarios import FAMILIES
from nn import MLP
from hl import make_rule_margin, make_rule_mobil, rule_stay_only, BudgetGovernor, PredictionGate
import evaluate as E

R0 = os.environ.get('RESULTS', os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, 'results'))


def main(seed, variant):
    c = Cfg(); t0 = time.time()
    if variant.startswith('dual'):
        assert c.dual_lane
    actor = MLP.load(f'{R0}/ppo_s{seed}/actor.pkl'); q = MLP.load(f'{R0}/dqn_{variant}_s{seed}/q.pkl')
    tn1 = pickle.load(open(f'{R0}/eval_s{seed}.pkl', 'rb'))['tuned']
    rho = pickle.load(open(f'{R0}/eval2_s{seed}.pkl', 'rb'))['tuned']['gate_rho']
    B0, b1 = tn1['governor']; p_mob = tn1['mobil']
    test, _ = E.build_sets(c); T = {f: test[f][0] for f in test}
    fname = f'{R0}/eval5_{variant}_s{seed}.pkl'
    out = dict(seed=seed, variant=variant, test={}, gov={}, tuned=dict(governor=(B0, b1), gate_rho=rho))
    if os.path.exists(fname):
        prev = pickle.load(open(fname, 'rb')); out['test'].update(prev['test']); out['gov'].update(prev['gov'])
    methods = {'pib': (make_rule_margin([q], c, 0.0, 0.0), None),
               'gate_ca': (PredictionGate([q], actor, c, rho=rho, model='ca'), None)}
    if variant.startswith('dual'):
        methods['stay_only'] = (rule_stay_only, None)
        methods['mobil'] = (make_rule_mobil(actor, c, p=p_mob[0], a_th=p_mob[1]), None)
        methods['gate_true'] = (PredictionGate([q], actor, c, rho=rho, model='true'), None)
        for model in ['true', 'ca']:
            g = BudgetGovernor([q], actor, c, B0=B0, beta0=1.0, beta1=b1, model=model)
            methods['gov_' + model] = (g, g)
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
              {f: (round(float(out['test'][name][f]['G'].mean()), 1), round(float(out['test'][name][f]['n_sw'].mean()), 2),
                   int(out['test'][name][f]['fail'].sum())) for f in FAMILIES}, flush=True)
        pickle.dump(out, open(fname, 'wb'))
    print(f'[{time.time()-t0:7.1f}s] saved {fname}', flush=True)


if __name__ == '__main__':
    main(int(sys.argv[1]), sys.argv[2])
