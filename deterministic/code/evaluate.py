"""Evaluation protocol (paper Sec. V). Usage: python evaluate.py SEED
Writes results/eval_s{SEED}.pkl with per-scenario metrics for every method x family, the tuned
parameters, and the stress tests.
"""
import sys, os, json, time, pickle, itertools
import numpy as np
from dataclasses import replace
from config import Cfg
from env import Env
from scenarios import FAMILIES, make_test_set, load_set, TRAIN_MIX, sample_params
from nn import MLP
from hl import (run, metrics, make_rule_margin, make_rule_mobil, make_rule_predictive,
                rule_stay_only, BudgetGovernor)

RES = os.environ.get('RESULTS', os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, 'results'))
N_TEST = int(os.environ.get('N_TEST', 100))
N_VAL = int(os.environ.get('N_VAL', 300))


def build_sets(c):
    test = {f: make_test_set(c, f, N_TEST, seed=10_000 + 17 * i) for i, f in enumerate(FAMILIES)}
    # validation: training mixture, disjoint seed
    rng = np.random.default_rng(55_555)
    fams = list(TRAIN_MIX); pr = np.array([TRAIN_MIX[f] for f in fams]); pr /= pr.sum()
    val = []
    for j in range(N_VAL):
        f = fams[rng.choice(len(fams), p=pr)]
        s, _ = make_test_set(c, f, 1, seed=900_000 + j)
        val += s
    return test, val


def evaluate(c, actor, rule, scen, governor=None):
    env = Env(c, len(scen)); load_set(env, scen); env.done[:] = False
    gov_stats = None
    if governor is not None:
        governor.reset(env.B)
        governor.stats = dict(proposals=0, accepted=0, debit=0.0, base_infeasible=0)
        governor.check_initial(env)
    res = run(env, actor, rule, c)
    m = metrics(env, res)
    m['n_both'] = res['n_both'].astype(float); m['n_dec'] = res['n_dec'].astype(float)
    m['fail_mask'] = res['fail_mask'].astype(float); m['fail_int'] = res['fail_int'].astype(float)
    if governor is not None:
        m['budget_used'] = governor.B0 - governor.budget
        gov_stats = dict(governor.stats)
    return m, gov_stats


def main(seed):
    c = Cfg()
    t0 = time.time()
    actor = MLP.load(f'{RES}/ppo_s{seed}/actor.pkl')
    q = MLP.load(f'{RES}/dqn_s{seed}/q.pkl')
    test, val = build_sets(c)
    out = dict(seed=seed, cfg=c.as_dict(), rejects={f: test[f][1] for f in test}, tuned={}, val={}, test={}, gov={})
    T = {f: test[f][0] for f in test}

    def log(*a):
        print(f'[{time.time()-t0:7.1f}s]', *a, flush=True)

    # ---------------- tuning on validation (fixed external score = realised return G incl. C_fail)
    out['val_detail'] = {}
    def vscore(rule, gov=None, key=None):
        m, _ = evaluate(c, actor, rule, val, gov)
        if key is not None:
            out['val_detail'][key] = dict(G=float(m['G'].mean()), n_sw=float(m['n_sw'].mean()),
                                          fail=float(m['fail'].mean()), score=float(m['score'].mean()))
        return float(m['G'].mean())
    greedy = make_rule_margin([q], c, 0.0, 0.0)
    out['val']['greedy'] = vscore(greedy, key=('greedy', 0.0, 0.0))
    fixed_grid = [0.5, 1.0, 2.0, 3.0, 5.0, 8.0]
    fx = {th: vscore(make_rule_margin([q], c, th, 0.0), key=('fixed', th, 0.0)) for th in fixed_grid}
    out['val']['fixed'] = fx
    th_fixed = max(fx, key=fx.get)
    sd_grid = list(itertools.product([0.0, 0.5, 1.0, 2.0], [1.0, 2.0, 4.0, 8.0, 16.0]))
    sd = {g: vscore(make_rule_margin([q], c, g[0], g[1]), key=('statedep',) + g) for g in sd_grid}
    out['val']['statedep'] = sd
    th_sd = max(sd, key=sd.get)
    mob_grid = list(itertools.product([0.0, 0.25, 0.5, 1.0], [0.1, 0.3, 0.6, 1.0]))
    mb = {g: vscore(make_rule_mobil(actor, c, p=g[0], a_th=g[1]), key=('mobil',) + g) for g in mob_grid}
    out['val']['mobil'] = mb
    p_mob = max(mb, key=mb.get)
    gov_grid = list(itertools.product([5.0, 15.0], [4.0, 16.0]))
    gv = {}
    for B0, b1 in gov_grid:
        gov = BudgetGovernor([q], actor, c, B0=B0, beta0=1.0, beta1=b1)
        gv[(B0, b1)] = vscore(gov, gov, key=('governor', B0, b1))
    out['val']['governor'] = gv
    g_best = max(gv, key=gv.get)
    out['tuned'] = dict(fixed=th_fixed, statedep=th_sd, mobil=p_mob, governor=g_best)
    log('tuned', out['tuned'])

    methods = {
        'stay_only': (rule_stay_only, None),
        'mobil': (make_rule_mobil(actor, c, p=p_mob[0], a_th=p_mob[1]), None),
        'greedy': (greedy, None),
        'fixed': (make_rule_margin([q], c, th_fixed, 0.0), None),
        'statedep': (make_rule_margin([q], c, th_sd[0], th_sd[1]), None),
    }
    gov = BudgetGovernor([q], actor, c, B0=g_best[0], beta0=1.0, beta1=g_best[1])
    methods['governor'] = (gov, gov)
    methods['predictive'] = (make_rule_predictive(actor, c), None)

    for name, (rule, g) in methods.items():
        out['test'][name] = {}
        for f in FAMILIES:
            m, st = evaluate(c, actor, rule, T[f], g)
            out['test'][name][f] = m
            if st is not None:
                out['gov'][f] = st
        log('done', name, {f: round(float(out['test'][name][f]['G'].mean()), 1) for f in FAMILIES})
        pickle.dump(out, open(f'{RES}/eval_s{seed}.pkl', 'wb'))

    # ---------------- dwell ablation (deployment mask only; critic trained with tau_d = 2 s)
    out['dwell'] = {}
    allT = sum([T[f] for f in FAMILIES], [])
    for td in [1.0, 2.0, 4.0]:
        cc = replace(c, tau_d=td)
        for name, rule in [('greedy', make_rule_margin([q], cc, 0.0, 0.0)),
                           ('statedep', make_rule_margin([q], cc, th_sd[0], th_sd[1]))]:
            m, _ = evaluate(cc, actor, rule, allT)
            out['dwell'][(td, name)] = m
    log('dwell done')

    # ---------------- timestep dependence (safety): same actor/critic, T_m kept at 1 s
    out['dt'] = {}
    for dt in [0.05, 0.1, 0.2]:
        cc = replace(c, dt=dt, N=int(round(1.0 / dt)))
        rule = make_rule_margin([q], cc, th_sd[0], th_sd[1])
        m, _ = evaluate(cc, actor, rule, allT)
        out['dt'][dt] = m
    log('dt done')

    # ---------------- model mismatch: certificate assumptions violated on purpose
    out['mismatch'] = {}
    rng = np.random.default_rng(4242)
    for kind in ['nominal', 'follower_alpha', 'leader_pulse']:
        env = Env(c, len(allT)); load_set(env, allT); env.done[:] = False
        if kind == 'follower_alpha':
            env.f_alpha[:] = 0.3                     # true gain << declared 0.8
        if kind == 'leader_pulse':
            # only leaders whose speed stays nonnegative under the extra pulse
            env.pulse_t[:] = np.where(env.vbar - env.A > 4.6, rng.uniform(1, 12, (env.B, 2)), np.inf)
            env.pulse_T[:] = 1.5
            env.pulse_a[:] = -3.0                    # extra braking beyond declared -A|w|
        res = run(env, actor, make_rule_margin([q], c, th_sd[0], th_sd[1]), c)
        out['mismatch'][kind] = metrics(env, res)
    log('mismatch done')
    pickle.dump(out, open(f'{RES}/eval_s{seed}.pkl', 'wb'))
    log('saved')


if __name__ == '__main__':
    main(int(sys.argv[1]))
