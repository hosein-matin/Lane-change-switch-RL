"""Trade-off sweeps for the amortized gate on the test sets (characterisation, not tuning).
Usage: python amortize_sweep.py SEED [TAG]  -> RESULTS/sweep8_{TAG}_s{SEED}.pkl (resumable)
  tau sweep (student only, no rollouts): return, lane changes, hard cut-ins vs tau
  delta sweep (cascade, constant-acceleration fallback): return vs rollouts per episode
"""
import sys, os, time, pickle
import numpy as np
from config import Cfg
from scenarios import FAMILIES
from nn import MLP
import evaluate as E
from amortize import AmortGate, Stats, R0

TAUS = [-2.0, -1.0, 0.0, 1.0, 2.0, 4.0, 6.0, 8.0, 12.0]
DELTAS = [0.5, 1.0, 2.0, 4.0, 6.0, 10.0, 15.0, 25.0]


def main(seed, tag='best'):
    c = Cfg(); t0 = time.time()
    actor = MLP.load(f'{R0}/ppo_s{seed}/actor.pkl'); q = MLP.load(f'{R0}/dqn_{tag}_s{seed}/q.pkl')
    d = f'{R0}/amort_{tag}_s{seed}'
    net = MLP.load(f'{d}/student.pkl'); nz = np.load(f'{d}/norm.npz')
    test, _ = E.build_sets(c); T = {f: test[f][0] for f in test}
    fname = f'{R0}/sweep8_{tag}_s{seed}.pkl'
    out = pickle.load(open(fname, 'rb')) if os.path.exists(fname) else dict(seed=seed, tau={}, delta={})
    keys = ['G', 'n_sw', 'n_cut_hard', 'hb_time']
    for tau in TAUS:
        if tau in out['tau']:
            continue
        r = AmortGate(q, actor, c, net, nz['mu'], nz['sd'], tau=tau)
        ms = {f: E.evaluate(c, actor, r, T[f])[0] for f in FAMILIES}
        out['tau'][tau] = {k: np.concatenate([ms[f][k] for f in FAMILIES]) for k in keys}
        pickle.dump(out, open(fname, 'wb'))
    for dl in DELTAS:
        if dl in out['delta']:
            continue
        r = AmortGate(q, actor, c, net, nz['mu'], nz['sd'], delta=dl, fallback='ca')
        ms = {}; rolled = 0; steps = 0; n = 0
        for f in FAMILIES:
            r.st = Stats(); ms[f] = E.evaluate(c, actor, r, T[f])[0]
            rolled += r.st.rolled; steps += r.st.sim_steps; n += len(T[f])
        out['delta'][dl] = {k: np.concatenate([ms[f][k] for f in FAMILIES]) for k in keys}
        out['delta'][dl].update(rolled=rolled / n, sim_steps=steps / n)
        pickle.dump(out, open(fname, 'wb'))
        print(f'[{time.time()-t0:.0f}s] delta {dl}: G={out["delta"][dl]["G"].mean():.2f} rolled={rolled/n:.2f}', flush=True)
    print(f'[{time.time()-t0:.0f}s] saved {fname}', flush=True)


if __name__ == '__main__':
    main(int(sys.argv[1]), sys.argv[2] if len(sys.argv) > 2 else 'best')
