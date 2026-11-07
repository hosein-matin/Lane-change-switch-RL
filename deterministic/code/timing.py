"""Wall-clock per decision (single thread): critic, student, filtered control step, and the two-rollout gate
decision with exact / constant-acceleration rollouts, to T_f and truncated at 5 s.
Usage: python timing.py [SEED] [TAG] -> RESULTS/timing.json
"""
import sys, os, time, json
import numpy as np
from config import Cfg
from env import Env
from scenarios import Sampler
from nn import MLP
from hl import make_rule_margin, u_from_a, model_clone, run, rule_stay_only, CHANGE, STAY
import amortize as A

R0 = A.R0


def main(seed=0, tag='best'):
    c = Cfg()
    actor = MLP.load(f'{R0}/ppo_s{seed}/actor.pkl'); q = MLP.load(f'{R0}/dqn_{tag}_s{seed}/q.pkl')
    net = MLP.load(f'{R0}/amort_{tag}_s{seed}/student.pkl'); nz = np.load(f'{R0}/amort_{tag}_s{seed}/norm.npz')
    base = make_rule_margin([q], c, 0, 0)
    env = Env(c, 1); smp = Sampler(c, 4242)
    tt = {k: [] for k in ['critic', 'student', 'control_step', 'gate_exact', 'gate_ca', 'gate_exact_T5', 'gate_ca_T5']}
    for k in range(20):
        smp.reset_rows(env, [0]); env.done[:] = False; env.eta[:] = 5.0
        t0 = time.perf_counter()
        for _ in range(200): q.forward(env.critic_obs())
        tt['critic'].append((time.perf_counter() - t0) / 200)
        t0 = time.perf_counter()
        for _ in range(200): net.forward((A.feats(env) - nz['mu']) / nz['sd'])
        tt['student'].append((time.perf_counter() - t0) / 200)
        t0 = time.perf_counter()
        for _ in range(100): env.filt_cur(u_from_a(actor.forward(env.actor_obs())[:, 0], c))
        tt['control_step'].append((time.perf_counter() - t0) / 100)
        for model in ['true', 'ca']:
            for H in [None, 5.0]:
                key = f"gate_{'exact' if model == 'true' else 'ca'}" + ('' if H is None else '_T5')
                t0 = time.perf_counter()
                sc = model_clone(env, np.array([0]), 1, model); ss = model_clone(env, np.array([0]), 1, model)
                te = None if H is None else sc.t + H
                run(sc, actor, base, c, forced_first=np.array([CHANGE]), t_end=te)
                run(ss, actor, rule_stay_only, c, forced_first=np.array([STAY]), t_end=te)
                tt[key].append(time.perf_counter() - t0)
    out = {k: dict(median_ms=1e3 * float(np.median(v)), p90_ms=1e3 * float(np.percentile(v, 90))) for k, v in tt.items()}
    import platform
    out['machine'] = dict(processor=platform.processor(), python=platform.python_version(), numpy=np.__version__)
    json.dump(out, open(f'{R0}/timing.json', 'w'), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == '__main__':
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 0, sys.argv[2] if len(sys.argv) > 2 else 'best')
