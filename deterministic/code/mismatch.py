"""Model-mismatch test traffic (applied after loading a test set; deterministic per scenario).
Leaders: piecewise-constant accelerations instead of sinusoids, redrawn every 0.5-3 s within the declared
bound [-A|w|, +A|w|] (so the safety filter's hypotheses still hold), with frequent hard braking.
Followers: desired speeds scaled by a factor in [0.85, 1.15] (the declared follower bound uses the true value).
Nothing is retrained: all models, thresholds and forecasts stay those of the nominal traffic.
"""
import numpy as np


def apply_mismatch(env, seed=0):
    c = env.c; L = env.L; K = env.SCH_K
    for r in range(env.B):
        rng = np.random.default_rng(1_000_003 * (seed + 1) + r)
        for k in range(L):
            b = env.A[r, k] * abs(env.w[r, k])
            dur = rng.uniform(0.5, 3.0, K)
            t0 = np.concatenate([[0.0], np.cumsum(dur)[:-1]])
            hard = rng.random(K) < 0.35
            a = np.where(hard, rng.uniform(-1.0, -0.5, K), rng.uniform(-0.5, 1.0, K)) * b
            env.sch_t[r, k] = t0; env.sch_a[r, k] = a
        env.lmode[r, :] = 2
        f = rng.uniform(0.85, 1.15)
        env.fvmax[r, :] = np.clip(env.fvmax[r, :] * f, 10.0, 22.0)
    return env
