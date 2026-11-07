"""Stage 1 of Algorithm 1: longitudinal PPO in full fixed-lane episodes.

Nominal action a ~ N(mu(o), sigma), env uses clip(a,-1,1) mapped by (eq:actionmap); the executed
input is the filtered u. The NOMINAL action and its (unsquashed Gaussian) log-probability are stored.
Reward r_n = w_d (x_{n+1}-x_n) - w_u u_n^2 dt (executed u). Empty I^Delta => failure, -C_fail once,
episode terminates. gamma = 1, GAE(lambda), remaining time in the observation.
Usage: python ppo.py SEED ITERS OUTDIR
"""
import sys, os, json, time
import numpy as np
from config import Cfg
from env import Env
from scenarios import Sampler
from nn import MLP, Adam


def u_from_a(a, c):
    a = np.clip(a, -1.0, 1.0)
    return c.u_lo + 0.5 * (a + 1.0) * (c.u_hi - c.u_lo)


def run_actor_episodes(actor, c, env, max_steps=None):
    """Deterministic fixed-lane rollout of an already-initialised env batch. Returns returns, fails."""
    B = env.B
    ret = np.zeros(B); fail = np.zeros(B, bool); live = np.ones(B, bool)
    for n in range(max_steps or c.M):
        o = env.actor_obs()
        uc = u_from_a(actor.forward(o)[:, 0], c)
        u, feas, act = env.filt_cur(uc)
        newfail = live & ~feas
        ret -= np.where(newfail, c.C_fail, 0); fail |= newfail; live &= ~newfail
        r = env.step(u, active_mask=live)
        ret += r
        live &= env.t < c.Tf - 1e-9
        if not live.any():
            break
    return ret, fail


def train(seed, iters, outdir, B=64, lr_pi=3e-4, lr_v=1e-3, epochs=8, nmb=8, clip=0.2, lam=0.95, ent=1e-3):
    c = Cfg()
    os.makedirs(outdir, exist_ok=True)
    rng = np.random.default_rng(seed)
    obs_dim = 10
    actor = MLP([obs_dim, 64, 64, 1], rng, out_scale=0.01)
    critic = MLP([obs_dim, 64, 64, 1], rng, out_scale=1.0)
    logstd = np.array([-0.5])
    opt_pi = Adam(actor.params() + [logstd], lr=lr_pi)
    opt_v = Adam(critic.params(), lr=lr_v)
    sampler = Sampler(c, seed + 1000)
    env = Env(c, B, rng)
    sampler.reset_rows(env, range(B))
    # fixed validation batch
    vs = Sampler(c, 99_999)
    venv0 = Env(c, 128); vs.reset_rows(venv0, range(128))
    log = []; best = -np.inf; t0 = time.time()
    T = c.M
    for it in range(iters):
        O = np.zeros((T, B, obs_dim)); Aa = np.zeros((T, B)); LP = np.zeros((T, B))
        R = np.zeros((T, B)); D = np.zeros((T, B)); Vv = np.zeros((T, B))
        ep_ret = []; ep_fail = []
        if not hasattr(env, 'acc'):
            env.acc = np.zeros(B)
        sig = np.exp(logstd[0])
        for n in range(T):
            o = env.actor_obs()
            mu = actor.forward(o)[:, 0]
            a = mu + sig * rng.standard_normal(B)
            lp = -0.5 * ((a - mu) / sig) ** 2 - logstd[0] - 0.5 * np.log(2 * np.pi)
            uc = u_from_a(a, c)
            u, feas, act = env.filt_cur(uc)
            r = np.where(feas, 0.0, -c.C_fail)
            r = r + env.step(u, active_mask=feas)
            done = ~feas | (env.t >= c.Tf - 1e-9)
            O[n] = o; Aa[n] = a; LP[n] = lp; R[n] = r * c.rew_scale; D[n] = done
            Vv[n] = critic.forward(o)[:, 0]
            env.acc += r
            if done.any():
                idx = np.where(done)[0]
                ep_ret += list(env.acc[idx]); ep_fail += list(~feas[idx])
                env.acc[idx] = 0
                sampler.reset_rows(env, idx)
        v_last = critic.forward(env.actor_obs())[:, 0]
        adv = np.zeros((T, B)); g = np.zeros(B)
        for n in reversed(range(T)):
            nv = v_last if n == T - 1 else Vv[n + 1]
            delta = R[n] + (1 - D[n]) * nv - Vv[n]
            g = delta + lam * (1 - D[n]) * g
            adv[n] = g
        ret = adv + Vv
        O2 = O.reshape(-1, obs_dim); A2 = Aa.reshape(-1); LP2 = LP.reshape(-1)
        ADV = adv.reshape(-1); RET = ret.reshape(-1)
        ADV = (ADV - ADV.mean()) / (ADV.std() + 1e-8)
        n_all = len(A2); mb = n_all // nmb
        for ep in range(epochs):
            perm = rng.permutation(n_all)
            for j in range(nmb):
                ix = perm[j * mb:(j + 1) * mb]
                mu, hs = actor.forward(O2[ix], cache=True); mu = mu[:, 0]
                sig = np.exp(logstd[0])
                lp = -0.5 * ((A2[ix] - mu) / sig) ** 2 - logstd[0] - 0.5 * np.log(2 * np.pi)
                ratio = np.exp(lp - LP2[ix]); adv_b = ADV[ix]
                unclipped = (ratio * adv_b <= np.clip(ratio, 1 - clip, 1 + clip) * adv_b)
                dL_dlp = np.where(unclipped, -adv_b * ratio, 0.0) / len(ix)
                dmu = dL_dlp * (A2[ix] - mu) / sig ** 2
                dlogstd = (dL_dlp * (((A2[ix] - mu) / sig) ** 2 - 1.0)).sum() - ent
                grads = actor.backward(hs, dmu[:, None]) + [np.array([dlogstd])]
                opt_pi.step(grads)
                logstd[0] = np.clip(logstd[0], -2.5, 0.5)
                vpred, hv = critic.forward(O2[ix], cache=True)
                dv = (vpred[:, 0] - RET[ix]) / len(ix)
                opt_v.step(critic.backward(hv, dv[:, None]))
        if it % 10 == 0 or it == iters - 1:
            venv = venv0.subset(np.arange(128))
            vr, vf = run_actor_episodes(actor, c, venv)
            rec = dict(it=it, train_ret=float(np.mean(ep_ret)) if ep_ret else None,
                       train_fail=float(np.mean(ep_fail)) if ep_fail else None,
                       val_ret=float(vr.mean()), val_fail=float(vf.mean()),
                       logstd=float(logstd[0]), time=time.time() - t0)
            log.append(rec); print(json.dumps(rec), flush=True)
            if vr.mean() > best:
                best = vr.mean(); actor.save(os.path.join(outdir, 'actor.pkl'))
    json.dump(dict(log=log, best_val=best, rejects=sampler.n_reject, accepts=sampler.n_accept),
              open(os.path.join(outdir, 'ppo_log.json'), 'w'), indent=1)


if __name__ == '__main__':
    seed, iters, out = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
    train(seed, iters, out)
