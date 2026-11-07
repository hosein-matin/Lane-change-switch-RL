"""Double DQN for L = 3 lanes (actions stay / left / right), same settings as dqn.py.
Usage: N_LANES=3 [C0_TRAIN=..] python dqn3.py SEED ACTOR_PATH EPISODES OUTDIR
"""
import sys, os, json, time
from dataclasses import replace as _rep
import numpy as np
from config import Cfg
from env import Env
from scenarios3 import Sampler3
from nn import MLP, Adam
from hl3 import run3
from dqn import Replay, huber_grad


def train(seed, actor_path, episodes, outdir, B=32, lr=3e-4, bs=256, target_every=1000,
          grads_per_trans=0.125, eps_frac=0.5, eps_end=0.05):
    c = Cfg(); assert c.n_lanes == 3
    c_tr = _rep(c, c0=float(os.environ.get('C0_TRAIN', c.c0)))
    os.makedirs(outdir, exist_ok=True)
    rng = np.random.default_rng(seed + 7)
    actor = MLP.load(actor_path)
    env = Env(c, B, rng); sampler = Sampler3(c, seed + 2000)
    ydim = env.critic_obs().shape[1]; A = 3
    q = MLP([ydim, 128, 128, A], rng, out_scale=0.1); qt = MLP([ydim, 128, 128, A], rng); qt.copy_from(q)
    opt = Adam(q.params(), lr=lr, clip=10.0)
    buf = Replay(300_000, ydim); buf.m2 = np.zeros((300_000, A), bool)
    vs = Sampler3(c, 123_456); venv0 = Env(c, 96); vs.reset_rows(venv0, range(96))
    n_grad = 0; ep_done = 0; log = []; t0 = time.time(); best = -np.inf
    rounds = int(np.ceil(episodes / B))
    for rd in range(rounds):
        eps = max(eps_end, 1.0 - (1.0 - eps_end) * (ep_done / (eps_frac * episodes)))
        sampler.reset_rows(env, range(B))
        def decide(env_, rows, M):
            qq = np.where(M, q.forward(env_.critic_obs()[rows]), -np.inf)
            greedy = np.argmax(qq, 1)
            rnd = np.array([rng.choice(np.where(m)[0]) if m.any() else 0 for m in M])
            return np.where(rng.random(len(rows)) < eps, rnd, greedy)
        coll = []
        res = run3(env, actor, decide, c_tr, collect=coll)
        for (y, a, R, y2, z, m2) in coll:
            buf.add(y, a, R * c.rew_scale, y2, z, m2)
        n_new = sum(len(t[1]) for t in coll); ep_done += B
        if buf.n >= 5000:
            for _ in range(int(n_new * grads_per_trans)):
                y, a, R, y2, z, m2 = buf.sample(rng, bs)
                q2o = np.where(m2, q.forward(y2), -np.inf); q2t = qt.forward(y2)
                astar = np.argmax(q2o, 1); anyok = m2.any(1)
                boot = np.where(~z & anyok, q2t[np.arange(bs), astar], 0.0)
                Y = R + boot
                qv, hs = q.forward(y, cache=True)
                err = qv[np.arange(bs), a] - Y
                dout = np.zeros_like(qv); dout[np.arange(bs), a] = huber_grad(err) / bs
                opt.step(q.backward(hs, dout)); n_grad += 1
                if n_grad % target_every == 0:
                    qt.copy_from(q)
        if rd % 10 == 0 or rd == rounds - 1:
            venv = venv0.subset(np.arange(96)); venv.done[:] = False
            def greedy_rule(env_, rows, M):
                qq = np.where(M, q.forward(env_.critic_obs()[rows]), -np.inf)
                b_ = np.argmax(qq, 1)
                return np.where(qq[np.arange(len(rows)), 0] >= qq[np.arange(len(rows)), b_], 0, b_)
            vres = run3(venv, actor, greedy_rule, c)
            rec = dict(round=rd, episodes=ep_done, eps=eps, n_grad=n_grad, buf=buf.n, train_G=float(res['G'].mean()),
                       val_G=float(vres['G'].mean()), val_fail=float(vres['fail'].mean()), val_nsw=float(venv.n_sw.mean()),
                       time=time.time() - t0)
            log.append(rec); print(json.dumps(rec), flush=True)
            if rd > rounds * eps_frac and vres['G'].mean() > best:
                best = vres['G'].mean(); q.save(os.path.join(outdir, 'q.pkl'))
    q.save(os.path.join(outdir, 'q_last.pkl'))
    if not os.path.exists(os.path.join(outdir, 'q.pkl')):
        q.save(os.path.join(outdir, 'q.pkl'))
    json.dump(dict(log=log, best_val=best), open(os.path.join(outdir, 'dqn_log.json'), 'w'), indent=1)


if __name__ == '__main__':
    train(int(sys.argv[1]), sys.argv[2], int(sys.argv[3]), sys.argv[4])
