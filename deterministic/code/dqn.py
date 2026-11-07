"""Stages 3-4 of Algorithm 1: masked macro-transitions and Double DQN with gamma = 1.

Target: Y = R if z = 1, else R + Q_target(y', argmax_{b in A0(y')} Q_online(y', b)).
No deployment margin enters the target. Exploration: epsilon-greedy over the admissible set.
Usage: python dqn.py SEED ACTOR_PATH EPISODES OUTDIR
"""
import sys, os, json, time
import numpy as np
from config import Cfg
from env import Env
from scenarios import Sampler
from nn import MLP, Adam
from hl import run, STAY, CHANGE


class Replay:
    def __init__(self, cap, ydim):
        self.y = np.zeros((cap, ydim)); self.a = np.zeros(cap, int); self.R = np.zeros(cap)
        self.y2 = np.zeros((cap, ydim)); self.z = np.zeros(cap, bool); self.m2 = np.zeros((cap, 2), bool)
        self.cap, self.n, self.i = cap, 0, 0

    def add(self, y, a, R, y2, z, m2):
        k = len(a)
        idx = (self.i + np.arange(k)) % self.cap
        self.y[idx] = y; self.a[idx] = a; self.R[idx] = R; self.y2[idx] = y2; self.z[idx] = z; self.m2[idx] = m2
        self.i = (self.i + k) % self.cap; self.n = min(self.n + k, self.cap)

    def sample(self, rng, bs):
        ix = rng.integers(0, self.n, bs)
        return self.y[ix], self.a[ix], self.R[ix], self.y2[ix], self.z[ix], self.m2[ix]


def huber_grad(err, d=1.0):
    return np.clip(err, -d, d)


def train(seed, actor_path, episodes, outdir, B=32, lr=3e-4, bs=256, target_every=1000,
          grads_per_trans=0.125, eps_frac=0.5, eps_end=0.05):
    c = Cfg()
    from dataclasses import replace as _rep
    # training-time switching cost (baseline 'larger c0'); validation/selection always use the true c0
    c_tr = _rep(c, c0=float(os.environ.get('C0_TRAIN', c.c0)))
    cf_h = os.environ.get('CF_H')       # CF horizon variant: None = option window (1 s); else seconds
    os.makedirs(outdir, exist_ok=True)
    rng = np.random.default_rng(seed + 7)
    actor = MLP.load(actor_path)
    env = Env(c, B, rng)
    sampler = Sampler(c, seed + 2000)
    ydim = env.critic_obs().shape[1]
    q = MLP([ydim, 128, 128, 2], rng, out_scale=0.1)
    qt = MLP([ydim, 128, 128, 2], rng); qt.copy_from(q)
    opt = Adam(q.params(), lr=lr, clip=10.0)
    buf = Replay(300_000, ydim)
    # validation set (fixed)
    vs = Sampler(c, 123_456); venv0 = Env(c, 96); vs.reset_rows(venv0, range(96))
    n_grad = 0; ep_done = 0; log = []; t0 = time.time(); best = -np.inf
    rounds = int(np.ceil(episodes / B))
    for rd in range(rounds):
        eps = max(eps_end, 1.0 - (1.0 - eps_end) * (ep_done / (eps_frac * episodes)))
        sampler.reset_rows(env, range(B))
        def decide(env_, rows, ms, mc):
            y = env_.critic_obs()[rows]
            qq = q.forward(y)
            greedy = np.where(qq[:, 1] > qq[:, 0], CHANGE, STAY)
            rnd = rng.integers(0, 2, len(rows))
            return np.where(rng.random(len(rows)) < eps, rnd, greedy)
        coll = []
        res = run(env, actor, decide, c_tr, collect=coll, cf_w=float(os.environ.get('CF_W', '0')),
                  cf_h=None if cf_h is None else float(cf_h), cf_q=q)
        for (y, a, R, y2, z, m2) in coll:
            buf.add(y, a, R * c.rew_scale, y2, z, m2)
        n_new = sum(len(t[1]) for t in coll)
        ep_done += B
        if buf.n >= 5000:
            for _ in range(int(n_new * grads_per_trans)):
                y, a, R, y2, z, m2 = buf.sample(rng, bs)
                q2o = q.forward(y2); q2t = qt.forward(y2)
                q2o = np.where(m2, q2o, -np.inf)
                astar = np.argmax(q2o, 1)
                anyok = m2.any(1)
                boot = np.where(~z & anyok, q2t[np.arange(bs), astar], 0.0)
                Y = R + boot
                qv, hs = q.forward(y, cache=True)
                err = qv[np.arange(bs), a] - Y
                dout = np.zeros_like(qv); dout[np.arange(bs), a] = huber_grad(err) / bs
                opt.step(q.backward(hs, dout))
                n_grad += 1
                if n_grad % target_every == 0:
                    qt.copy_from(q)
        if rd % 10 == 0 or rd == rounds - 1:
            venv = venv0.subset(np.arange(96)); venv.done[:] = False
            def greedy_rule(env_, rows, ms, mc):
                qq = q.forward(env_.critic_obs()[rows])
                return np.where(qq[:, 1] > qq[:, 0], CHANGE, STAY)
            vres = run(venv, actor, greedy_rule, c)
            rec = dict(round=rd, episodes=ep_done, eps=eps, n_grad=n_grad, buf=buf.n,
                       train_G=float(res['G'].mean()), train_fail=float(res['fail'].mean()),
                       val_G=float(vres['G'].mean()), val_fail=float(vres['fail'].mean()),
                       val_nsw=float(venv.n_sw.mean()), time=time.time() - t0)
            log.append(rec); print(json.dumps(rec), flush=True)
            if rd > rounds * eps_frac and vres['G'].mean() > best:
                best = vres['G'].mean(); q.save(os.path.join(outdir, 'q.pkl'))
    q.save(os.path.join(outdir, 'q_last.pkl'))
    if not os.path.exists(os.path.join(outdir, 'q.pkl')):
        q.save(os.path.join(outdir, 'q.pkl'))
    json.dump(dict(log=log, best_val=best, rejects=sampler.n_reject, accepts=sampler.n_accept),
              open(os.path.join(outdir, 'dqn_log.json'), 'w'), indent=1)


if __name__ == '__main__':
    train(int(sys.argv[1]), sys.argv[2], int(sys.argv[3]), sys.argv[4])
