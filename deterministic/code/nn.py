"""Minimal NumPy MLP (tanh hidden layers, linear output) with Adam. Float64 for reproducibility."""
import numpy as np
import pickle


class MLP:
    def __init__(self, sizes, rng, out_scale=1.0):
        self.W, self.b = [], []
        for i, (a, o) in enumerate(zip(sizes[:-1], sizes[1:])):
            last = i == len(sizes) - 2
            self.W.append(rng.normal(0, np.sqrt(1.0 / a), (a, o)) * (out_scale if last else 1.0))
            self.b.append(np.zeros(o))

    def params(self):
        return self.W + self.b

    def forward(self, x, cache=False):
        hs = [x]
        h = x
        n = len(self.W)
        for i in range(n):
            z = h @ self.W[i] + self.b[i]
            h = np.tanh(z) if i < n - 1 else z
            hs.append(h)
        return (h, hs) if cache else h

    def backward(self, hs, dout):
        n = len(self.W)
        gW = [None] * n; gb = [None] * n
        g = dout
        for i in reversed(range(n)):
            gW[i] = hs[i].T @ g
            gb[i] = g.sum(0)
            if i > 0:
                g = (g @ self.W[i].T) * (1.0 - hs[i] ** 2)
        return gW + gb

    def copy_from(self, other):
        self.W = [w.copy() for w in other.W]; self.b = [b.copy() for b in other.b]

    def save(self, path):
        with open(path, 'wb') as f:
            pickle.dump({'W': self.W, 'b': self.b}, f)

    @staticmethod
    def load(path):
        with open(path, 'rb') as f:
            d = pickle.load(f)
        m = MLP.__new__(MLP); m.W, m.b = d['W'], d['b']
        return m


class Adam:
    def __init__(self, params, lr=3e-4, b1=0.9, b2=0.999, eps=1e-8, clip=0.5):
        self.p = params
        self.lr, self.b1, self.b2, self.eps, self.clip = lr, b1, b2, eps, clip
        self.m = [np.zeros_like(x) for x in params]; self.v = [np.zeros_like(x) for x in params]
        self.t = 0

    def step(self, grads):
        if self.clip is not None:
            gn = np.sqrt(sum((g ** 2).sum() for g in grads))
            if gn > self.clip:
                grads = [g * (self.clip / gn) for g in grads]
        self.t += 1
        for i, (p, g) in enumerate(zip(self.p, grads)):
            self.m[i] = self.b1 * self.m[i] + (1 - self.b1) * g
            self.v[i] = self.b2 * self.v[i] + (1 - self.b2) * g * g
            mh = self.m[i] / (1 - self.b1 ** self.t); vh = self.v[i] / (1 - self.b2 ** self.t)
            p -= self.lr * mh / (np.sqrt(vh) + self.eps)
