"""K-means that can be reproduced exactly in another language.

sklearn's KMeans is deterministic for a fixed random_state, but its random
draws come from numpy's generator inside sklearn's C code, which cannot be
reproduced in MQL5. This module is the same algorithm (k-means++ seeding,
Lloyd's iterations, several restarts, lowest inertia wins) on top of a
generator small enough to write in a dozen lines anywhere: xorshift64*.

Every choice that affects the result is pinned down here so a port can
match it to floating-point rounding:

  generator     xorshift64*, state seeded as below, never zero
  uniform       (next() >> 11) * 2**-53                     in [0, 1)
  k-means++     first centre: index floor(u * n)
                each next: D2(x) = min squared distance to chosen centres;
                draw u; pick the first index where cumsum(D2) > u * sum(D2)
                (plain D2 sampling, not sklearn's greedy local trials)
  Lloyd         assign to nearest centre, ties to the lowest index;
                centre = mean of members, an empty cluster keeps its centre;
                stop when no label changes or after max_iter
  restarts      n_init runs on the SAME generator stream, in order;
                keep the lowest inertia, strict less-than
  inertia       sum of squared distances to the assigned centre

Reference implementation for mql5/CTAEngine.mqh.
"""
import numpy as np

_MASK = (1 << 64) - 1
_MULT = 0x2545F4914F6CDD1D
_GOLD = 0x9E3779B97F4A7C15


class XorShift64Star:
    def __init__(self, seed: int = 0):
        self.state = ((int(seed) * _GOLD) + _MULT) & _MASK
        if self.state == 0:
            self.state = _MULT

    def next_u64(self) -> int:
        x = self.state
        x ^= x >> 12
        x ^= (x << 25) & _MASK
        x ^= x >> 27
        self.state = x
        return (x * _MULT) & _MASK

    def uniform(self) -> float:
        return (self.next_u64() >> 11) * (2.0 ** -53)


def _sq_dist(Z: np.ndarray, c: np.ndarray) -> np.ndarray:
    d = Z - c
    return np.einsum("ij,ij->i", d, d)


def _kpp_init(Z: np.ndarray, k: int, rng: XorShift64Star) -> np.ndarray:
    n = Z.shape[0]
    centres = np.empty((k, Z.shape[1]))
    centres[0] = Z[int(rng.uniform() * n)]
    d2 = _sq_dist(Z, centres[0])
    for j in range(1, k):
        total = float(d2.sum())
        u = rng.uniform()
        if total <= 0.0:
            idx = 0
        else:
            cs = np.cumsum(d2)
            hit = np.nonzero(cs > u * total)[0]
            idx = int(hit[0]) if len(hit) else n - 1
        centres[j] = Z[idx]
        d2 = np.minimum(d2, _sq_dist(Z, centres[j]))
    return centres


def _lloyd(Z: np.ndarray, centres: np.ndarray, max_iter: int) -> tuple:
    k = centres.shape[0]
    labels = np.full(Z.shape[0], -1)
    for _ in range(max_iter):
        D = np.stack([_sq_dist(Z, centres[j]) for j in range(k)], axis=1)
        new = D.argmin(axis=1)                      # argmin takes the lowest index on ties
        if np.array_equal(new, labels):
            break
        labels = new
        for j in range(k):
            m = Z[labels == j]
            if len(m):
                centres[j] = m.mean(axis=0)
    D = np.stack([_sq_dist(Z, centres[j]) for j in range(k)], axis=1)
    labels = D.argmin(axis=1)
    inertia = float(D[np.arange(len(Z)), labels].sum())
    return centres, labels, inertia


class PortableKMeans:
    """Same call surface as the parts of sklearn.cluster.KMeans the engine uses."""

    def __init__(self, n_clusters: int, n_init: int = 10, random_state: int = 0, max_iter: int = 300):
        self.n_clusters, self.n_init, self.random_state, self.max_iter = n_clusters, n_init, random_state, max_iter
        self.cluster_centers_ = None
        self.inertia_ = None
        self.labels_ = None

    def fit(self, Z):
        Z = np.ascontiguousarray(Z, dtype="float64")
        rng = XorShift64Star(self.random_state)
        best = None
        for _ in range(self.n_init):
            c0 = _kpp_init(Z, self.n_clusters, rng)
            c, lab, inertia = _lloyd(Z, c0.copy(), self.max_iter)
            if best is None or inertia < best[2]:
                best = (c, lab, inertia)
        self.cluster_centers_, self.labels_, self.inertia_ = best
        return self

    def predict(self, X):
        X = np.ascontiguousarray(X, dtype="float64")
        D = np.stack([_sq_dist(X, c) for c in self.cluster_centers_], axis=1)
        return D.argmin(axis=1)
