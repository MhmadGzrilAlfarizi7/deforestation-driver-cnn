"""Generator data sintetis patch Sentinel-1 + Sentinel-2 untuk 7 kelas driver deforestasi.

Channel: 0-3 = Sentinel-2 (B2, B3, B4, B8 / reflektansi), 4-5 = Sentinel-1 (VV, VH / dB).
Data ini SIMULASI untuk keperluan belajar. Untuk riset nyata, ganti dengan patch
Sentinel hasil labeling (lihat README bagian "Menggunakan data nyata").
"""
import numpy as np

CLASSES = ["smallholder_agri", "large_agri", "road", "selective_logging",
           "mining", "wildfire", "natural_disturbance"]
PRIORS = np.array([0.28, 0.34, 0.08, 0.10, 0.03, 0.10, 0.07])  # agriculture dominan

FOREST = np.array([0.04, 0.07, 0.04, 0.35, -7.0, -13.0], dtype=np.float32)
DELTA = np.array([
    [0.02, 0.03, 0.05, -0.15, -2.0, -2.5],   # smallholder
    [0.02, 0.04, 0.06, -0.18, -2.5, -3.0],   # large agri
    [0.04, 0.05, 0.08, -0.20, -1.5, -2.0],   # road
    [0.01, 0.02, 0.03, -0.08, -1.0, -1.5],   # logging
    [0.08, 0.09, 0.10, -0.20, -3.0, -3.0],   # mining
    [0.00, 0.00, 0.01, -0.20, -1.0, -2.0],   # wildfire (mirip natural -> sengaja membingungkan)
    [0.00, 0.01, 0.00, -0.12, -4.5, -4.0],   # natural / flooding
], dtype=np.float32)
NOISE = np.array([0.02, 0.02, 0.02, 0.05, 1.2, 1.2], dtype=np.float32)


def _grid(size):
    return np.mgrid[:size, :size]


def _blob(rng, size, r):
    yy, xx = _grid(size)
    cy, cx = size / 2 + rng.uniform(-2, 2, 2)
    return ((yy - cy) ** 2 + (xx - cx) ** 2 <= r ** 2).astype(np.float32)


def _line(rng, size, width):
    yy, xx = _grid(size)
    a, c = rng.uniform(0, np.pi), size / 2
    return (np.abs((xx - c) * np.sin(a) - (yy - c) * np.cos(a)) <= width).astype(np.float32)


def _mask(rng, cls, size):
    if cls == 0:
        return _blob(rng, size, rng.uniform(2, 3.2))
    if cls == 1:
        return _blob(rng, size, rng.uniform(5.5, 7.5))
    if cls == 2:
        return _line(rng, size, rng.uniform(0.7, 1.2))
    if cls == 3:
        m = _line(rng, size, rng.uniform(0.5, 0.9))
        for _ in range(rng.integers(2, 5)):
            m = np.maximum(m, _blob(rng, size, 1.2) * (rng.random((size, size)) > 0.5))
        return m
    if cls == 4:
        return _blob(rng, size, rng.uniform(3.5, 5))
    if cls == 5:
        return _blob(rng, size, rng.uniform(4.5, 7)) * (rng.random((size, size)) > 0.25)
    return _line(rng, size, rng.uniform(2.2, 3.5))


def make_dataset(n=6000, size=16, cloud_prob=0.0, seed=0):
    """Return X (n, 6, size, size) dan y (n,). cloud_prob = peluang S2 tertutup awan."""
    rng = np.random.default_rng(seed)
    y = rng.choice(len(CLASSES), size=n, p=PRIORS / PRIORS.sum())
    X = np.empty((n, 6, size, size), dtype=np.float32)
    for i, c in enumerate(y):
        m = _mask(rng, c, size)
        base = FOREST[:, None, None] + DELTA[c][:, None, None] * m
        X[i] = base + rng.normal(0, 1, (6, size, size)) * NOISE[:, None, None]
        if rng.random() < cloud_prob:  # awan: optik rusak, radar tetap
            X[i, :4] = 0.45 + rng.normal(0, 0.05, (4, size, size))
    return X, y.astype(np.int64)


def standardize(X, mean=None, std=None):
    if mean is None:
        mean, std = X.mean((0, 2, 3), keepdims=True), X.std((0, 2, 3), keepdims=True) + 1e-6
    return (X - mean) / std, mean, std
