# interventional_sem.py

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF

def fit_sem_gp(X, adj):
    p = X.shape[1]
    models = {}

    for j in range(p):
        parents = np.where(adj[:, j] == 1)[0]

        if len(parents) == 0:
            models[j] = None
            continue

        gp = GaussianProcessRegressor(kernel=RBF(length_scale=1.0), alpha=1e-6)
        gp.fit(X[:, parents], X[:, j])

        models[j] = (parents, gp)

    return models


def simulate_do(models, adj, k, value, n_samples=1000):
    p = adj.shape[0]
    X_do = np.zeros((n_samples, p))

    # topological order
    topo = list(np.argsort(np.sum(adj, axis=0)))

    for j in topo:
        if j == k:
            X_do[:, j] = value
            continue

        entry = models[j]
        if entry is None:
            X_do[:, j] = 0.0
            continue

        parents, gp = entry
        X_do[:, j] = gp.predict(X_do[:, parents])

    return X_do


def interventional_mse(X_est, X_true):
    return np.mean((X_est - X_true)**2)
