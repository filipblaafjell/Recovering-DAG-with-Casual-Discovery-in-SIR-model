"""
Generates data compatible with nonlinear SEMs of the form:
    X_j = f_j(Parents(X_j)) + N_j,
where each f_j is a smooth nonlinear function
"""

import os
import numpy as np
import pandas as pd
import networkx as nx


def generate_random_dag(n_nodes: int, edge_prob: float, seed: int = 0) -> nx.DiGraph:
    """Generate a random DAG by sampling upper-triangular edges."""
    rng = np.random.default_rng(seed)
    G = nx.DiGraph()
    G.add_nodes_from(range(n_nodes))
    for i in range(n_nodes):
        for j in range(i + 1, n_nodes):
            if rng.random() < edge_prob:
                G.add_edge(i, j)
    return G


def simulate_sem(G: nx.DiGraph, n_samples: int = 5000,
                 noise_scale: float = 0.2, seed: int = 0) -> pd.DataFrame:
    """Simulate smooth nonlinear additive SEMs using tanh transformations."""
    rng = np.random.default_rng(seed)
    n_nodes = len(G.nodes)
    X = np.zeros((n_samples, n_nodes))
    topo_order = list(nx.topological_sort(G))

    # Randomized nonlinearities per node
    hidden_dim = 5
    hidden_bias = {node: rng.normal(0, 0.3, size=(hidden_dim,)) for node in G.nodes}
    output_weights = {node: rng.normal(0, 0.5, size=(hidden_dim,)) for node in G.nodes}
    output_bias = {node: rng.normal(0, 0.3) for node in G.nodes}

    for node in topo_order:
        parents = list(G.predecessors(node))
        if parents:
            parent_vals = X[:, parents]

            # Random linear mixing of parent inputs
            W_parents = rng.normal(0.3, 0.2, size=(len(parents), hidden_dim))
            hidden_input = np.tanh(parent_vals @ W_parents + hidden_bias[node])
            f_val = hidden_input @ output_weights[node] + output_bias[node]

            X[:, node] = f_val + noise_scale * rng.normal(size=n_samples)
        else:
            X[:, node] = rng.normal(size=n_samples)

    # Standardize each variable (zero mean, unit variance)
    X = (X - X.mean(axis=0)) / X.std(axis=0)
    return pd.DataFrame(X, columns=[f"X{i}" for i in range(n_nodes)])


def save_outputs(X: pd.DataFrame, G: nx.DiGraph, out_dir: str, n_nodes: int):
    os.makedirs(out_dir, exist_ok=True)
    X.to_csv(os.path.join(out_dir, f"data_{n_nodes}vars.csv"), index=False)
    dag = nx.to_numpy_array(G, dtype=int)
    pd.DataFrame(dag).to_csv(os.path.join(out_dir, f"dag_{n_nodes}vars.csv"), index=False)
    print(f"Saved data and DAG to {out_dir}")


def main(n_nodes: int = 20, n_samples: int = 5000, edge_prob: float = 0.2):
    base_dir = os.path.dirname(__file__)
    out_dir = os.path.join(base_dir, "..", "data", "raw")
    G = generate_random_dag(n_nodes, edge_prob)
    X = simulate_sem(G, n_samples)
    save_outputs(X, G, out_dir, n_nodes)


if __name__ == "__main__":
    main()
