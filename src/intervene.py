"""
intervene.py
------------
Simulates interventions (do-operations) on the true SEM and compares results
to those implied by a learned DAG structure.

Usage:
    python src/intervene.py
"""

import numpy as np
import pandas as pd
import os


def load_graph(path):
    """Load adjacency matrix from CSV."""
    return pd.read_csv(path, header=None).values


def load_data(path):
    """Load data samples from CSV."""
    return pd.read_csv(path).values


def intervene(data, dag, node_idx, new_value):
    """
    Perform do(X_i = new_value) intervention:
    - Remove all incoming edges to node i
    - Replace that node’s values with new_value
    - Propagate effects forward using DAG order
    """
    data_intervened = data.copy()
    data_intervened[:, node_idx] = new_value

    # propagate forward (topological order approximation)
    num_vars = dag.shape[0]
    for j in range(num_vars):
        parents = np.where(dag[:, j] == 1)[0]
        if len(parents) > 0 and j != node_idx:
            data_intervened[:, j] = np.tanh(np.sum(data_intervened[:, parents], axis=1)) + np.random.normal(0, 0.1, size=len(data))
    return data_intervened


def compare_effects(original, intervened, node_idx):
    """Compare pre- and post-intervention means for each variable."""
    delta = intervened.mean(axis=0) - original.mean(axis=0)
    df = pd.DataFrame({
        "Variable": np.arange(len(delta)),
        "MeanChange": delta,
    })
    df["Intervened"] = df["Variable"] == node_idx
    return df


def main():
    base_dir = os.path.join(os.path.dirname(__file__), "..")
    data_path = os.path.join(base_dir, "data", "raw", "data_20vars.csv")
    true_path = os.path.join(base_dir, "data", "raw", "dag_20vars.csv")
    est_path = os.path.join(base_dir, "results", "graphs", "est_dag_grandag.csv")

    data = load_data(data_path)
    true_dag = load_graph(true_path)
    est_dag = load_graph(est_path)

    node_to_intervene = 3  # example: intervene on variable X3
    new_value = 2.0

    # intervention on true model
    data_intervened_true = intervene(data, true_dag, node_to_intervene, new_value)
    effects_true = compare_effects(data, data_intervened_true, node_to_intervene)

    # intervention on learned model
    data_intervened_est = intervene(data, est_dag, node_to_intervene, new_value)
    effects_est = compare_effects(data, data_intervened_est, node_to_intervene)

    print("\n--- True Model Intervention Effects ---")
    print(effects_true)

    print("\n--- Estimated Model Intervention Effects ---")
    print(effects_est)

    # optional: save results
    out_dir = os.path.join(base_dir, "results", "interventions")
    os.makedirs(out_dir, exist_ok=True)
    effects_true.to_csv(os.path.join(out_dir, "effects_true.csv"), index=False)
    effects_est.to_csv(os.path.join(out_dir, "effects_est.csv"), index=False)


if __name__ == "__main__":
    main()