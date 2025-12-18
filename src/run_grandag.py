# =====================================================
# IMPORTS
# =====================================================
import os

# Force CDT to use correct R installation
os.environ["R_HOME"] = r"C:\Program Files\R\R-4.5.2"
os.environ["R_USER"] = r"C:\Users\filip"
os.environ["PATH"] = r"C:\Program Files\R\R-4.5.2\bin;" + os.environ["PATH"]

import time
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from tqdm import tqdm

import torch
from castle.algorithms.gradient.gran_dag.torch import GraNDAG

from generate_data import generate_scm_data
from evaluation.metrics import compute_all_metrics


# =====================================================
# PLOT
# =====================================================
def plot_side_by_side(true_adj, adj_est, filename):
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    titles = ["True DAG", "GraNDAG"]
    mats = [true_adj, adj_est]

    for ax, M, title in zip(axes, mats, titles):
        G = nx.from_numpy_array(M, create_using=nx.DiGraph)
        nx.draw(G, ax=ax, with_labels=True, node_color="lightblue")
        ax.set_title(title)

    plt.tight_layout()
    plt.savefig(filename)
    plt.close()


# =====================================================
# GRaNDAG WRAPPER
# =====================================================
def run_grandag(X):
    torch.set_default_dtype(torch.double)

    start = time.time()

    model = GraNDAG(
        input_dim=X.shape[1],
        batch_size=64,
        iterations=2000,
        lr=1e-3
    )

    model.learn(X)

    runtime = time.time() - start

    W = model.causal_matrix
    W_bin = (np.abs(W) > 0.3).astype(int)

    return W_bin, runtime


# =====================================================
# EXPERIMENT CONFIG
# =====================================================
ALGO_NAME = "GraNDAG"

METRICS = [
    "adj_precision", "adj_recall", "adj_f1",
    "arrow_precision", "arrow_recall", "arrow_f1",
    "shd", "sid", "runtime"
]

node_sizes = [5, 10, 15]
sample_sizes = [500, 1000, 2000]
num_trials = 20


# =====================================================
# MAIN LOOP OVER GRAPH TYPES (ER → SF)
# =====================================================
for GRAPH_TYPE in ["ER", "SF"]:

    base_dir = f"experiments/grandag_{GRAPH_TYPE}"
    save_dir = f"{base_dir}/saved_runs"
    os.makedirs(base_dir, exist_ok=True)
    os.makedirs(save_dir, exist_ok=True)

    # Results file identical to NOTEARS format
    results_file = f"{base_dir}/results.txt"
    with open(results_file, "w") as f:
        f.write("GRAPH_TYPE = " + GRAPH_TYPE + "\n")
        f.write("Nodes | Samples | Algo | AP | AR | A-F1 | HP | HR | H-F1 | SHD | SID | Time\n")

    # =====================================================
    # MONTE-CARLO EXPERIMENT
    # =====================================================
    for p in node_sizes:
        for n in sample_sizes:

            results = {m: [] for m in METRICS}

            for seed in tqdm(range(num_trials),
                             desc=f"{GRAPH_TYPE} {p} nodes, {n} samples"):

                # Generate ER or SF data
                data_envs, targets, true_adj = generate_scm_data(
                    num_nodes=p,
                    total_samples=n,
                    num_interventions=0,
                    intervention_size=0,
                    seed=seed,
                    graph_type=GRAPH_TYPE
                )

                X = data_envs[0]

                # Run GraNDAG
                adj_est, runtime = run_grandag(X)

                # Save run
                run_id = f"{GRAPH_TYPE}_p{p}_n{n}_seed{seed}"
                np.save(f"{save_dir}/{run_id}_adj_est.npy", adj_est)
                np.save(f"{save_dir}/{run_id}_true_adj.npy", true_adj)
                np.save(f"{save_dir}/{run_id}_X_obs.npy", X)

                mt = compute_all_metrics(true_adj, adj_est)

                results["adj_precision"].append(mt["adjacency_precision"])
                results["adj_recall"].append(mt["adjacency_recall"])
                results["adj_f1"].append(mt["adjacency_f1"])
                results["arrow_precision"].append(mt["arrowhead_precision"])
                results["arrow_recall"].append(mt["arrowhead_recall"])
                results["arrow_f1"].append(mt["arrowhead_f1"])
                results["shd"].append(mt["shd"])
                results["sid"].append(mt["sid"])
                results["runtime"].append(runtime)

            # Plot last trial DAG
            adj_last, _ = run_grandag(X)
            plot_side_by_side(
                true_adj,
                adj_last,
                f"{base_dir}/dag_{GRAPH_TYPE}_p{p}_n{n}.png"
            )

            # Write summary
            with open(results_file, "a") as f:
                line = f"{p} | {n} | {ALGO_NAME} | "
                for m in METRICS:
                    mean = np.mean(results[m])
                    std = np.std(results[m])
                    line += f"{mean:.3f}±{std:.3f} | "
                line += "\n"
                f.write(line)
