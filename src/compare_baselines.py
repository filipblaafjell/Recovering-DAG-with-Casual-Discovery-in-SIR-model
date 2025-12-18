# run_pc_ges_structural.py
# =====================================================
# IMPORTS
# =====================================================
import os
import warnings
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning)

# Force CDT to use correct R installation
os.environ["R_HOME"] = r"C:\Program Files\R\R-4.5.2"
os.environ["R_USER"] = r"C:\Users\filip"
os.environ["PATH"] = r"C:\Program Files\R\R-4.5.2\bin;" + os.environ["PATH"]

import time
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from tqdm import tqdm

from causallearn.search.ConstraintBased.PC import pc
from causallearn.search.ScoreBased.GES import ges

from generate_data import generate_scm_data
from evaluation.metrics import compute_all_metrics


# =====================================================
# CPDAG → DAG (remove undirected edges)
# =====================================================
def cpdag_to_dag(adj):
    adj = adj.copy()
    adj[adj < 0] = 0   # remove undirected edges
    return adj.astype(int)


# =====================================================
# PLOT (raw CPDAGs for visual inspection)
# =====================================================
def plot_side_by_side(true_adj, adj_pc, adj_ges, filename):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    titles = ["True DAG", "PC", "GES"]
    mats = [true_adj, adj_pc, adj_ges]

    for ax, M, title in zip(axes, mats, titles):
        G = nx.from_numpy_array(M, create_using=nx.DiGraph)
        nx.draw(G, ax=ax, with_labels=True, node_color="lightblue")
        ax.set_title(title)

    plt.tight_layout()
    plt.savefig(filename)
    plt.close()


# =====================================================
# ALGORITHM WRAPPERS
# =====================================================
def run_pc(X):
    start = time.time()
    cg = pc(
        X,
        alpha=0.05,
        indep_test="fisherz",
        verbose=False,
        show_progress=False
    )
    runtime = time.time() - start
    adj = cg.G.graph.astype(int)   # CPDAG
    return adj, runtime


def run_ges(X):
    start = time.time()
    cg = ges(X)
    runtime = time.time() - start
    adj = cg["G"].graph.astype(int)   # CPDAG
    return adj, runtime


ALGORITHMS = {
    "PC": run_pc,
    "GES": run_ges
}


# =====================================================
# EXPERIMENT CONFIG
# =====================================================
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

    base_dir = f"experiments/nonlinear_{GRAPH_TYPE}_baseline"
    save_dir = f"{base_dir}/saved_runs"
    os.makedirs(base_dir, exist_ok=True)
    os.makedirs(save_dir, exist_ok=True)

    results_file = f"{base_dir}/results.txt"
    with open(results_file, "w") as f:
        f.write("GRAPH_TYPE = " + GRAPH_TYPE + "\n")
        f.write("Nodes | Samples | Algo | AP | AR | A-F1 | HP | HR | H-F1 | SHD | SID | Time\n")

    # =================================================
    # MONTE CARLO
    # =================================================
    for p in node_sizes:
        for n in sample_sizes:

            results = {
                algo: {m: [] for m in METRICS}
                for algo in ALGORITHMS
            }

            for seed in tqdm(
                range(num_trials),
                desc=f"{GRAPH_TYPE} {p} nodes, {n} samples"
            ):

                # Generate data
                data_envs, targets, true_adj = generate_scm_data(
                    num_nodes=p,
                    total_samples=n,
                    num_interventions=0,
                    intervention_size=0,
                    seed=seed,
                    graph_type=GRAPH_TYPE
                )

                X = data_envs[0]

                # Run algorithms
                for algo_name, algo_fn in ALGORITHMS.items():
                    adj_est_cpdag, runtime = algo_fn(X)

                    # Convert CPDAG → DAG for metrics
                    adj_est_dag = cpdag_to_dag(adj_est_cpdag)

                    # Save artefacts (raw CPDAG + true DAG + data)
                    run_id = f"{GRAPH_TYPE}_p{p}_n{n}_seed{seed}_{algo_name}"
                    np.save(f"{save_dir}/{run_id}_adj_est.npy", adj_est_cpdag)
                    np.save(f"{save_dir}/{run_id}_true_adj.npy", true_adj)
                    np.save(f"{save_dir}/{run_id}_X_obs.npy", X)

                    # Metrics (DAG only)
                    mt = compute_all_metrics(true_adj, adj_est_dag)

                    results[algo_name]["adj_precision"].append(mt["adjacency_precision"])
                    results[algo_name]["adj_recall"].append(mt["adjacency_recall"])
                    results[algo_name]["adj_f1"].append(mt["adjacency_f1"])
                    results[algo_name]["arrow_precision"].append(mt["arrowhead_precision"])
                    results[algo_name]["arrow_recall"].append(mt["arrowhead_recall"])
                    results[algo_name]["arrow_f1"].append(mt["arrowhead_f1"])
                    results[algo_name]["shd"].append(mt["shd"])
                    results[algo_name]["sid"].append(mt["sid"])
                    results[algo_name]["runtime"].append(runtime)

            # Plot last trial (raw CPDAGs)
            adj_pc_last, _ = run_pc(X)
            adj_ges_last, _ = run_ges(X)
            plot_side_by_side(
                true_adj,
                adj_pc_last,
                adj_ges_last,
                f"{base_dir}/dag_{GRAPH_TYPE}_p{p}_n{n}.png"
            )

            # Write summaries
            with open(results_file, "a") as f:
                for algo_name in ALGORITHMS:
                    line = f"{p} | {n} | {algo_name} | "
                    for m in METRICS:
                        mean = np.mean(results[algo_name][m])
                        std = np.std(results[algo_name][m])
                        line += f"{mean:.3f}±{std:.3f} | "
                    line += "\n"
                    f.write(line)
