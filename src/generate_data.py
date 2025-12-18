import numpy as np
import networkx as nx
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF


# =====================================================
# 1. GP FOR EACH EDGE (corrected GP prior sampling)
# =====================================================

def sample_gp_function(seed=None):
    rng = np.random.default_rng(seed)

    kernel = RBF(length_scale=1.0)

    # Sample from GP PRIOR on a dense grid
    X_grid = np.linspace(-4, 4, 200).reshape(-1, 1)
    mean = np.zeros(len(X_grid))
    cov = kernel(X_grid)

    f_values = rng.multivariate_normal(mean, cov)

    gp = GaussianProcessRegressor(
        kernel=kernel,
        alpha=1e-6,
        optimizer=None
    )
    gp.fit(X_grid, f_values)

    return gp


# =====================================================
# 2. GRAPH GENERATORS
# =====================================================

def generate_er_dag(num_nodes, seed=None):
    """ER graph with expected edges ≈ num_nodes (ER1)."""
    rng = np.random.default_rng(seed)
    ordering = rng.permutation(num_nodes)
    edge_prob = 2 / (num_nodes - 1)   # ER1

    G = nx.DiGraph()
    G.add_nodes_from(range(num_nodes))

    for a in range(num_nodes):
        for b in range(a + 1, num_nodes):
            i = ordering[a]
            j = ordering[b]
            if rng.random() < edge_prob:
                G.add_edge(i, j)

    true_adj = nx.to_numpy_array(G, dtype=int)
    return G, true_adj, ordering


def generate_sf_dag(num_nodes, seed=None):
    """Scale-free graph with expected edges ≈ num_nodes (SF1)."""
    rng = np.random.default_rng(seed)

    # Undirected scale-free BA model (≈ d edges)
    G_undirected = nx.barabasi_albert_graph(num_nodes, m=1, seed=seed)

    # Random ordering for acyclic orientation
    ordering = rng.permutation(num_nodes)
    order_idx = {node: idx for idx, node in enumerate(ordering)}

    G = nx.DiGraph()
    G.add_nodes_from(G_undirected.nodes())

    for u, v in G_undirected.edges():
        if order_idx[u] < order_idx[v]:
            G.add_edge(u, v)
        else:
            G.add_edge(v, u)

    true_adj = nx.to_numpy_array(G, dtype=int)
    return G, true_adj, ordering


# =====================================================
# 3. SAMPLE DATA
# =====================================================

def sample_data(
    n_samples,
    num_nodes,
    true_adj,
    gp_edge_funcs,
    ordering,
    noise_scales,
    rng,
    intervention_targets=None
):
    X = np.zeros((n_samples, num_nodes))

    for j in ordering:

        parents = np.where(true_adj[:, j] == 1)[0]

        # Perfect intervention override
        if intervention_targets and j in intervention_targets:
            X[:, j] = rng.normal(loc=2.0, scale=np.sqrt(0.2), size=n_samples)
            continue

        noise = rng.normal(0, noise_scales[j], size=n_samples)

        if len(parents) == 0:
            X[:, j] = noise
            continue

        total = np.zeros(n_samples)
        for i in parents:
            f = gp_edge_funcs[(i, j)]
            total += f.predict(X[:, i].reshape(-1, 1))

        X[:, j] = total + noise

    return X


# =====================================================
# 4. MAIN GENERATOR
# =====================================================

def generate_scm_data(
    num_nodes=10,
    total_samples=1000,
    num_interventions=1,
    intervention_size=1,
    seed=None,
    graph_type="ER"    # NEW: "ER" or "SF"
):

    rng = np.random.default_rng(seed)

    # -------------------------------------------------
    # DAG type selection
    # -------------------------------------------------
    if graph_type == "ER":
        G, true_adj, ordering = generate_er_dag(num_nodes, seed)
    elif graph_type == "SF":
        G, true_adj, ordering = generate_sf_dag(num_nodes, seed)
    else:
        raise ValueError("graph_type must be 'ER' or 'SF'.")

    # Noise per node
    noise_scales = rng.uniform(0.1, 0.5, size=num_nodes)

    # GP edge functions
    gp_edge_funcs = {}
    for i in range(num_nodes):
        for j in range(num_nodes):
            if true_adj[i, j] == 1:
                gp_edge_funcs[(i, j)] = sample_gp_function(seed + i * 31 + j if seed else None)

    # -------------------------------------------------
    # SPLIT SAMPLES ACROSS ENVIRONMENTS
    # -------------------------------------------------
    num_envs = 1 + num_interventions
    samples_per_env = total_samples // num_envs

    data_envs = []
    intervention_targets = []

    # Observational
    data_obs = sample_data(
        samples_per_env, num_nodes, true_adj,
        gp_edge_funcs, ordering, noise_scales, rng,
        intervention_targets=None
    )
    data_envs.append(data_obs)
    intervention_targets.append([])

    # Interventional
    for k in range(num_interventions):
        targets = rng.choice(num_nodes, size=intervention_size, replace=False)

        data_int = sample_data(
            samples_per_env, num_nodes, true_adj,
            gp_edge_funcs, ordering, noise_scales, rng,
            intervention_targets=list(targets)
        )

        data_envs.append(data_int)
        intervention_targets.append(list(targets))

    return data_envs, intervention_targets, true_adj
