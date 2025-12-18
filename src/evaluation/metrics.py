import numpy as np

from causallearn.graph.GeneralGraph import GeneralGraph
from causallearn.graph.GraphNode import GraphNode
from causallearn.graph.ArrowConfusion import ArrowConfusion
from causallearn.graph.AdjacencyConfusion import AdjacencyConfusion
from causallearn.graph.SHD import SHD
from cdt.metrics import SID


def adj_to_graph(adj):
    """
    Convert a binary adjacency matrix (np.ndarray) to a causal-learn CPDAG Graph.
    1 in adj[i, j] means a directed edge i → j.
    """
    n = adj.shape[0]
    nodes = [GraphNode(str(i)) for i in range(n)]
    G = GeneralGraph(nodes)

    for i in range(n):
        for j in range(n):
            if adj[i, j] == 1:
                G.add_directed_edge(nodes[i], nodes[j])

    return G


def safe_precision(tp, fp):
    return tp / (tp + fp) if (tp + fp) > 0 else 0.0


def safe_recall(tp, fn):
    return tp / (tp + fn) if (tp + fn) > 0 else 0.0


def safe_f1(p, r):
    return (2 * p * r / (p + r)) if (p + r) > 0 else 0.0


def compute_all_metrics(true_adj, est_adj):
    """
    Evaluate using causal-learn’s built-in classes + CDT's SID.
    """

    # Convert adjacency matrices to causal-learn graph objects
    true_graph = adj_to_graph(true_adj)
    est_graph = adj_to_graph(est_adj)

    # =====================================================
    # Adjacency metrics (undirected)
    # =====================================================
    adj = AdjacencyConfusion(true_graph, est_graph)

    tp = adj.get_adj_tp()
    fp = adj.get_adj_fp()
    fn = adj.get_adj_fn()

    ap = safe_precision(tp, fp)
    ar = safe_recall(tp, fn)
    adj_f1 = safe_f1(ap, ar)

    # =====================================================
    # Arrowhead metrics (orientations)
    # =====================================================
    arrow = ArrowConfusion(true_graph, est_graph)

    tp_a = arrow.get_arrows_tp()
    fp_a = arrow.get_arrows_fp()
    fn_a = arrow.get_arrows_fn()

    ahp = safe_precision(tp_a, fp_a)
    ahr = safe_recall(tp_a, fn_a)
    arrow_f1 = safe_f1(ahp, ahr)

    # =====================================================
    # Structural Hamming Distance
    # =====================================================
    shd = SHD(true_graph, est_graph).get_shd()

    # =====================================================
    # Structural Intervention Distance (SID)
    # FIX: use adjacency matrices (NumPy), not GeneralGraph
    # =====================================================
    sid = SID(true_adj, est_adj)

    return {
        "adjacency_precision": ap,
        "adjacency_recall": ar,
        "adjacency_f1": adj_f1,
        "arrowhead_precision": ahp,
        "arrowhead_recall": ahr,
        "arrowhead_f1": arrow_f1,
        "shd": shd,
        "sid": sid,
    }


def print_metrics(m, name="Algorithm"):
    print(f"\n{name} metrics:")
    print(f"AP:        {m['adjacency_precision']:.3f}")
    print(f"AR:        {m['adjacency_recall']:.3f}")
    print(f"Adj-F1:    {m['adjacency_f1']:.3f}")
    print(f"AHP:       {m['arrowhead_precision']:.3f}")
    print(f"AHR:       {m['arrowhead_recall']:.3f}")
    print(f"Arrow-F1:  {m['arrowhead_f1']:.3f}")
    print(f"SHD:       {m['shd']}")
    print(f"SID:       {m['sid']}")
