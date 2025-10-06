import networkx as nx

def get_sir_gt_dag():
    """
    Returns the ground truth DAG for the SIR model.
    Represented as a directed acyclic graph (DiGraph) using networkx.
    
    Nodes:
        - S_t, I_t, R_t : state variables at time t
        - S_t+1, I_t+1, R_t+1 : state variables at time t+1
    Edges:
        Derived directly from the SIR equations.
    """
    G = nx.DiGraph()

    # Define nodes
    nodes = ["S_t", "I_t", "R_t", "S_t+1", "I_t+1", "R_t+1"]
    G.add_nodes_from(nodes)

    # Add edges based on causal dependencies
    edges = [
        ("S_t", "S_t+1"),
        ("I_t", "S_t+1"),
        ("S_t", "I_t+1"),
        ("I_t", "I_t+1"),
        ("I_t", "R_t+1"),
        ("R_t", "R_t+1"),
    ]
    G.add_edges_from(edges)

    return G


if __name__ == "__main__":
    dag = get_sir_gt_dag()
    print("Ground truth SIR DAG edges:")
    print(list(dag.edges()))
