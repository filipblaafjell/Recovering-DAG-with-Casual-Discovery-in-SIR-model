import matplotlib.pyplot as plt
import networkx as nx
from networkx.drawing.nx_pydot import read_dot
import io

from src.generate_data import simulate_sir
from src.discovery import run_pc
from src.evaluate import structural_hamming_distance


def main():
    # Step 1: Simulate data
    print("Simulating SIR data...")
    df = simulate_sir(beta=0.3, gamma=0.1, T=50, dt=0.1)
    print(df.head())

    # Step 2: Run causal discovery (PC)
    print("\nRunning PC algorithm...")
    graph_str = run_pc(df, alpha=0.05)

    # Step 3: Visualize learned graph
    dot_bytes = graph_str.encode("utf-8")
    G = nx.drawing.nx_pydot.read_dot(io.BytesIO(dot_bytes))

    nx.draw(G, with_labels=True, node_size=1500, node_color="lightblue", arrows=True)
    plt.title("Discovered DAG (PC algorithm)")
    plt.show()

    # Step 4: Compare to ground truth DAG
    true_edges = {("S", "I"), ("I", "R")}
    learned_edges = set(G.edges())

    print("\nTrue edges:", true_edges)
    print("Learned edges:", learned_edges)
    print("Structural Hamming Distance (SHD):", 
          structural_hamming_distance(true_edges, learned_edges))


if __name__ == "__main__":
    main()
