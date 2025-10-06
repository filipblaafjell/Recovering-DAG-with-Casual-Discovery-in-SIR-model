import numpy as np
import pandas as pd
import os
from typing import Tuple


def load_graphs(true_path: str, est_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """Load true and estimated adjacency matrices from CSV files."""
    true_dag = pd.read_csv(true_path, header=None).values
    est_dag = pd.read_csv(est_path, header=None).values
    return true_dag, est_dag


def structural_hamming_distance(true_dag: np.ndarray, est_dag: np.ndarray) -> int:
    """
    Compute Structural Hamming Distance (SHD):
    Counts the minimum number of edge additions, deletions, or reversals
    needed to transform est_dag into true_dag.
    """
    diff = np.abs(true_dag - est_dag)
    undirected_mismatch = np.logical_and(true_dag + true_dag.T == 1,
                                         est_dag + est_dag.T == 1)
    shd = np.sum(diff) / 2 - np.sum(undirected_mismatch)
    return int(shd)


def orientation_accuracy(true_dag: np.ndarray, est_dag: np.ndarray) -> float:
    """Compute the fraction of correctly oriented edges."""
    true_edges = set(map(tuple, np.argwhere(true_dag == 1)))
    est_edges = set(map(tuple, np.argwhere(est_dag == 1)))
    correct = len(true_edges & est_edges)
    return correct / len(true_edges) if len(true_edges) > 0 else 0.0


def evaluate(true_path: str, est_path: str) -> None:
    true_dag, est_dag = load_graphs(true_path, est_path)
    shd = structural_hamming_distance(true_dag, est_dag)
    acc = orientation_accuracy(true_dag, est_dag)
    print(f"Structural Hamming Distance: {shd}")
    print(f"Orientation Accuracy: {acc:.3f}")


if __name__ == "__main__":
    base_dir = os.path.join(os.path.dirname(__file__), "..")
    true_path = os.path.join(base_dir, "data", "raw", "dag_20vars.csv")
    est_path = os.path.join(base_dir, "results", "graphs", "est_dag_notears_mlp.csv")
    evaluate(true_path, est_path)
