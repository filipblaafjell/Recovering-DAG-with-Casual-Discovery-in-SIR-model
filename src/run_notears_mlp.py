import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from notears.nonlinear import NotearsMLP, notears_nonlinear
import torch
from tqdm import tqdm


def run_notears_mlp(data_path: str, out_dir: str):
    # Set default dtype for stability
    torch.set_default_dtype(torch.float32)

    # Load and normalize data
    data = pd.read_csv(data_path)
    data = pd.DataFrame(StandardScaler().fit_transform(data), columns=data.columns)
    X = data.values.astype(np.float32)

    print(f"\n=== Running NOTEARS-MLP ===")
    print(f"Dataset shape: {data.shape}")
    print(f"Data mean (first 5 cols): {data.mean().values[:5]}")
    print(f"Data std (first 5 cols): {data.std().values[:5]}")

    # Hyperparameters
    lambda1 = 0.005
    lambda2 = 0.005
    hidden_layers = [64, 32, 1]
    max_iter = 1000
    w_threshold = 0.05

    print("\n--- Hyperparameters ---")
    print(f"Lambda1: {lambda1}")
    print(f"Lambda2: {lambda2}")
    print(f"Hidden layers: {hidden_layers}")
    print(f"Max iterations: {max_iter}")
    print(f"Weight threshold: {w_threshold}")
    print("------------------------\n")

    # Build model
    model = NotearsMLP(dims=[data.shape[1]] + hidden_layers, bias=True)
    model = model.float()

    # Training with tqdm progress bar
    print("Starting training...")
    W_est = None
    for i in tqdm(range(1), desc="NOTEARS Optimization", ncols=80):
        W_est = notears_nonlinear(
            model,
            X,
            lambda1=lambda1,
            lambda2=lambda2,
            max_iter=max_iter,
            w_threshold=w_threshold
        )

    print("NOTEARS-MLP training complete.")

    # Inspect results
    print("\n--- Model Output Summary ---")
    print(f"Adjacency shape: {W_est.shape}")
    print(f"Nonzero edges: {np.sum(W_est > 0)}")
    print(f"Max weight: {np.max(W_est):.4f}, Min weight: {np.min(W_est):.4f}")
    print("-----------------------------")

    # Save results
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "est_dag_notears_mlp.csv")
    pd.DataFrame(W_est).to_csv(out_path, index=False, header=False)
    print(f"Saved estimated DAG to {out_path}")


def main():
    base_dir = os.path.join(os.path.dirname(__file__), "..")
    data_path = os.path.join(base_dir, "data", "raw", "data_20vars.csv")
    out_dir = os.path.join(base_dir, "results", "graphs")
    run_notears_mlp(data_path, out_dir)


if __name__ == "__main__":
    main()
