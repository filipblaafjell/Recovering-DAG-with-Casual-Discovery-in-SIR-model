# run_notears_interventions.py
# =====================================================
# Evaluate interventional accuracy on saved NOTEARS DAGs
# =====================================================

import os
import numpy as np
from tqdm import tqdm

from generate_data import generate_scm_data
from interventional_sem import fit_sem_gp, simulate_do, interventional_mse


def run_intervention_eval():

    node_sizes = [5, 10]
    sample_sizes = [500, 2000]
    num_trials = 20

    save_dir = "experiments/nonlinear/saved_runs"

    for p in node_sizes:
        for n in sample_sizes:

            print(f"\n=== Evaluating interventions for p={p}, n={n} ===")
            mse_list = []

            for seed in tqdm(range(num_trials)):

                run_id = f"p{p}_n{n}_seed{seed}"

                adj_path  = f"{save_dir}/{run_id}_adj_est.npy"
                true_path = f"{save_dir}/{run_id}_true_adj.npy"
                obs_path  = f"{save_dir}/{run_id}_X_obs.npy"

                # Skip seeds with missing files
                if not (os.path.exists(adj_path)
                        and os.path.exists(true_path)
                        and os.path.exists(obs_path)):
                    print(f"Skipping seed {seed}: missing saved files.")
                    continue

                # Load saved DAG and observational data
                adj_est = np.load(adj_path)
                true_adj = np.load(true_path)
                X_obs = np.load(obs_path)

                # Generate TRUE intervention data
                data_envs, targets, _ = generate_scm_data(
                    num_nodes=p,
                    total_samples=n,
                    num_interventions=1,
                    intervention_size=1,
                    seed=seed
                )

                X_true_do = data_envs[1]
                target = targets[1][0]

                # Fit NONLINEAR GP SEM
                models = fit_sem_gp(X_obs, adj_est)

                # Simulate intervention
                X_est_do = simulate_do(
                    models,
                    adj_est,
                    target,
                    value=2.0,
                    n_samples=X_true_do.shape[0]
                )

                # Compute interventional prediction error
                mse = interventional_mse(X_est_do, X_true_do)
                mse_list.append(mse)

            # Summary
            if len(mse_list) > 0:
                mse_list = np.array(mse_list)
                print(f"Interventional MSE (p={p}, n={n}): "
                      f"mean={mse_list.mean():.4f}, std={mse_list.std():.4f}")
            else:
                print("No valid seeds found for this setting.")


if __name__ == "__main__":
    run_intervention_eval()
