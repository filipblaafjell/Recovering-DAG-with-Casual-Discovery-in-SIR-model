import torch
import numpy as np
import pandas as pd
from dagma import utils
from dagma.nonlinear import DagmaMLP, DagmaNonlinear

# load dataset
data = pd.read_csv("data/raw/data_20vars.csv").values
B_true = pd.read_csv("data/raw/dag_20vars.csv", header=None).values

# Set precision
torch.set_default_dtype(torch.double)

# create nonlinear dagma model
d = data.shape[1]
eq_model = DagmaMLP(dims=[d, 10, 1], bias=True, dtype=torch.double)
model = DagmaNonlinear(eq_model, dtype=torch.double)

# fit model
W_est = model.fit(
    data,
    lambda1=0.02,   # L1 reg
    lambda2=0.005,  # L2 reg
)

# eval
acc = utils.count_accuracy(B_true, W_est != 0)
print("Results:")
print(acc)

# Save output adjacency
pd.DataFrame(W_est).to_csv("results/graphs/est_dag_dagma.csv", index=False, header=False)
print("Saved estimated DAG to results/graphs/est_dag_dagma.csv")