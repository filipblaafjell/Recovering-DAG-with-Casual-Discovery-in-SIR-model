import numpy as np
from castle.algorithms import GraNDAG
from castle.common import GraphDAG

# Generate toy data (n=1000 samples, d=5 variables)
X = np.random.randn(1000, 5)

# Initialize model
model = GraNDAG(hiddens=[16, 16], lambda1=0.02, lambda2=0.005, lr=0.001)

# Train model
model.learn(X)

# Retrieve estimated adjacency matrix
GraphDAG(model.causal_matrix)

print("Estimated DAG:\n", model.causal_matrix)
