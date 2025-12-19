# Benchmarking Causal Discovery Algorithms

DTU Special Course - Department of Technology, Management and Economics

## Overview

This project benchmarks various causal discovery algorithms on synthetic data generated from Erdős–Rényi (ER) and Scale-Free (SF) graphs, using both linear and nonlinear structural causal models (SCMs). The goal is to evaluate the performance of gradient-based methods like GraNDAG and NOTEARS-MLP against traditional constraint-based and score-based baselines like PC and GES.

## Project Structure

```
├── experiments/
│   ├── grandag_ER/          # GraNDAG on ER graphs
│   │   ├── results.txt      # Performance metrics
│   │   └── saved_runs/      # Saved adjacency matrices and data
│   ├── grandag_SF/          # GraNDAG on SF graphs
│   ├── notearsmlp_ER/       # NOTEARS-MLP on ER graphs
│   ├── notearsmlp_SF/       # NOTEARS-MLP on SF graphs
│   ├── nonlinear_ER_baseline/  # PC and GES on nonlinear ER
│   └── nonlinear_SF_baseline/  # PC and GES on nonlinear SF
├── src/
│   ├── run_grandag.py       # GraNDAG experiment script
│   ├── run_notearsmlp.py    # NOTEARS-MLP experiment script
│   ├── compare_baselines.py # PC and GES baseline script
│   ├── generate_data.py     # Synthetic data generation
│   ├── evaluation/
│   │   └── metrics.py       # Evaluation metrics (SHD, SID, etc.)
│   └── notears/             # Local NOTEARS implementation
├── requirements.txt         # Python dependencies
└── README.md
```

## Data Generation

Synthetic data is generated using:
- **Graph Types**: Erdős–Rényi (ER) and Scale-Free (SF) directed acyclic graphs (DAGs)
- **SCM Types**: Linear (Gaussian noise) and nonlinear (using Gaussian Processes)
- **Node Sizes**: 5, 10, 15 nodes
- **Sample Sizes**: 500, 1000, 2000 samples per experiment
- **Trials**: 20 Monte Carlo trials per configuration

## Algorithms

- **GraNDAG**: Gradient-based DAG learning using NOTEARS constraint
- **NOTEARS-MLP**: Nonlinear NOTEARS with multi-layer perceptrons
- **PC**: Peter-Clark algorithm (constraint-based)
- **GES**: Greedy Equivalence Search (score-based)

## Metrics

- **Adjacency Precision/Recall/F1** (AP/AR/A-F1): Edge existence accuracy
- **Arrowhead Precision/Recall/F1** (HP/HR/H-F1): Edge orientation accuracy
- **SHD**: Structural Hamming Distance
- **SID**: Structural Intervention Distance
- **Time**: Runtime in seconds

## Setup

```bash
pip install -r requirements.txt
```

**Note:** This project uses the CDT library, which requires R installation for certain causal discovery algorithms and metrics. Ensure R is installed and accessible. You may also need to install R packages such as `pcalg`, `kpcalg`, and `devtools`. On macOS, you can install R via Homebrew:

```bash
brew install r
```

Then, in R:

```r
install.packages(c("pcalg", "kpcalg", "devtools"))
```

## Running Experiments

To run the experiments (note: experiments may take significant time due to Monte Carlo trials):

```bash
# Run GraNDAG on ER graphs
python src/run_grandag.py  # (Modify GRAPH_TYPE inside script)

# Run NOTEARS-MLP on ER graphs
python src/run_notearsmlp.py  # (Modify GRAPH_TYPE inside script)

# Run PC and GES baselines
python src/compare_baselines.py
```

Results are saved in the `experiments/` directory with performance summaries and saved model outputs.

## Results

Experiments show that gradient-based methods like GraNDAG and NOTEARS-MLP generally outperform traditional methods on nonlinear data, though they can be computationally expensive. Detailed results are available in the `results.txt` files within each experiment directory.
