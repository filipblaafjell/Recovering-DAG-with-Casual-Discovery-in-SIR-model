# Benchmarking Causal Discovery Algorithms

DTU Special Course - Department of Technology, Management and Economics

## Project Structure

```
├── experiments/
│   ├── experiment1_linear_mixed/     # Experiment 1: Linear SCMs with interventions
│   │   ├── data/                     # Generated synthetic data
│   │   ├── results/                  # Algorithm outputs
│   │   └── run_experiment1.py        # Main runner
│   │
│   └── archive/                      # Previous exploratory work
│
├── src/
│   ├── algorithms/                   # Algorithm wrappers
│   ├── data_generation/             # Synthetic data generators
│   ├── evaluation/                  # Metrics and evaluation
│   └── utils/                       # Helper functions
│
├── configs/                         # Configuration files
├── requirements.txt                 # Python dependencies
└── survey_causal_discovery.pdf      # Reference paper
```

## Experiment 1: Linear SCMs with Mixed Data

**Goal:** Benchmark interventional vs observational causal discovery on linear SCMs with perfect interventions.

**Data:**
- Synthetic linear DAG (15 nodes, 30 edges)
- 1 observational dataset (1000 samples)  
- 15 interventional datasets (800 samples each, perfect single-variable interventions)
- Known intervention targets

**Algorithms:**
- **GIES** - Greedy Interventional Equivalence Search (uses interventions)
- **GES** - Greedy Equivalence Search (observational baseline)
- *(IGSP/UT-IGSP not available in Python packages - would require R/pcalg)*

**Metrics:** SHD, AP/AR (adjacency), AHP/AHR (orientation), SID

**Results:** GIES achieves perfect recovery (SHD=0) using interventions, while GES cannot orient edges from observational data alone (AHP=0.031, SHD=34).

## Setup

```bash
pip install -r requirements.txt
```

## Running

```bash
# Generate data
python src/data_generation/generate_linear_data.py

# Run experiment
python experiments/experiment1_linear_mixed/run_experiment1.py
```
