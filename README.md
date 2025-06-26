# Best Subset Selection

## Overview
This repository contains an implementation of algorithms for solving the Best Subset Selection problem using Mixed Integer Optimization (MIO). The Best Subset Selection problem is a fundamental problem in statistics and machine learning, where the goal is to select a subset of features that best predicts the target variable. The provided algorithms are based on the paper [*Best Subset Selection via a Modern Optimization Lens*](https://arxiv.org/abs/1507.03133) by Berstismas. Project was prepared for the *'Optimalisation in Data Analysis'* course.

## Features
- **Mixed Integer Optimization (MIO):** Efficient implementation of MIO models for subset selection.
- **Direct Function Optimization (DFO):** Algorithms for optimizing feature selection.
- **Synthetic Data Generation:** Tools for generating synthetic datasets for testing.
- **Visualization:** Notebooks for visualizing results and comparing models.

## Repository Structure
```
LICENSE
README.md
requirements.txt
notebooks/
    best_subset_selection.ipynb
    lasso_sparsenet.R
    plots.ipynb
results/
    sos_tighter_bounds_test.csv
    test.csv
    model_comparison/
        results.csv
    upper_bound/
        good_results.csv
        results_2_4.csv
        results_2_6.csv
        results_n_samples_change_30.csv
        results.csv
src/
    __init__.py
    dfo.py
    generate_synthetic.py
    mio.py
    experiments/
        __init__.py
        config.json
        run_experiments.py
        snr_different_datasets.py
        upper_bound.py
```

## Installation

### Prerequisites
- Python 3.11 or higher
- Gurobi Optimizer

### Steps
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/best-subset-selection.git
   cd best-subset-selection
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Running Experiments
1. Generate synthetic data using `src/generate_synthetic.py`.
2. Run optimization algorithms using `src/dfo.py` and `src/mio.py`.
3. Visualize results using the notebooks in the `notebooks/` directory.

### Notebooks
- `notebooks/best_subset_selection.ipynb`: Demonstrates the use of Gurobi for solving subset selection problems.
- `notebooks/plots.ipynb`: Visualizes results from experiments.

## Results
Results from experiments are stored in the `results/` directory. Subdirectories include:
- `upper_bound/`: Results for upper bound experiments.
- `model_comparison/`: Comparison of different models.

## License
This project is licensed under the MIT License. See the `LICENSE` file for details.

## Author
Wojciech Kutak, Kinga Frańczak

## Acknowledgments
- Gurobi Optimizer for providing tools for Mixed Integer Optimization.
- Contributors to open-source libraries used in this project.
