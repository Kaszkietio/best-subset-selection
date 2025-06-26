import os
from itertools import product
from math import ceil
import numpy as np
import sys
import pandas as pd
from timeit import default_timer
from tqdm import tqdm


sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from generate_synthetic import SyntheticDataGenerator
from dfo import DfoModel
from mio import BssMioModel

RESULTS_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "results")

data_functions = {
    "data_1": SyntheticDataGenerator.data_1,
    "data_2": SyntheticDataGenerator.data_2,
    "data_3": SyntheticDataGenerator.data_3,
    "data_4": SyntheticDataGenerator.data_4
}

def get_config():
    import argparse
    import json
    parser = argparse.ArgumentParser(description="Generate synthetic data for testing.")
    parser.add_argument("--config", type=str,
                        default=os.path.join(os.path.dirname(__file__), "config.json"),
                        help="Path to the configuration file.")
    config_path = parser.parse_args()
    with open(config_path.config, 'r') as f:
        config = json.load(f)
    return config


def run_experiment(experiment: dict):
    data_params = experiment["data_params"]
    mio_params = experiment["mio_params"]
    dfo_params = experiment["dfo_params"]
    data_function = experiment["data"]

    X, y, beta_true = data_function(**data_params)
    n, p = X.shape
    dfo = DfoModel(**dfo_params)
    dfo_best_objective = float('inf')
    dfo_best_beta = None
    start = default_timer()
    for i in tqdm(range(50)):
        eps = np.random.normal(0.0, 4.0, size=p)
        beta_start = min(float(i) - 1.0, 1.0)*eps
        dfo.beta_start = beta_start
        dfo_beta = dfo.optimize(X, y)
        dfo_objective = dfo._objective(X, y, dfo_beta)
        if dfo_objective < dfo_best_objective:
            dfo_best_objective = dfo_objective
            dfo_best_beta = dfo_beta
    end = default_timer()
    dfo_time = int(ceil(end - start))

    print(f"DFO best objective: {dfo_best_objective}, time: {dfo_time} seconds")

    mio = BssMioModel(dfo=dfo, **mio_params)
    mio.beta_warm = dfo_best_beta
    mio.optimize_2_4(X, y)

    mio_cold = BssMioModel(dfo=None, **experiment["mio_params"])
    mio_cold.optimize_2_4(X, y)

    df = pd.DataFrame({
        "data": ["data_2"],
        "k": [mio.k],
        "snr": [data_params["snr"]],
        "beta_true": [beta_true.tolist()],

        "dfo_beta": [dfo_best_beta.tolist()],
        "dfo_objective": [dfo_best_objective],
        "dfo_time": [dfo_time],

        "warm_beta_opt": [mio.beta_opt.tolist()],
        "warm_objective": [mio.model.ObjVal],
        "warm_incumbent": [mio.incumbent_time],
        "warm_status": [mio.model.Status],
        "warm_time": [mio.model.Runtime],
        "warm_mip_gap": [mio.model.MIPGap],

        "cold_beta": [mio_cold.beta_opt.tolist()],
        "cold_objective": [mio_cold.model.ObjVal],
        "cold_incumbent": [mio_cold.incumbent_time],
        "cold_status": [mio_cold.model.Status],
        "cold_time": [mio_cold.model.Runtime],
        "cold_mip_gap": [mio_cold.model.MIPGap],
    })
    return df


def main():
    snrs = [3.0, 7.0]
    k0 = [5, 7, 9]
    config = {
        "data": SyntheticDataGenerator.data_2,
        "data_params": {
            "n_samples": 30,
            "n_features": 500
        },
        "mio_params": {
            "tau": 2.0,
            "sos": True,
            "tighter_bounds": False,
            "time_limit": 250
        },
        "dfo_params": {
            "max_iter":100,
            "tol": 1e-4,
            "verbose": False,
            "algorithm": '2'
        },
    }
    df = pd.DataFrame()

    path_dir = os.path.join(RESULTS_PATH, "upper_bound")
    os.makedirs(path_dir, exist_ok=True)

    for snr, k in product(snrs, k0):
        config["data_params"]["snr"] = snr
        config["mio_params"]["k"] = k
        config["dfo_params"]["k"] = k
        df_1 = run_experiment(config)
        df = pd.concat([df, df_1], axis=0, ignore_index=True)

    path = os.path.join(path_dir, "results_2_4.csv")
    df.to_csv(path, index=False)
    print(f"Results saved to {path}")

if __name__ == "__main__":
    main()