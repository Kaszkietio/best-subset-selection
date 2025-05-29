from tqdm import tqdm
import os
import pandas as pd

from generate_synthetic import SyntheticDataGenerator
from dfo import DfoModel
from mio import BssMioModel

RESULTS_PATH = os.path.join(os.path.dirname(__file__), "..", "results")

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
    data_function = data_functions[experiment["data"]]
    k = experiment["k"]

    X, y, beta_true = data_function(**data_params)
    dfo = DfoModel(k)
    mio = BssMioModel(k=k, dfo=dfo, **experiment["mio_params"])
    beta_opt, objective, mip_gap, status = mio.optimize(X, y)

    mio_cold = BssMioModel(k=k, dfo=None, **experiment["mio_params"])
    beta_cold, objective_cold, mip_gap_cold, status_cold = mio_cold.optimize(X, y)

    df = pd.DataFrame({
        "data": [experiment["data"]],
        "k": [k],
        "objective": [objective],
        "status": [status],
        "mip_gap": [mip_gap],
        "beta_true": [beta_true.tolist()],
        "beta_opt": [beta_opt.tolist()],
        "use_dfo": [experiment.get("use_dfo", False)],
        "beta_warm": [mio.beta_warm.tolist() if mio.beta_warm is not None else None],
        "beta_cold": [beta_cold.tolist()],
        "objective_cold": [objective_cold],
        "status_cold": [status_cold],
        "mip_gap_cold": [mip_gap_cold],
        **data_params,
        **mio_params
    })
    return df


def main():
    config = get_config()

    df = pd.DataFrame()

    print("Tests: ", config["test_name"])
    for experiment in tqdm(config['experiments']):
        df_1 = run_experiment(experiment)
        df = pd.concat([df, df_1], axis=0, ignore_index=True)

    path = os.path.join(RESULTS_PATH, f"{config['test_name']}.csv")
    df.to_csv(path, index=False)
    print(f"Results saved to {path}")

if __name__ == "__main__":
    main()