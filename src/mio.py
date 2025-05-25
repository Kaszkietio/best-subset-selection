import numpy as np
import gurobipy as gp
from gurobipy import GRB

from dfo import DfoModel

class BssMioModel:
    def __init__(self, k: int, dfo: DfoModel = None, tau: float = 2.0):
        self.k = k
        self.tau = tau
        self.dfo = dfo


    def optimize(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        n, p = X.shape
        beta_warm = self.dfo.optimize(X, y)
        MU = self.tau * np.max(np.abs(beta_warm))

        model = gp.Model("Best Subset Selection")
        model.setParam("OutputFlag", 0)

        beta = model.addVars(p, lb=-MU, ub=MU, vtype=GRB.CONTINUOUS, name="beta")
        z = model.addVars(p, vtype=GRB.BINARY, name="z")

        for i in range(p):
            model.addConstr(beta[i] <= MU * z[i])
            model.addConstr(beta[i] >= -MU * z[i])

        model.addConstr(gp.quicksum(z[i] for i in range(p)) <= self.k)

        # Quadratic loss
        expr = gp.QuadExpr()
        for i in range(n):
            row_expr = gp.LinExpr(gp.quicksum(X[i, j] * beta[j] for j in range(p)))
            expr.add((row_expr - y[i]) * (row_expr - y[i]))
        model.setObjective(0.5 * expr, GRB.MINIMIZE)

        model.optimize()
        beta_opt = np.array([beta[i].X for i in range(p)])
        return beta_opt, model.ObjVal


    def _get_upper_bounds(self, X: np.ndarray, y: np.ndarray):
        if self.dfo:
            beta_warm = self.dfo.optimize(X, y)
            print("Beta warm start:", beta_warm)
            MU = self.tau * np.max(np.abs(beta_warm))
            beta_1 = self.k * MU
            beta_inf = MU
        else:
            raise ValueError("DfoModel is not provided for warm start optimization.")

        x_beta_1 = min(np.sum(np.linalg.vector_norm(X, ord=2, axis=1)) * beta_1,
                       np.sqrt(self.k)*np.linalg.vector_norm(y, ord=2))

        top_k_sum = np.sum(np.sort(np.abs(X), axis=1)[:, -self.k:], axis=1)
        x_beta_inf = np.max(top_k_sum) * beta_inf

        return beta_1, beta_inf, x_beta_1, x_beta_inf



if __name__ == "__main__":
    import numpy as np
    from generate_synthetic import SyntheticDataGenerator
    from dfo import DfoModel

    n_samples = 3
    n_features = 5
    n_informative = 3
    ro = 0
    snr = 5.0

    generator = SyntheticDataGenerator()
    X, y, beta = generator.data_1(n_samples, n_features, n_informative, ro, snr)

    dfo = DfoModel(k=n_informative, max_iter=1000, tol=1e-4, verbose=True)
    bss_model = BssMioModel(k=n_informative, dfo=dfo)


