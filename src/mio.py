import numpy as np
import gurobipy as gp
from gurobipy import GRB
from warnings import warn

from dfo import DfoModel

class BssMioModel:
    def __init__(
            self,
            k: int,
            dfo: DfoModel = None,
            tau: float = 2.0,
            sos: bool = True,
            tighter_bounds: bool = True,
            time_limit: int = 500
    ):
        self.k = k
        self.tau = tau
        self.dfo = dfo
        self.sos = sos
        self.time_limit = time_limit
        self.tighter_bounds = tighter_bounds
        if not dfo and self.tighter_bounds:
            warn("Tighter bounds are enabled, but no DFO model is provided. " \
            "Model will throw if one of the 1-norm bounds cannot be computed.")
        # Initial bounds
        self.beta_1 = float('inf')
        self.beta_inf = float('inf')
        self.zeta_1 = float('inf')
        self.zeta_inf = float('inf')


    def optimize(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        n, p = X.shape
        self.beta_warm = dfo.optimize(X, y) if self.dfo else None
        self._compute_bounds(X, y)
        print(f"Upper bounds: beta_1={self.beta_1}, beta_inf={self.beta_inf}, x_beta_1={self.zeta_1}, x_beta_inf={self.zeta_inf}")

        # Create model
        model = gp.Model("subset_selection_f2.6")
        model.setParam("OutputFlag", 0)
        model.setParam("TimeLimit", self.time_limit)

        # Variables
        beta = model.addVars(p, lb=-self.beta_inf, ub=self.beta_inf, vtype=GRB.CONTINUOUS, name="beta")
        z = model.addVars(p, vtype=GRB.BINARY, name="z")
        # zeta = X @ beta
        zeta = model.addVars(n, lb=-self.zeta_inf, ub=self.zeta_inf, vtype=GRB.CONTINUOUS, name="zeta")
        for i in range(n):
            model.addConstr(zeta[i] == gp.quicksum(X[i, j] * beta[j] for j in range(p)))

        # Linking constraints
        if self.sos:

            # Link constraint via SOS-1: if z[i] == 1 => beta[i] == 0
            for i in range(p):
                model.addSOS(GRB.SOS_TYPE1, [beta[i], z[i]])

            model.addConstr(gp.quicksum(z[i] for i in range(p)) >= p - self.k, "sparsity_constraint")

        else:
            # Linking constraints: if z[i] == 1 => beta[i] is bounded by beta_inf
            for i in range(p):
                model.addConstr(beta[i] <= self.beta_inf * z[i])
                model.addConstr(beta[i] >= -self.beta_inf * z[i])

            # Sparsity constraint
            model.addConstr(gp.quicksum(z[i] for i in range(p)) <= self.k)

        # Bounds on beta and zeta
        if self.tighter_bounds:
            # Optional bounds to tighten the model
            beta_abs = model.addVars(p, vtype=GRB.CONTINUOUS, name="beta_abs")
            for i in range(p):
                model.addConstr(beta_abs[i] == gp.abs_(beta[i]), name=f"beta_abs_{i}_constraint")
            model.addConstr(gp.quicksum(beta_abs[i] for i in range(p)) <= self.beta_1, "beta_abs_sum_constraint")

            zeta_abs = model.addVars(n, vtype=GRB.CONTINUOUS, name="zeta_abs")
            for i in range(n):
                model.addConstr(zeta_abs[i] == gp.abs_(zeta[i]), name=f"zeta_abs_{i}_constraint")
            model.addConstr(gp.quicksum(zeta_abs[i] for i in range(n)) <= self.zeta_1, "zeta_abs_sum_constraint")

        # Objective: 0.5 * ||y - zeta||^2
        obj = gp.QuadExpr()
        for i in range(n):
            obj += (zeta[i] - y[i]) * (zeta[i] - y[i])

        # Warm start
        if self.dfo:
            for i in range(p):
                beta[i].start = self.beta_warm[i]
                z[i].start = 1 if self.beta_warm[i] != 0 else 0
            for i in range(n):
                zeta[i].start = float(np.dot(X[i], self.beta_warm))
        model.update()

        model.setObjective(obj, GRB.MINIMIZE)
        model.optimize()

        beta_opt = np.array([beta[i].X for i in range(p)])
        return beta_opt, model.ObjVal, model.Status


    # def _optimize_wihout_SOS(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
    #     n, p = X.shape

    #     # Estimate bounds
    #     if self.dfo:
    #         beta_warm = self.dfo.optimize(X, y)
    #     else:
    #         beta_warm = np.zeros(p)
    #     print("Beta warm start:", beta_warm)
    #     bigM = self.tau * np.max(np.abs(beta_warm))
    #     zeta_bd = bigM * np.max(np.sum(np.abs(np.sort(X, axis=1)[:, -self.k:]), axis=1))
    #     print(f"bigM: {bigM}, zeta_bd: {zeta_bd}")

    #     # Create model
    #     model = gp.Model("subset_selection_f2.6")
    #     model.setParam("OutputFlag", 0)
    #     model.setParam("TimeLimit", self.time_limit)

    #     # Variables
    #     beta = model.addVars(p, lb=-bigM, ub=bigM, vtype=GRB.CONTINUOUS, name="beta")
    #     z = model.addVars(p, vtype=GRB.BINARY, name="z")
    #     zeta = model.addVars(n, lb=-zeta_bd, ub=zeta_bd, vtype=GRB.CONTINUOUS, name="zeta")

    #     # Linking constraints
    #     for i in range(p):
    #         model.addConstr(beta[i] <= bigM * z[i])
    #         model.addConstr(beta[i] >= -bigM * z[i])

    #     # Sparsity constraint
    #     model.addConstr(gp.quicksum(z[i] for i in range(p)) <= self.k)

    #     # zeta = X * beta
    #     for i in range(n):
    #         model.addConstr(zeta[i] == gp.quicksum(X[i, j] * beta[j] for j in range(p)))

    #     # Objective: minimize 0.5 * ||y - zeta||^2
    #     obj = gp.QuadExpr()
    #     for i in range(n):
    #         obj += (zeta[i] - y[i]) * (zeta[i] - y[i])
    #     model.setObjective(0.5 * obj, GRB.MINIMIZE)

    #     # Warm start
    #     if beta_warm is not None:
    #         for i in range(p):
    #             beta[i].start = beta_warm[i]
    #             z[i].start = 1 if beta_warm[i] != 0 else 0
    #         for i in range(n):
    #             zeta[i].start = float(np.dot(X[i], beta_warm))

    #     model.optimize()

    #     beta_opt = np.array([beta[i].X for i in range(p)])
    #     return beta_opt, model.ObjVal, model.Status

    # def _optimize_with_SOS(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
    #     n, p = X.shape
    #     self.beta_warm = dfo.optimize(X, y) if self.dfo else None
    #     beta_1, beta_inf, zeta_1, zeta_inf = self._get_upper_bounds(X, y)
    #     print(f"Upper bounds: beta_1={beta_1}, beta_inf={beta_inf}, x_beta_1={zeta_1}, x_beta_inf={zeta_inf}")
    #     model = gp.Model("subset_selection_f2.6")
    #     model.setParam("OutputFlag", 0)
    #     model.setParam("TimeLimit", self.time_limit)

    #     beta = model.addVars(p, lb=-beta_inf, ub=beta_inf, vtype=GRB.CONTINUOUS, name="beta")
    #     z = model.addVars(p, vtype=GRB.BINARY, name="z")
    #     zeta = model.addVars(n, lb=-zeta_inf, ub=zeta_inf, vtype=GRB.CONTINUOUS, name="zeta")


    #     if self.dfo:
    #         for i in range(p):
    #             beta[i].start = self.beta_warm[i]
    #             z[i].start = 1 if self.beta_warm[i] != 0 else 0
    #         for i in range(n):
    #             zeta[i].start = float(np.dot(X[i], self.beta_warm))
    #     model.update()

    #     # Link constraint via SOS-1: if z[i] == 0 => beta[i] == 0
    #     for i in range(p):
    #         model.addSOS(GRB.SOS_TYPE1, [beta[i], z[i]])

    #     model.addConstr(gp.quicksum(z[i] for i in range(p)) >= p - self.k, "sparsity_constraint")

    #     # zeta = X @ beta
    #     for i in range(n):
    #         model.addConstr(zeta[i] == gp.quicksum(X[i, j] * beta[j] for j in range(p)))

    #     # Optional bounds to tighten the model
    #     beta_abs = model.addVars(p, vtype=GRB.CONTINUOUS, name="beta_abs")
    #     for i in range(p):
    #         model.addConstr(beta_abs[i] == gp.abs_(beta[i]), name=f"beta_abs_{i}_constraint")
    #     model.addConstr(gp.quicksum(beta_abs[i] for i in range(p)) <= beta_1, "beta_abs_sum_constraint")

    #     zeta_abs = model.addVars(n, vtype=GRB.CONTINUOUS, name="zeta_abs")
    #     for i in range(n):
    #         model.addConstr(zeta_abs[i] == gp.abs_(zeta[i]), name=f"zeta_abs_{i}_constraint")
    #     model.addConstr(gp.quicksum(zeta_abs[i] for i in range(n)) <= zeta_1)

    #     # Objective: 0.5 * ||y - zeta||^2
    #     obj = gp.QuadExpr()
    #     for i in range(n):
    #         obj += (zeta[i] - y[i]) * (zeta[i] - y[i])

    #     model.setObjective(obj, GRB.MINIMIZE)
    #     model.optimize()

    #     beta_opt = np.array([beta[i].X for i in range(p)])
    #     return beta_opt, model.ObjVal, model.Status


    def _compute_bounds(self, X: np.ndarray, y: np.ndarray) -> tuple:
        if self.dfo:
            print("Beta warm start:", self.beta_warm)
            MU = self.tau * np.max(np.abs(self.beta_warm))
            self.beta_1 = self.k * MU
            self.beta_inf = MU
        else:
            coherence = np.abs(X.T @ X)
            np.fill_diagonal(coherence, 0)
            print("Coherence matrix:", coherence)
            mu = np.max(coherence)
            print("mu", mu)
            mu_k_1 = mu * (self.k - 1)
            print("mu_k_1", mu_k_1)
            if mu_k_1 < 1.0:
                gamma_k = 1 - mu_k_1
                print("gamma_k", gamma_k)

                target_corr = np.abs(X.T @ y)
                print("Target correlation:", target_corr)
                self.beta_1 = np.sum(target_corr) / (1 - mu_k_1)
                beta_inf_1 = np.sqrt(np.sum(target_corr**2, axis=0))/gamma_k
                beta_inf_2 = np.linalg.vector_norm(y, ord=2)/np.sqrt(gamma_k)
                print("beta_inf_1:", beta_inf_1)
                print("beta_inf_2:", beta_inf_2)
                self.beta_inf = np.min([beta_inf_1, beta_inf_2])
            else:
                self.beta_1 = float('inf')
                self.beta_inf = float('inf')


        self.zeta_1 = min(np.sum(np.linalg.vector_norm(X, ord=2, axis=1)) * self.beta_1,
                       np.sqrt(self.k)*np.linalg.vector_norm(y, ord=2))

        top_k_sum = np.sum(np.sort(np.abs(X), axis=1)[:, -self.k:], axis=1)
        self.zeta_inf = np.max(top_k_sum) * self.beta_inf



if __name__ == "__main__":
    import numpy as np
    from generate_synthetic import SyntheticDataGenerator
    from dfo import DfoModel
    from timeit import default_timer

    n_samples = 50
    n_features = 1000
    n_informative = 5
    ro = 0.0
    snr = 10.0

    generator = SyntheticDataGenerator()
    X, y, beta = generator.data_1(n_samples, n_features, n_informative, ro, snr)

    dfo = DfoModel(k=n_informative, max_iter=1000, tol=1e-4, verbose=False)
    bss_model = BssMioModel(k=n_informative, dfo=None, tighter_bounds=False, sos=True, time_limit=100)
    start = default_timer()
    beta_opt, objective, status = bss_model.optimize(X, y)
    end = default_timer()
    print("######### O")
    print("Optimization time:", end - start)
    print("Optimal beta:", np.nonzero(beta_opt))
    print("Objective value:", objective)
    print("True beta:", np.nonzero(beta))
    print("Status:", status)


