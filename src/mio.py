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
            time_limit: int = 500,
            verbose: bool = True,
            MIPFocus=None
    ):
        self.k = k
        self.tau = tau
        self.dfo = dfo
        self.sos = sos
        self.time_limit = time_limit
        self.tighter_bounds = tighter_bounds
        self.verbose = verbose
        # Warm start for beta, will be set by DFO model if provided
        self.beta_warm = None
        self.MIPFocus = MIPFocus
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
        if self.beta_warm is None and self.dfo:
            self.beta_warm = self.dfo.optimize(X, y)
        self._compute_bounds(X, y)
        if self.verbose:
            print(f"Upper bounds: beta_1={self.beta_1}, beta_inf={self.beta_inf}, x_beta_1={self.zeta_1}, x_beta_inf={self.zeta_inf}")

        # Create model
        model = gp.Model("subset_selection_f2.6")
        model.setParam("OutputFlag", 1 if self.verbose else 0)
        if self.time_limit:
            model.setParam("TimeLimit", self.time_limit)
        if self.MIPFocus is not None:
            model.setParam(GRB.param.MIPFocus, self.MIPFocus)

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

        # Objective: 0.5 * zeta^t @ zeta - dot(X^t @ y, beta) + 0.5*||y||_2^2
        obj = gp.QuadExpr()
        # Precompute X^T y
        XTy = X.T @ y
        y_norm_sq = np.dot(y, y)

        # 0.5 * zeta^t @ zeta
        for i in range(n):
            obj += 0.5 * zeta[i] * zeta[i]

        # dot(X^t @ y, beta)
        for j in range(p):
            obj -= XTy[j] * beta[j]

        obj += 0.5 * y_norm_sq

        # Warm start
        if self.dfo:
            for i in range(p):
                beta[i].start = self.beta_warm[i]
                if self.sos:
                    z[i].start = 1 if self.beta_warm[i] == 0.0 else 0
                else:
                    z[i].start = 1 if self.beta_warm[i] != 0.0 else 0
            for i in range(n):
                zeta[i].start = float(np.dot(X[i], self.beta_warm))
        model.update()

        model.setObjective(obj, GRB.MINIMIZE)
        def incumbent_callback(model, where):
            if where == GRB.Callback.MIP:
                sol_count = model.cbGet(GRB.Callback.MIP_SOLCNT)
                if sol_count > 0:
                    model._incumbent_time = model.cbGet(GRB.Callback.RUNTIME)
        model.optimize(incumbent_callback)

        beta_opt = np.array([beta[i].X for i in range(p)])
        self.model = model  # Store the model for later inspection if needed
        self.beta_opt = beta_opt
        self.incumbent_time = getattr(model, '_incumbent_time', None)
        return beta_opt, model.ObjVal, model.MIPGap, model.Status


    def optimize_2_4(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        n, p = X.shape
        self.beta_warm = self.dfo.optimize(X, y) if self.dfo else None

        # Create model
        model = gp.Model("subset_selection_f2.4")
        model.setParam("OutputFlag", 1 if self.verbose else 0)

        # https://support.gurobi.com/hc/en-us/community/posts/4409420791185-Optimal-solution-found-early-but-long-time-to-close-gap
        if self.time_limit:
            model.setParam("TimeLimit", self.time_limit)
        if self.MIPFocus is not None:
            model.setParam(GRB.param.MIPFocus, self.MIPFocus)

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
            raise NotImplementedError("SOS-1 linking constraints are not implemented for non-SOS models.")

        # Objective: 0.5||y - X@beta||_2^2
        obj = gp.QuadExpr()
        for i in range(n):
            obj += 0.5 * (zeta[i] - y[i]) * (zeta[i] - y[i])

        # Warm start
        if self.dfo:
            for i in range(p):
                beta[i].start = self.beta_warm[i]
                if self.sos:
                    z[i].start = 1 if self.beta_warm[i] == 0.0 else 0
                else:
                    z[i].start = 1 if self.beta_warm[i] != 0.0 else 0
            for i in range(n):
                zeta[i].start = float(np.dot(X[i], self.beta_warm))
        model.update()

        model.setObjective(obj, GRB.MINIMIZE)
        def incumbent_callback(model, where):
            if where == GRB.Callback.MIP:
                sol_count = model.cbGet(GRB.Callback.MIP_SOLCNT)
                if sol_count == 1:
                    model._incumbent_time = model.cbGet(GRB.Callback.RUNTIME)
                    print(f"Incumbent found at time {model._incumbent_time:.2f} seconds")
        model.optimize(incumbent_callback)

        beta_opt = np.array([beta[i].X for i in range(p)])
        # Store the model for later inspection if needed
        self.model = model
        self.beta_opt = beta_opt
        self.incumbent_time = getattr(model, '_incumbent_time', None)
        return beta_opt, model.ObjVal, model.MIPGap, model.Status

    def _compute_bounds(self, X: np.ndarray, y: np.ndarray) -> tuple:
        if self.dfo:
            MU = self.tau * np.max(np.abs(self.beta_warm))
            self.beta_1 = self.k * MU
            self.beta_inf = MU
        else:
            coherence = np.abs(X.T @ X)
            np.fill_diagonal(coherence, 0)
            mu = np.max(coherence)
            mu_k_1 = mu * (self.k - 1)
            if mu_k_1 < 1.0:
                gamma_k = 1 - mu_k_1
                target_corr = np.abs(X.T @ y)
                self.beta_1 = np.sum(target_corr) / (1 - mu_k_1)
                beta_inf_1 = np.sqrt(np.sum(target_corr**2, axis=0))/gamma_k
                beta_inf_2 = np.linalg.vector_norm(y, ord=2)/np.sqrt(gamma_k)
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

    np.random.seed(43)

    n_samples = 50
    n_features = 500
    n_informative = 5
    # ro = 0.9
    snr = 7.0

    X, y, beta = SyntheticDataGenerator.data_2(n_samples, n_features, snr=snr)

    dfo = DfoModel(k=n_informative, max_iter=1000, tol=1e-4, verbose=False)
    bss_model = BssMioModel(k=n_informative, dfo=dfo, tighter_bounds=False, sos=True, time_limit=200)
    start = default_timer()
    beta_opt, objective, mip_gap, status = bss_model.optimize(X, y)
    end = default_timer()
    print("#########")
    print("Optimization time:", end - start)
    print("Optimal beta:", np.nonzero(beta_opt))
    print("Warm start beta:", np.nonzero(bss_model.beta_warm) if bss_model.beta_warm is not None else None)

    print("Objective value:", objective)
    print("True beta:", np.nonzero(beta))
    print("Status:", status)
    print("MIP Gap:", mip_gap)


