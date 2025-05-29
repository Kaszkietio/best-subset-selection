import numpy as np
from scipy.optimize import minimize_scalar

from generate_synthetic import SyntheticDataGenerator

class DfoModel:
    def __init__(self, k: int, max_iter=100, tol=1e-4, verbose=False, algorithm='2'):
        self.k = k
        self.max_iter = max_iter
        self.tol = tol
        self.verbose = verbose
        if algorithm not in ['1', '2']:
            raise ValueError("Algorithm must be '1' or '2'")
        self.algorithm = algorithm

    def optimize(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        if self.algorithm == '1':
            return self._algorithm_1(X, y)
        elif self.algorithm == '2':
            return self._algorithm_2(X, y)

    def _algorithm_1(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        n, p = X.shape
        if p < self.k:
            raise ValueError("Number of samples must be greater or equal than k")

        # We need L >= lipshitz constant
        L = np.linalg.norm(X.T @ X, 2)
        alpha = 1 / L
        beta = np.zeros((p,))
        val = self._objective(X, y, beta)
        if self.verbose:
            print(f"Using Lipschitz constant L = {L}, step size alpha = {alpha}")
            print(f"Starting optimization with k = {self.k}, max_iter = {self.max_iter}, tol = {self.tol}")
            print(f"Number of samples: {n}, Number of features: {p}")
            print(f"Initial coefficients: {beta}")
            print(f"Initial objective value: {val}")

        for iteration in range(self.max_iter):
            # Compute the gradient
            grad = X.T @ (X @ beta - y)

            # Update the coefficients
            beta_new = self._hard_threshold(beta - alpha * grad)

            # Check for convergence
            val_new = self._objective(X, y, beta_new)
            if self.verbose and iteration % 10 == 0:
                print(f"Iteration {iteration + 1}:")
                print(f"\tGradient norm: {np.linalg.norm(grad)}")
                print(f"\tUpdated coefficients:", beta_new.round(4))
                print(f"\tObjective value: {val_new}")
            if val - val_new < self.tol:
                break
            beta = beta_new
            val = val_new

        if self.verbose:
            print(f"Converged after {iteration + 1} iterations.")

        support = np.flatnonzero(beta)
        Xk = X[:, support]
        beta_polished = np.zeros(p)
        beta_ls = np.linalg.lstsq(Xk, y, rcond=None)[0]
        beta_polished[support] = beta_ls

        return beta

    def _algorithm_2(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        n, p = X.shape
        if p < self.k:
            raise ValueError("Number of samples must be greater than or equal k")

        # We need L >= lipshitz constant
        L = np.linalg.norm(X.T @ X, 2)
        alpha = 1 / L
        beta = np.zeros(p)
        val = self._objective(X, y, beta)
        eta_list = []
        g_eta_list = []
        if self.verbose:
            print(f"Using Lipschitz constant L = {L}, step size alpha = {alpha}")
            print(f"Starting optimization with k = {self.k}, max_iter = {self.max_iter}, tol = {self.tol}")
            print(f"Number of samples: {n}, Number of features: {p}")
            print(f"Initial coefficients: {beta}")
            print(f"Initial objective value: {val}")

        for iteration in range(self.max_iter):
            grad = X.T @ (X @ beta - y)
            eta = self._hard_threshold(beta - alpha * grad)

            # Line search for lambda in [0, 1] minimizing g(lambda*η + (1 - lambda)*beta)
            def line_obj(lmbda):
                trial_beta = lmbda * eta + (1 - lmbda) * beta
                return self._objective(X, y, trial_beta)

            res = minimize_scalar(line_obj, bounds=(0, 1), method='bounded')
            lambda_star = res.x

            beta_new = lambda_star * eta + (1 - lambda_star) * beta
            g_eta = self._objective(X, y, eta)

            if self.verbose and iteration % 10 == 0:
                print(f"Iter {iteration}: g(eta)={g_eta:.4f}, lambda={lambda_star:.4f}")
                print(f"Iteration {iteration + 1}:")
                print(f"\tGradient norm: {np.linalg.norm(grad)}")
                print(f"\tUpdated coefficients:", beta_new.round(4))
                print(f"\tObjective value: {g_eta:.4f}")

            # Check convergence
            if iteration > 0 and abs(g_eta - g_eta_list[-1]) < self.tol:
                break

            eta_list.append(eta)
            g_eta_list.append(g_eta)
            beta = beta_new

        # Pick best eta (lowest objective)
        best_idx = np.argmin(g_eta_list)
        eta_star = eta_list[best_idx]

        support = np.flatnonzero(eta_star)
        Xk = X[:, support]
        beta_polished = np.zeros(p)
        beta_ls = np.linalg.lstsq(Xk, y, rcond=None)[0]
        beta_polished[support] = beta_ls

        return beta_polished



    def _hard_threshold(self, c: np.ndarray) -> np.ndarray:
        """Hard thresholding operator."""
        if self.k >= len(c):
            return c
        idx = np.argsort(np.abs(c))[-self.k:]
        mask = np.zeros_like(c, dtype=bool)
        mask[idx] = True
        return np.where(mask, c, 0)

    def _objective(self, X: np.ndarray, y: np.ndarray, beta: np.ndarray) -> float:
        """Compute the objective value."""
        residual = y - X @ beta
        return 0.5 * np.dot(residual, residual)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="DFO Optimization Example")
    parser.add_argument("--k", type=int, default=5, help="Number of non-zero coefficients to keep")
    parser.add_argument("--max_iter", type=int, default=1000, help="Maximum number of iterations")
    parser.add_argument("--tol", type=float, default=1e-6, help="Convergence tolerance")
    parser.add_argument("--n_samples", type=int, default=30, help="Number of samples")
    parser.add_argument("--n_features", type=int, default=100, help="Number of features")
    parser.add_argument("--n_indices", type=int, default=5, help="Number of indices to select")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    args = parser.parse_args()

    n_features = args.n_features
    n_samples = args.n_samples
    n_indices = args.n_indices
    seed = args.seed
    np.random.seed(seed)

    X, y, beta = SyntheticDataGenerator.data_2(n_samples, n_features, snr=20.0)
    print("Random indices selected:", beta.nonzero())

    print("\n\nOptimizing with DFO...")
    dfo = DfoModel(k=args.k, max_iter=args.max_iter, tol=args.tol, verbose=True)
    beta_opt = dfo.optimize(X, y)
    print("Optimized coefficients:", beta_opt.round(3))
    print("Indices of non-zero coefficients:", np.nonzero(beta_opt)[0])
    print("Number of non-zero coefficients:", np.count_nonzero(beta_opt))
    print("Objective value:", np.linalg.norm(y - X @ beta_opt, 2) ** 2 / 2)


    print("\n\nOptimizing with Algorithm 2...")
    beta_opt_2 = dfo._algorithm_2(X, y)
    print("Optimized coefficients:", beta_opt_2.round(3))
    print("Indices of non-zero coefficients:", np.nonzero(beta_opt_2)[0])
    print("Number of non-zero coefficients:", np.count_nonzero(beta_opt_2))
    print("Objective value:", np.linalg.norm(y - X @ beta_opt_2, 2) ** 2 / 2)