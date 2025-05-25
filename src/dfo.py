import numpy as np

class DfoModel:
    def __init__(self, k: int, max_iter=100, tol=1e-4, verbose=False):
        self.k = k
        self.max_iter = max_iter
        self.tol = tol
        self.verbose = verbose

    def optimize(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        n, p = X.shape
        if p < self.k:
            raise ValueError("Number of samples must be greater than k")

        # We need L >= lipshitz constant
        L = np.linalg.norm(X.T @ X, 2)
        alpha = 1 / L
        beta = np.zeros((p))
        val = np.linalg.norm(y - X @ beta, 2) ** 2 / 2
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
            val_new = np.linalg.norm(y - X @ beta_new, 2) ** 2 / 2
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
        return beta

    def _hard_threshold(self, c: np.ndarray) -> np.ndarray:
        """Hard thresholding operator."""
        if self.k >= len(c):
            return c
        idx = np.argsort(np.abs(c))[-self.k:]
        mask = np.zeros_like(c, dtype=bool)
        mask[idx] = True
        return np.where(mask, c, 0)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="DFO Optimization Example")
    parser.add_argument("--k", type=int, default=5, help="Number of non-zero coefficients to keep")
    parser.add_argument("--max_iter", type=int, default=1000, help="Maximum number of iterations")
    parser.add_argument("--tol", type=float, default=1e-5, help="Convergence tolerance")
    parser.add_argument("--n_samples", type=int, default=100, help="Number of samples")
    parser.add_argument("--n_features", type=int, default=20, help="Number of features")
    parser.add_argument("--n_indices", type=int, default=5, help="Number of indices to select")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    args = parser.parse_args()

    n_features = args.n_features
    n_samples = args.n_samples
    n_indices = args.n_indices
    seed = args.seed
    np.random.seed(seed)
    random_indices = np.random.permutation(n_features)[:n_indices]
    indices_weights = np.random.rand(n_indices)
    indices_weights /= np.sum(indices_weights)  # Normalize weights
    indices_weights = 0.9 * indices_weights + 0.1

    print("Random indices selected:", random_indices)
    print("Weights for selected indices:", indices_weights.round(3))
    X = np.random.randn(n_samples, n_features)
    y = X[:, random_indices] @ indices_weights + np.random.randn(n_samples) * 0.1  # Linear combination with noise

    print("\n\nOptimizing with DFO...")
    dfo = DfoModel(k=args.k, max_iter=args.max_iter, tol=args.tol, verbose=True)
    beta = dfo.optimize(X, y)
    print("Optimized coefficients:", beta.round(3))