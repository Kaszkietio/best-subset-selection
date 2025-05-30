import numpy as np

class SyntheticDataGenerator:
    @staticmethod
    def data_1(n_samples: int, n_features: int, n_informative: int, ro: float, snr: float = 1.0):
        sigma = np.zeros((n_features, n_features))
        j = np.arange(n_features)
        for i in range(n_features):
            sigma[i, :] = ro ** np.abs(i - j)
        X = np.random.multivariate_normal(np.zeros(n_features), sigma, n_samples)
        X = SyntheticDataGenerator._standardize_matrix(X)

        beta = np.zeros(n_features)
        informative_indices = np.linspace(0, n_features - 1, n_informative, dtype=int)
        beta[informative_indices] = 1.0

        y = SyntheticDataGenerator._calculate_target(X, beta, snr)
        return X, y, beta


    @staticmethod
    def data_2(n_samples: int, n_features: int, snr: float = 1.0):
        assert n_features >= 5, "n_features must be greater or equal to 5 for this dataset"
        sigma = np.eye(n_features, n_features)
        X = np.random.multivariate_normal(np.zeros(n_features), sigma, n_samples)
        X = SyntheticDataGenerator._standardize_matrix(X)

        k = 5
        beta = np.concat([np.ones(k), np.zeros(n_features - k)])
        y = SyntheticDataGenerator._calculate_target(X, beta, snr)
        return X, y, beta


    @staticmethod
    def data_3(n_samples: int, n_features: int, snr: float = 1.0):
        assert n_features >= 10, "n_features must be greater or equal to 10 for this dataset"
        sigma = np.eye(n_features, n_features)
        X = np.random.multivariate_normal(np.zeros(n_features), sigma, n_samples)
        X = SyntheticDataGenerator._standardize_matrix(X)

        k = 10
        beta = np.concat([0.5 + 9.5*np.arange(k)/(k - 1), np.zeros(n_features - k)])
        y = SyntheticDataGenerator._calculate_target(X, beta, snr)
        return X, y, beta


    @staticmethod
    def data_4(n_samples: int, n_features: int, snr: float = 1.0):
        assert n_features >= 6, "n_features must be greater or equal to 6 for this dataset"
        sigma = np.eye(n_features, n_features)
        X = np.random.multivariate_normal(np.zeros(n_features), sigma, n_samples)
        X = SyntheticDataGenerator._standardize_matrix(X)

        k = 6
        beta = np.concat([np.array([-10, -6, -2, 2, 6, 10]), np.zeros(n_features - k)])
        y = SyntheticDataGenerator._calculate_target(X, beta, snr)
        return X, y, beta


    @staticmethod
    def _standardize_matrix(X: np.ndarray):
        assert X.ndim == 2, "Input must be a 2D array"
        for feature in range(X.shape[1]):
            mean = np.mean(X[:, feature])
            X[:, feature] -= mean
            column = X[:, feature].copy()
            l2_norm = np.linalg.vector_norm(column, ord=2)
            X[:, feature] /= l2_norm
            # print(f"Feature {feature}: mean = {mean}, l2 norm = {l2_norm}")
            # print(f"Standardized feature {feature}: mean = {np.mean(X[:, feature])}, l2 norm = {np.linalg.vector_norm(X[:, feature], ord=2)}")
        return X


    @staticmethod
    def _calculate_target(X: np.ndarray, beta: np.ndarray, snr: float) -> np.ndarray:
        error_var = np.var(X @ beta) / snr
        eps = np.random.normal(0, np.sqrt(error_var), X.shape[0])
        return X @ beta + eps


if __name__ == "__main__":
    n_samples = 10
    n_features = 10
    n_informative = 3
    ro = 0.9

    X, y, beta = SyntheticDataGenerator.data_1(n_samples, n_features, n_informative, ro)
    print("Data 1:")
    print("X shape:", X.shape)
    print("y shape:", y.shape)
    print("beta", beta)

    X, y, beta = SyntheticDataGenerator.data_2(n_samples, n_features)
    print("\nData 2:")
    print("X shape:", X.shape)
    print("y shape:", y.shape)
    print("beta", beta)

    X, y, beta = SyntheticDataGenerator.data_3(n_samples, n_features=12)
    print("\nData 3:")
    print("X shape:", X.shape)
    print("y shape:", y.shape)
    print("beta", beta)

    X, y, beta = SyntheticDataGenerator.data_4(n_samples, n_features = 12)
    print("\nData 4:")
    print("X shape:", X.shape)
    print("y shape:", y.shape)
    print("beta", beta)