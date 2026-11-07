import numpy as np


def simulate_gbm(S0, mu, sigma, T, n_steps, n_paths, seed=None, antithetic=False):
    """Returns array of shape (n_paths, n_steps + 1)."""
    rng = np.random.default_rng(seed)
    dt = T / n_steps

    if antithetic:
        half = n_paths // 2
        Z = rng.standard_normal((half, n_steps))
        Z = np.vstack([Z, -Z])            # mirror paths
    else:
        Z = rng.standard_normal((n_paths, n_steps))

    log_returns = (mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z
    log_paths = np.cumsum(log_returns, axis=1)
    zeros = np.zeros((Z.shape[0], 1))
    return S0 * np.exp(np.hstack([zeros, log_paths]))


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    S = simulate_gbm(100, 0.0, 0.2, 1/252, 390, 20, seed=1)
    plt.plot(S.T)
    plt.title("GBM sample paths (1 trading day, 1-min steps)")
    plt.show()
