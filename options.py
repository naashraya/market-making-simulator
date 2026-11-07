import numpy as np
from scipy.stats import norm


def bs_price(S0, K, r, sigma, T, kind="call"):
    d1 = (np.log(S0 / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if kind == "call":
        return S0 * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    return K * np.exp(-r * T) * norm.cdf(-d2) - S0 * norm.cdf(-d1)


def mc_price(S0, K, r, sigma, T, n_draws, kind="call", antithetic=False, seed=None):
    """Both modes use the same number of normal draws (equal cost)."""
    rng = np.random.default_rng(seed)
    Z = rng.standard_normal(n_draws // 2 if antithetic else n_draws)

    def payoff(z):
        ST = S0 * np.exp((r - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * z)
        return np.maximum(ST - K, 0) if kind == "call" else np.maximum(K - ST, 0)

    pay = 0.5 * (payoff(Z) + payoff(-Z)) if antithetic else payoff(Z)
    disc = np.exp(-r * T)
    price = disc * pay.mean()
    stderr = disc * pay.std(ddof=1) / np.sqrt(len(pay))
    return price, stderr


if __name__ == "__main__":
    S0, K, r, sigma, T = 100, 105, 0.03, 0.2, 1.0
    n = 100_000
    exact = bs_price(S0, K, r, sigma, T)
    p1, se1 = mc_price(S0, K, r, sigma, T, n, antithetic=False, seed=7)
    p2, se2 = mc_price(S0, K, r, sigma, T, n, antithetic=True, seed=7)

    print(f"Black-Scholes : {exact:.4f}")
    print(f"Plain MC      : {p1:.4f} ± {1.96*se1:.4f}  (95% CI)")
    print(f"Antithetic MC : {p2:.4f} ± {1.96*se2:.4f}  (95% CI)")
    print(
        f"Variance reduction: {(se1/se2)**2:.2f}x   |   std-error cut: {1-se2/se1:.0%}")

    # Convergence study: error vs. number of draws
    for m in [1_000, 10_000, 100_000, 1_000_000]:
        a, _ = mc_price(S0, K, r, sigma, T, m, seed=1)
        b, _ = mc_price(S0, K, r, sigma, T, m, antithetic=True, seed=1)
        print(
            f"n={m:>9,}  plain err={abs(a-exact):.5f}   antithetic err={abs(b-exact):.5f}")
