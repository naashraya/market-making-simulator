import numpy as np
import matplotlib.pyplot as plt
from gbm import simulate_gbm
from market_maker import run_market_maker

N_TRIALS = 5000
T, N_STEPS = 1/252, 390  # one trading day, 1-minute steps
DT = T / N_STEPS
S0 = 100.0


def sweep_spreads(sigma, spreads, seed=42):
    S = simulate_gbm(S0, 0.0, sigma, T, N_STEPS, N_TRIALS, seed=seed)
    rows = []
    for h in spreads:
        r = run_market_maker(S, h, DT, seed=seed + 1)
        rows.append({
            "h": h,
            "pnl_mean": r["pnl"].mean(),
            "pnl_std": r["pnl"].std(),
            "sharpe": r["pnl"].mean() / r["pnl"].std(),
            "inv_std": r["inventory"].std(),
            "fill_rate": r["fill_rate"].mean(),
            "informed": r["informed_share"].mean(),
        })
    return rows


# --- Experiment A: spread sweep at fixed volatility ---
spreads = np.linspace(0.01, 0.30, 15)
rows = sweep_spreads(sigma=0.20, spreads=spreads)

fig, ax = plt.subplots(1, 3, figsize=(15, 4))
ax[0].errorbar(spreads, [r["pnl_mean"] for r in rows],
               yerr=[r["pnl_std"] for r in rows], marker="o", capsize=3)
ax[0].set(title="P&L vs half-spread (±1 std)",
          xlabel="half-spread ($)", ylabel="P&L ($)")

ax[1].plot(spreads, [r["fill_rate"] for r in rows], marker="o")
ax[1].set(title="Fill rate vs half-spread",
          xlabel="half-spread ($)", ylabel="fills / quote")

ax[2].plot(spreads, [r["informed"] for r in rows], marker="o", color="tab:red")
ax[2].set(title="Adverse selection share",
          xlabel="half-spread ($)", ylabel="informed fills / fills")
plt.tight_layout()
plt.savefig("spread_sweep.png", dpi=150)
plt.show()

# --- Experiment B: volatility x spread heatmap of mean P&L ---
sigmas = [0.10, 0.20, 0.30, 0.50]
grid = np.array([[r["pnl_mean"] for r in sweep_spreads(sg, spreads)]
                for sg in sigmas])

plt.figure(figsize=(8, 4))
plt.imshow(grid, aspect="auto", origin="lower",
           extent=[spreads[0], spreads[-1], 0, len(sigmas)], cmap="RdYlGn")
plt.yticks(np.arange(len(sigmas)) + 0.5, [f"{s:.0%}" for s in sigmas])
plt.colorbar(label="mean P&L ($)")
plt.xlabel("half-spread ($)")
plt.ylabel("annual volatility")
plt.title("Mean P&L: volatility × spread")
plt.tight_layout()
plt.savefig("vol_spread_heatmap.png", dpi=150)
plt.show()
