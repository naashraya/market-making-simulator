import numpy as np


def run_market_maker(S, half_spread, dt, A=200.0, k=15.0,
                     p_informed=0.05, skew=0.002, q_max=20, seed=None):
    """
    S: (n_paths, n_steps+1) price paths. Returns dict of per-path results.
    Fair value = the simulated price S_t.
    """
    rng = np.random.default_rng(seed)
    n_paths, n_cols = S.shape
    n_steps = n_cols - 1

    cash = np.zeros(n_paths)
    inv = np.zeros(n_paths)
    n_fills = np.zeros(n_paths)
    n_informed = np.zeros(n_paths)
    max_abs_inv = np.zeros(n_paths)

    p_fill = 1.0 - np.exp(-A * np.exp(-k * half_spread) * dt)

    for t in range(n_steps):
        s, s_next = S[:, t], S[:, t + 1]

        reservation = s - skew * inv
        bid = reservation - half_spread
        ask = reservation + half_spread

        can_buy = inv < q_max       # maker may add to the long side
        can_sell = inv > -q_max     # maker may add to the short side

        informed = rng.random(n_paths) < p_informed

        # Customer buys at our ask / sells at our bid
        buy_u = rng.random(n_paths) < p_fill
        sell_u = rng.random(n_paths) < p_fill
        buy_i = informed & (s_next > ask)
        sell_i = informed & (s_next < bid)

        hit_ask = (buy_u | buy_i) & can_sell     # we SELL at ask
        hit_bid = (sell_u | sell_i) & can_buy    # we BUY at bid

        cash += ask * hit_ask - bid * hit_bid
        inv += hit_bid.astype(float) - hit_ask.astype(float)

        n_fills += hit_ask.astype(int) + hit_bid.astype(int)
        n_informed += (buy_i & can_sell).astype(int) + \
            (sell_i & can_buy).astype(int)
        max_abs_inv = np.maximum(max_abs_inv, np.abs(inv))

    pnl = cash + inv * S[:, -1]          # mark-to-market at final price
    return {
        "pnl": pnl,
        "inventory": inv,
        "max_abs_inv": max_abs_inv,
        # fills per quote opportunity
        "fill_rate": n_fills / (2 * n_steps),
        "informed_share": np.divide(n_informed, n_fills,
                                    out=np.zeros(n_paths), where=n_fills > 0),
    }
