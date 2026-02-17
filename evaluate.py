"""
Backtest & Evaluate Trained DQN Agent
=====================================
Run:  python evaluate.py --model models/best_model.pt
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

sys.path.insert(0, os.path.dirname(__file__))

from data.data_loader import download_eurusd, split_data
from env.trading_env  import ForexTradingEnv
from agents.dqn_agent import DQNAgent


# -----------------------------------------------------------------------
def compute_metrics(equity_curve: list, trade_log: list, initial_balance: float) -> dict:
    eq = np.array(equity_curve)
    final = eq[-1]
    total_return = (final - initial_balance) / initial_balance * 100

    # Drawdown
    peak = np.maximum.accumulate(eq)
    drawdown = (eq - peak) / (peak + 1e-9) * 100
    max_dd = float(drawdown.min())

    # Sharpe (annualized, hourly steps → 252 * 24 = 6048 periods/year)
    rets = np.diff(eq) / (eq[:-1] + 1e-9)
    sharpe = float((rets.mean() / (rets.std() + 1e-9)) * np.sqrt(6048))

    # Win rate
    if trade_log:
        wins = sum(1 for t in trade_log if t["pnl"] > 0)
        win_rate = wins / len(trade_log) * 100
        avg_pnl  = float(np.mean([t["pnl"] for t in trade_log]))
    else:
        win_rate = 0.0
        avg_pnl  = 0.0

    return {
        "Final Balance ($)":   round(final, 2),
        "Total Return (%)":    round(total_return, 2),
        "Max Drawdown (%)":    round(max_dd, 2),
        "Sharpe Ratio":        round(sharpe, 3),
        "Total Trades":        len(trade_log),
        "Win Rate (%)":        round(win_rate, 2),
        "Avg Trade PnL ($)":   round(avg_pnl, 2),
    }


def run_episode(agent: DQNAgent, env: ForexTradingEnv) -> tuple:
    """Run one greedy episode, return (equity_curve, trade_log, price_series)."""
    agent.epsilon = 0.0
    obs, _ = env.reset()
    done = truncated = False
    actions_taken = []

    while not (done or truncated):
        action = agent.select_action(obs)
        obs, _, done, truncated, _ = env.step(action)
        actions_taken.append(action)

    return env.equity_curve, env.trade_log, env.df["Close"].values, actions_taken


def plot_backtest(
    equity_curve: list,
    trade_log: list,
    prices: np.ndarray,
    actions: list,
    metrics: dict,
    save_path: str = "results/backtest.png",
):
    os.makedirs("results", exist_ok=True)
    fig = plt.figure(figsize=(16, 10))
    gs  = gridspec.GridSpec(3, 1, height_ratios=[2, 1, 1], hspace=0.35)
    fig.suptitle("DQN Trading Bot — EUR/USD Backtest", fontsize=14, fontweight="bold")

    # Price + Actions
    ax1 = fig.add_subplot(gs[0])
    price_slice = prices[: len(actions)]
    ax1.plot(price_slice, color="gray", lw=0.8, alpha=0.7, label="EUR/USD")
    buys  = [i for i, a in enumerate(actions) if a == 1]
    sells = [i for i, a in enumerate(actions) if a == 2]
    ax1.scatter(buys,  price_slice[buys],  marker="^", color="green",  s=15, zorder=5, label="Buy")
    ax1.scatter(sells, price_slice[sells], marker="v", color="red",    s=15, zorder=5, label="Sell")
    ax1.set_title("Price + Agent Actions")
    ax1.legend(fontsize=8)

    # Equity Curve
    ax2 = fig.add_subplot(gs[1])
    ax2.plot(equity_curve, color="steelblue", lw=1.5)
    ax2.axhline(y=equity_curve[0], color="gray", linestyle="--", lw=0.8)
    ax2.set_title("Equity Curve")
    ax2.set_ylabel("Balance ($)")

    # Drawdown
    eq  = np.array(equity_curve)
    peak= np.maximum.accumulate(eq)
    dd  = (eq - peak) / (peak + 1e-9) * 100
    ax3 = fig.add_subplot(gs[2])
    ax3.fill_between(range(len(dd)), dd, color="salmon", alpha=0.7)
    ax3.set_title("Drawdown (%)")
    ax3.set_ylabel("%")

    # Metrics text
    metrics_str = " | ".join([f"{k}: {v}" for k, v in metrics.items()])
    fig.text(0.01, 0.01, metrics_str, fontsize=7.5, wrap=True, va="bottom",
             bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8))

    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Backtest] Plot saved → {save_path}")


# -----------------------------------------------------------------------
def main(model_path: str = "models/best_model.pt"):
    df = download_eurusd()
    _, _, test_df = split_data(df)

    env = ForexTradingEnv(test_df, window_size=24, initial_balance=10_000.0)
    state_dim  = env.observation_space.shape[0]
    action_dim = env.action_space.n

    agent = DQNAgent(state_dim=state_dim, action_dim=action_dim)
    agent.load(model_path)

    equity_curve, trade_log, prices, actions = run_episode(agent, env)
    metrics = compute_metrics(equity_curve, trade_log, 10_000.0)

    print("\n" + "="*50)
    print("  BACKTEST RESULTS (Test Set)")
    print("="*50)
    for k, v in metrics.items():
        print(f"  {k:<25} {v}")
    print("="*50 + "\n")

    plot_backtest(equity_curve, trade_log, prices, actions, metrics)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="models/best_model.pt")
    args = parser.parse_args()
    main(args.model)