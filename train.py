"""
Train the DQN Agent on EUR/USD Hourly Data
==========================================
Run:  python train.py
"""

import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import deque

sys.path.insert(0, os.path.dirname(__file__))

from data.data_loader import download_eurusd, split_data
from env.trading_env  import ForexTradingEnv
from agents.dqn_agent import DQNAgent


# -----------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------
DEFAULT_CONFIG = {
    # Data
    "window_size":     24,
    "initial_balance": 10_000.0,
    "lot_size":        1_000.0,
    # Agent
    "lr":              1e-4,
    "gamma":           0.99,
    "epsilon_start":   1.0,
    "epsilon_end":     0.05,
    "epsilon_decay":   0.9995,
    "batch_size":      128,
    "target_update":   500,
    "buffer_size":     200_000,
    "hidden":          256,
    "double_dqn":      True,
    # Training
    "n_episodes":      200,
    "save_every":      25,
    "log_every":       10,
}


# -----------------------------------------------------------------------
def evaluate(agent: DQNAgent, env: ForexTradingEnv, n_episodes: int = 3) -> dict:
    """Run greedy evaluation episodes, return mean metrics."""
    rewards, balances, n_trades = [], [], []
    original_eps = agent.epsilon
    agent.epsilon = 0.0   # greedy

    for _ in range(n_episodes):
        obs, _ = env.reset()
        done = truncated = False
        ep_reward = 0.0
        while not (done or truncated):
            action = agent.select_action(obs)
            obs, reward, done, truncated, _ = env.step(action)
            ep_reward += reward
        rewards.append(ep_reward)
        balances.append(env.balance)
        n_trades.append(len(env.trade_log))

    agent.epsilon = original_eps
    return {
        "mean_reward":  float(np.mean(rewards)),
        "mean_balance": float(np.mean(balances)),
        "mean_trades":  float(np.mean(n_trades)),
    }


# -----------------------------------------------------------------------
def train(config: dict = None):
    cfg = {**DEFAULT_CONFIG, **(config or {})}
    os.makedirs("models", exist_ok=True)
    os.makedirs("logs",   exist_ok=True)
    os.makedirs("results", exist_ok=True)

    # -- Data --
    df = download_eurusd()
    train_df, val_df, test_df = split_data(df)

    # -- Environments --
    train_env = ForexTradingEnv(
        train_df,
        window_size=cfg["window_size"],
        initial_balance=cfg["initial_balance"],
        lot_size=cfg["lot_size"],
    )
    val_env = ForexTradingEnv(
        val_df,
        window_size=cfg["window_size"],
        initial_balance=cfg["initial_balance"],
        lot_size=cfg["lot_size"],
    )

    state_dim  = train_env.observation_space.shape[0]
    action_dim = train_env.action_space.n

    # -- Agent --
    agent = DQNAgent(
        state_dim      = state_dim,
        action_dim     = action_dim,
        lr             = cfg["lr"],
        gamma          = cfg["gamma"],
        epsilon_start  = cfg["epsilon_start"],
        epsilon_end    = cfg["epsilon_end"],
        epsilon_decay  = cfg["epsilon_decay"],
        batch_size     = cfg["batch_size"],
        target_update  = cfg["target_update"],
        buffer_size    = cfg["buffer_size"],
        double_dqn     = cfg["double_dqn"],
        hidden         = cfg["hidden"],
    )

    print(f"\n{'='*60}")
    print(f"  RL Trading Bot — DQN on EUR/USD 1H")
    print(f"  State dim : {state_dim}")
    print(f"  Episodes  : {cfg['n_episodes']}")
    print(f"{'='*60}\n")

    # -- Training Loop --
    history = {"train_reward": [], "val_reward": [], "val_balance": [], "epsilon": [], "loss": []}
    recent_rewards = deque(maxlen=50)
    best_val_balance = -np.inf

    for episode in range(1, cfg["n_episodes"] + 1):
        obs, _ = train_env.reset()
        done = truncated = False
        ep_reward = 0.0
        ep_loss   = []

        while not (done or truncated):
            action  = agent.select_action(obs)
            next_obs, reward, done, truncated, _ = train_env.step(action)
            agent.buffer.push(obs, action, reward, next_obs, done or truncated)
            loss = agent.learn()
            if loss > 0:
                ep_loss.append(loss)
            obs = next_obs
            ep_reward += reward

        agent.decay_epsilon()
        recent_rewards.append(ep_reward)
        avg_loss = float(np.mean(ep_loss)) if ep_loss else 0.0

        history["train_reward"].append(ep_reward)
        history["epsilon"].append(agent.epsilon)
        history["loss"].append(avg_loss)

        # -- Validation --
        if episode % cfg["log_every"] == 0:
            val_metrics = evaluate(agent, val_env)
            history["val_reward"].append(val_metrics["mean_reward"])
            history["val_balance"].append(val_metrics["mean_balance"])

            print(
                f"Ep {episode:4d}/{cfg['n_episodes']} | "
                f"TrainRew {ep_reward:+.4f} | Avg50 {np.mean(recent_rewards):+.4f} | "
                f"ValBal ${val_metrics['mean_balance']:,.0f} | "
                f"ε {agent.epsilon:.3f} | Loss {avg_loss:.5f}"
            )

            # Save best model
            if val_metrics["mean_balance"] > best_val_balance:
                best_val_balance = val_metrics["mean_balance"]
                agent.save("models/best_model.pt")
                print(f"  ★ New best val balance: ${best_val_balance:,.2f}")

        # -- Periodic checkpoint --
        if episode % cfg["save_every"] == 0:
            agent.save(f"models/checkpoint_ep{episode}.pt")

    # -- Save training history --
    with open("logs/train_history.json", "w") as f:
        json.dump(history, f, indent=2)

    _plot_training(history, cfg["log_every"])
    print("\n[Train] Done! Best val balance: ${:,.2f}".format(best_val_balance))
    return agent, test_df


# -----------------------------------------------------------------------
def _plot_training(history: dict, log_every: int):
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    fig.suptitle("DQN Training — EUR/USD 1H", fontsize=14, fontweight="bold")

    axes[0, 0].plot(history["train_reward"], alpha=0.5, color="steelblue", label="Episode Reward")
    # Rolling average
    if len(history["train_reward"]) >= 50:
        roll = pd.Series(history["train_reward"]).rolling(50).mean()
        axes[0, 0].plot(roll, color="darkblue", lw=2, label="50-ep avg")
    axes[0, 0].set_title("Training Reward"); axes[0, 0].legend()

    if history["val_balance"]:
        x_val = [i * log_every for i in range(1, len(history["val_balance"]) + 1)]
        axes[0, 1].plot(x_val, history["val_balance"], color="green", marker="o", ms=4)
    axes[0, 1].axhline(y=10_000, color="gray", linestyle="--", label="Initial Balance")
    axes[0, 1].set_title("Validation Balance ($)"); axes[0, 1].legend()

    axes[1, 0].plot(history["epsilon"], color="orange")
    axes[1, 0].set_title("Epsilon Decay")

    axes[1, 1].plot(history["loss"], alpha=0.6, color="red")
    axes[1, 1].set_title("Training Loss (Huber)")

    plt.tight_layout()
    plt.savefig("results/training_curves.png", dpi=150)
    plt.close()
    print("[Plot] Training curves saved → results/training_curves.png")


# -----------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=DEFAULT_CONFIG["n_episodes"])
    parser.add_argument("--lr",       type=float, default=DEFAULT_CONFIG["lr"])
    parser.add_argument("--batch",    type=int, default=DEFAULT_CONFIG["batch_size"])
    args = parser.parse_args()

    config_override = {
        "n_episodes": args.episodes,
        "lr":         args.lr,
        "batch_size": args.batch,
    }
    train(config_override)