# 🤖 RL-TradingBot — Model-Free DQN Forex Agent

A **model-free Reinforcement Learning** trading bot for **EUR/USD hourly data** using a **Double Dueling DQN** agent. Trained on 10 years of historical data with a custom OpenAI Gymnasium-compatible trading environment.

---

## 🏗️ Architecture

```
RL-TradingBot/
├── agents/
│   └── dqn_agent.py       # Double Dueling DQN + Replay Buffer
├── env/
│   └── trading_env.py     # Custom Gym environment (ForexTradingEnv)
├── data/
│   └── data_loader.py     # EUR/USD downloader, cache, train/val/test split
├── models/                # Saved checkpoints
├── logs/                  # Training history JSON
├── results/               # Plots (equity curve, training curves)
├── train.py               # Training script
├── evaluate.py            # Backtest + metrics
└── requirements.txt
```

---

## 🧠 Model-Free RL Algorithm

| Component | Choice |
|-----------|--------|
| Algorithm | **Double DQN** (model-free, off-policy) |
| Network | **Dueling DQN** (separate value + advantage streams) |
| Exploration | **ε-Greedy** with exponential decay |
| Memory | **Experience Replay Buffer** (200k transitions) |
| Optimizer | **Adam** with gradient clipping |
| Loss | **Huber (Smooth L1)** |

### Why Double Dueling DQN?
- **Model-free**: no explicit model of market dynamics — learns purely from experience
- **Double DQN**: prevents Q-value overestimation by decoupling action selection from evaluation
- **Dueling**: separates "how good is this state?" from "how good is this action?" — better for Hold-heavy markets

---

## 📊 Trading Environment

| Parameter | Value |
|-----------|-------|
| Asset | EUR/USD |
| Timeframe | 1 Hour |
| History | 10 Years (~87,000 bars) |
| Actions | Hold (0), Buy/Long (1), Sell/Short (2) |
| Lookback window | 24 hours |
| Initial balance | $10,000 |
| Spread | 1.5 pips |

### State Features (per candle × 24 bars):
- **Hourly return** (normalized)
- **RSI(14)** (normalized 0–1)
- **MACD** (normalized by price)
- **Bollinger Band width**
- **High-Low range**
- **Volume** (z-score normalized)

### Reward Function:
```
reward = realized_PnL (on close) + 0.1 × unrealized_PnL (shaping)
```

---

## 🚀 Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Train the agent
```bash
python train.py
# Or with custom parameters:
python train.py --episodes 300 --lr 3e-4 --batch 256
```

### 3. Evaluate / Backtest
```bash
python evaluate.py --model models/best_model.pt
```

---

## 📈 Data Split

| Set | Ratio | Period (approx) |
|-----|-------|-----------------|
| Train | 70% | 2014–2021 |
| Validation | 15% | 2021–2022 |
| Test | 15% | 2022–2024 |

Data is downloaded automatically via **yfinance** and cached locally as `.parquet` for fast reloads.

---

## ⚙️ Hyperparameters

| Parameter | Default |
|-----------|---------|
| Learning rate | 1e-4 |
| Gamma (discount) | 0.99 |
| ε start / end / decay | 1.0 / 0.05 / 0.9995 |
| Batch size | 128 |
| Replay buffer | 200,000 |
| Target net update | every 500 steps |
| Hidden units | 256 |
| Episodes | 200 |

---

## 📉 Output Artifacts

After training:
- `models/best_model.pt` — best checkpoint by validation balance
- `models/checkpoint_epN.pt` — periodic checkpoints
- `logs/train_history.json` — full reward/loss/epsilon logs
- `results/training_curves.png` — 4-panel training dashboard
- `results/backtest.png` — price + actions + equity + drawdown

---

## ⚠️ Disclaimer

This project is for **educational and research purposes only**.  
It does **not** constitute financial advice. Past simulated performance does not guarantee future real-world results. Forex trading involves significant risk.

---

## 📄 License

MIT License