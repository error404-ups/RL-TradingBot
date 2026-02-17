"""
Custom Forex Trading Environment for EUR/USD
Compatible with OpenAI Gym interface
"""

import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces


class ForexTradingEnv(gym.Env):
    """
    A model-free RL trading environment for EUR/USD hourly data.

    Actions:
        0 - Hold
        1 - Buy (Long)
        2 - Sell (Short)

    Observation Space:
        Window of OHLCV + technical indicators (normalized)

    Reward:
        Realized PnL from closing positions + shaping via unrealized PnL
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        df: pd.DataFrame,
        window_size: int = 24,         # 24 hours lookback
        initial_balance: float = 10_000.0,
        lot_size: float = 1000.0,       # micro lot
        spread_pips: float = 1.5,       # EUR/USD typical spread
        pip_value: float = 0.1,         # pip value per micro lot
        max_steps: int = None,
        render_mode: str = None,
    ):
        super().__init__()

        self.df = df.reset_index(drop=True)
        self.window_size = window_size
        self.initial_balance = initial_balance
        self.lot_size = lot_size
        self.spread = spread_pips * 0.0001  # convert pips to price
        self.pip_value = pip_value
        self.max_steps = max_steps or len(df)
        self.render_mode = render_mode

        self.n_features = self._compute_features(self.df.iloc[:window_size]).shape[1]

        # Action space: 0=Hold, 1=Buy, 2=Sell
        self.action_space = spaces.Discrete(3)

        # Observation: flattened window of features
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(window_size * self.n_features,),
            dtype=np.float32,
        )

    # ------------------------------------------------------------------
    # Feature Engineering
    # ------------------------------------------------------------------
    def _compute_features(self, window: pd.DataFrame) -> np.ndarray:
        """Compute normalized OHLCV + technical indicators for a window."""
        df = window.copy()

        close = df["Close"].values
        high  = df["High"].values
        low   = df["Low"].values
        vol   = df["Volume"].values if "Volume" in df.columns else np.ones(len(df))

        # Returns
        ret = np.diff(close, prepend=close[0]) / (close[0] + 1e-9)

        # RSI (14)
        rsi = self._rsi(close, 14)

        # MACD line
        ema12 = self._ema(close, 12)
        ema26 = self._ema(close, 26)
        macd  = (ema12 - ema26) / (close + 1e-9)

        # Bollinger Band width
        sma20  = self._sma(close, min(20, len(close)))
        std20  = self._rolling_std(close, min(20, len(close)))
        bb_width = (2 * std20) / (sma20 + 1e-9)

        # High-Low range (normalized)
        hl_range = (high - low) / (close + 1e-9)

        # Volume normalized
        vol_norm = (vol - vol.mean()) / (vol.std() + 1e-9)

        features = np.column_stack([ret, rsi, macd, bb_width, hl_range, vol_norm])
        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
        return features.astype(np.float32)

    # ------------------------------------------------------------------
    # Gym Interface
    # ------------------------------------------------------------------
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = self.window_size
        self.balance      = self.initial_balance
        self.position     = 0          # 0=flat, 1=long, -1=short
        self.entry_price  = 0.0
        self.total_reward = 0.0
        self.trade_log    = []
        self.equity_curve = [self.initial_balance]
        return self._get_obs(), {}

    def step(self, action: int):
        assert self.action_space.contains(action)

        price = self.df.loc[self.current_step, "Close"]
        reward = 0.0

        # ----- Close existing position if action conflicts -----
        if self.position == 1 and action == 2:          # long → sell signal
            reward = self._close_position(price, "long")
        elif self.position == -1 and action == 1:       # short → buy signal
            reward = self._close_position(price, "short")

        # ----- Open new position -----
        if action == 1 and self.position == 0:
            self.position    = 1
            self.entry_price = price + self.spread       # pay spread on buy
        elif action == 2 and self.position == 0:
            self.position    = -1
            self.entry_price = price - self.spread       # pay spread on sell

        # Unrealized PnL shaping (small coefficient)
        if self.position == 1:
            unrealized = (price - self.entry_price) / self.entry_price
            reward += 0.1 * unrealized
        elif self.position == -1:
            unrealized = (self.entry_price - price) / self.entry_price
            reward += 0.1 * unrealized

        self.current_step += 1
        self.total_reward += reward
        self.equity_curve.append(self.balance)

        terminated = self.current_step >= min(self.max_steps, len(self.df) - 1)
        truncated  = self.balance <= 0

        # Force close at end of episode
        if terminated and self.position != 0:
            close_reward = self._close_position(
                self.df.loc[self.current_step - 1, "Close"],
                "long" if self.position == 1 else "short",
            )
            reward += close_reward

        if self.render_mode == "human":
            self.render()

        return self._get_obs(), reward, terminated, truncated, {}

    def _close_position(self, price: float, direction: str) -> float:
        if direction == "long":
            pnl = (price - self.spread - self.entry_price) / self.entry_price
        else:
            pnl = (self.entry_price - (price + self.spread)) / self.entry_price

        dollar_pnl = pnl * self.lot_size
        self.balance += dollar_pnl
        self.position = 0
        self.entry_price = 0.0
        self.trade_log.append({"pnl": dollar_pnl, "step": self.current_step})
        return pnl  # reward is normalized PnL

    def _get_obs(self) -> np.ndarray:
        window = self.df.iloc[self.current_step - self.window_size: self.current_step]
        features = self._compute_features(window)
        return features.flatten()

    def render(self):
        price = self.df.loc[self.current_step - 1, "Close"]
        pos_str = {0: "FLAT", 1: "LONG", -1: "SHORT"}[self.position]
        print(
            f"Step {self.current_step:5d} | Price {price:.5f} | "
            f"Pos {pos_str:5s} | Balance ${self.balance:,.2f} | "
            f"Total Reward {self.total_reward:.4f}"
        )

    # ------------------------------------------------------------------
    # Technical Indicator Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _ema(series: np.ndarray, span: int) -> np.ndarray:
        alpha = 2 / (span + 1)
        result = np.zeros_like(series, dtype=float)
        result[0] = series[0]
        for i in range(1, len(series)):
            result[i] = alpha * series[i] + (1 - alpha) * result[i - 1]
        return result

    @staticmethod
    def _sma(series: np.ndarray, window: int) -> np.ndarray:
        result = np.convolve(series, np.ones(window) / window, mode="same")
        return result

    @staticmethod
    def _rolling_std(series: np.ndarray, window: int) -> np.ndarray:
        result = np.zeros_like(series, dtype=float)
        for i in range(len(series)):
            start = max(0, i - window + 1)
            result[i] = series[start: i + 1].std()
        return result

    @staticmethod
    def _rsi(series: np.ndarray, period: int = 14) -> np.ndarray:
        delta = np.diff(series, prepend=series[0])
        gain  = np.where(delta > 0, delta, 0.0)
        loss  = np.where(delta < 0, -delta, 0.0)
        avg_gain = ForexTradingEnv._ema(gain, period)
        avg_loss = ForexTradingEnv._ema(loss, period)
        rs  = avg_gain / (avg_loss + 1e-9)
        rsi = 100 - (100 / (1 + rs))
        return rsi / 100.0   # normalize to [0,1]