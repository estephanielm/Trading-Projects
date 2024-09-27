import pandas as pd
import optuna
from itertools import combinations, chain
from signals import stoch_signals, tsi_signals, roc_signals, rsi_signals

def powerset(s):
    return chain.from_iterable(combinations(s, r) for r in range(1, len(s) + 1))

def optimize(trial, strategy, data):
    cash = 1_000_000
    com = 1.25 / 100
    n_shares = trial.suggest_int("n_shares", 5, 200)
    stop_loss = trial.suggest_float("stop_loss", 0.00250, 0.05)
    take_profit = trial.suggest_float("take_profit", 0.00250, 0.05)

    buy_signals = pd.DataFrame()
    sell_signals = pd.DataFrame()

    if "rsi" in strategy:
        rsi_params = {
            "rsi_window": trial.suggest_int("rsi_window", 5, 100),
            "rsi_upper": trial.suggest_float("rsi_upper", 65, 95),
            "rsi_lower": trial.suggest_float("rsi_lower", 5, 35)
        }
        rsi_buy, rsi_sell = rsi_signals(data, **rsi_params)
        buy_signals["rsi"] = rsi_buy
        sell_signals["rsi"] = rsi_sell

    if "roc" in strategy:
        roc_params = {
            "roc_window": trial.suggest_int("roc_window", 5, 100),
            "roc_upper": trial.suggest_float("roc_upper", 0.8, 1.5),
            "roc_lower": trial.suggest_float("roc_lower", -2, -1)
        }
        roc_buy, roc_sell = roc_signals(data, **roc_params)
        buy_signals["roc"] = roc_buy
        sell_signals["roc"] = roc_sell

    if "tsi" in strategy:
        tsi_params = {
            "tsi_window_slow": trial.suggest_int("tsi_window_slow", 5, 20),
            "tsi_window_fast": trial.suggest_int("tsi_window_fast", 20, 40),
            "tsi_upper": trial.suggest_float("tsi_upper", 25, 45),
            "tsi_lower": trial.suggest_float("tsi_lower", -40, -20)
        }
        tsi_buy, tsi_sell = tsi_signals(data, **tsi_params)
        buy_signals["tsi"] = tsi_buy
        sell_signals["tsi"] = tsi_sell

    if "stoch" in strategy:
        stoch_params = {
            "stoch_window": trial.suggest_int("stoch_window", 5, 21),
            "stoch_smooth_window": trial.suggest_int("stoch_smooth_window", 3, 10),
            "stoch_upper": trial.suggest_float("stoch_upper", 70, 90),
            "stoch_lower": trial.suggest_float("stoch_lower", 10, 30)
        }
        stoch_buy, stoch_sell = stoch_signals(data, **stoch_params)
        buy_signals["stoch"] = stoch_buy
        sell_signals["stoch"] = stoch_sell

    # Backtesting logic
    for i, row in data.iterrows():
        # Close active operations logic
        if row.Close < stop_loss or row.Close > take_profit:
            cash += (row.Close * n_shares) * (1 - com)

        # Check if we have enough cash
        if cash < (row.Close * (1 + com)):
            continue

        # Apply buy signals
        if buy_signals.loc[i].any():
            cash -= row.Close * (1 + com) * n_shares

        # Apply sell signals
        if sell_signals.loc[i].any():
            cash += (row.Close * n_shares) * (1 - com)

    return cash

def optimize_file(file_path: str):
    data = pd.read_csv(file_path).dropna()
    strategies = list(powerset(["rsi", "roc", "tsi", "stoch"]))
    best_strategy = None
    best_value = -1
    best_params = None

    for strat in strategies:
        study = optuna.create_study(direction="maximize")
        study.optimize(lambda x: optimize(x, strat, data), n_trials=50)
        value = study.best_value
        if value > best_value:
            best_value = value
            best_strategy = strat
            best_params = study.best_params

    return {
        "file": file_path,
        "strategy": best_strategy,
        "value": best_value,
        "params": best_params
    }


