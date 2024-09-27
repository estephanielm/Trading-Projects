import pandas as pd
import optuna
from itertools import combinations, chain
from multiprocessing import Pool
import ta

# Funciones de señales (rsi_signals, roc_signals, tsi_signals, stoch_signals) aquí...

# Clase Position y funciones de optimización aquí...

def powerset(s):
    """Genera todas las combinaciones posibles de una lista"""
    return chain.from_iterable(combinations(s, r) for r in range(1, len(s) + 1))

def backtest_strategy(data, strategy, params):
    """Realiza el backtest para una estrategia dada y sus parámetros"""
    portfolio_value = 1_000_000  # Capital inicial
    active_operations = []
    com = 0.0125  # Comisiones

    # Generar señales de compra y venta
    buy_signals, sell_signals = generate_signals(data, strategy, params)

    for i, row in data.iterrows():
        # Cierre de operaciones activas
        active_operations = close_active_operations(row, active_operations, portfolio_value, com)

        # Verificar si tenemos suficiente capital
        if portfolio_value < (row.Close * (1 + com)):
            continue

        # Aplicar señales de compra
        if buy_signals.loc[i].any():
            active_operations.append({
                "bought": row.Close,
                "n_shares": params['n_shares'],
                "stop_loss": row.Close * params['stop_loss'],
                "take_profit": row.Close * params['take_profit']
            })
            portfolio_value -= row.Close * (1 + com) * params['n_shares']

        # Aplicar señales de venta
        if sell_signals.loc[i].any():
            active_operations = close_active_operations(row, active_operations, portfolio_value, com)

    return portfolio_value

def generate_signals(data, strategy, params):
    """Genera señales de compra y venta según la estrategia y parámetros"""
    buy_signals = pd.DataFrame(index=data.index)
    sell_signals = pd.DataFrame(index=data.index)

    # Lógica para cada tipo de señal
    if "rsi" in strategy:
        rsi_buy, rsi_sell = rsi_signals(data, **params['rsi'])
        buy_signals["rsi"] = rsi_buy
        sell_signals["rsi"] = rsi_sell

    if "roc" in strategy:
        roc_buy, roc_sell = roc_signals(data, **params['roc'])
        buy_signals["roc"] = roc_buy
        sell_signals["roc"] = roc_sell

    if "tsi" in strategy:
        tsi_buy, tsi_sell = tsi_signals(data, **params['tsi'])
        buy_signals["tsi"] = tsi_buy
        sell_signals["tsi"] = tsi_sell

    if "stoch" in strategy:
        stoch_buy, stoch_sell = stoch_signals(data, **params['stoch'])
        buy_signals["stoch"] = stoch_buy
        sell_signals["stoch"] = sell_signals

    return buy_signals, sell_signals

def close_active_operations(row, active_operations, portfolio_value, com):
    """Cierra operaciones activas según las condiciones de stop loss y take profit"""
    new_active_operations = []
    for operation in active_operations:
        if row.Close < operation["stop_loss"] or row.Close > operation["take_profit"]:
            portfolio_value += (row.Close * operation["n_shares"]) * (1 - com)
        else:
            new_active_operations.append(operation)
    return new_active_operations

def optimize_file(file_path: str):
    data = pd.read_csv(file_path).dropna()
    strategies = list(powerset(["rsi", "roc", "tsi", "stoch"]))
    best_strategy = None
    best_value = -1
    best_params = None

    for strategy in strategies:
        # Crear un estudio de Optuna para optimizar la estrategia
        study = optuna.create_study(direction="maximize")
        study.optimize(lambda trial: optimize(trial, strategy, data), n_trials=50)

        final_value = study.best_value
        if final_value > best_value:
            best_value = final_value
            best_strategy = strategy
            best_params = study.best_params

    print(f"Mejor estrategia: {best_strategy}, Valor: {best_value}, Parámetros: {best_params}")
    return {"file": file_path, "strategy": best_strategy, "value": best_value, "params": best_params}

if __name__ == '__main__':
    with Pool() as p:
        res = p.map(optimize_file, ["../Project 002/aapl_5m_train.csv"])
        print(res)
