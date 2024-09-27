import pandas as pd
import ta

def rsi_signals(data, rsi_window, rsi_upper, rsi_lower):
    indicator_rsi = ta.momentum.RSIIndicator(close=data["Close"], window=rsi_window)
    buy_signal = indicator_rsi.rsi() < rsi_lower
    sell_signal = indicator_rsi.rsi() > rsi_upper
    return buy_signal, sell_signal

def roc_signals(data, roc_window, roc_upper, roc_lower):
    indicator_roc = ta.momentum.ROCIndicator(close=data["Close"], window=roc_window)
    buy_signal = indicator_roc.roc() > roc_lower
    sell_signal = indicator_roc.roc() < roc_upper
    return buy_signal, sell_signal

def tsi_signals(data, tsi_window_slow, tsi_window_fast, tsi_upper, tsi_lower):
    indicator_tsi = ta.momentum.TSIIndicator(close=data["Close"],
                                             window_slow=tsi_window_slow,
                                             window_fast=tsi_window_fast)
    buy_signal = indicator_tsi.tsi() > tsi_lower
    sell_signal = indicator_tsi.tsi() < tsi_upper
    return buy_signal, sell_signal

def stoch_signals(data, stoch_window, stoch_smooth_window, stoch_upper, stoch_lower):
    indicator_stoch = ta.momentum.StochasticOscillator(close=data["Close"], high=data["High"], low=data["Low"],
                                                       window=stoch_window, smooth_window=stoch_smooth_window)
    buy_signal = indicator_stoch.stoch() < stoch_lower
    sell_signal = indicator_stoch.stoch() > stoch_upper
    return buy_signal, sell_signal


