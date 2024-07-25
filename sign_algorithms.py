#!/usr/bin/env python3.11
"""This script finds the Trade sign according to various classification algorithms, following Jukartis (2020)."""
import pandas as pd
from classifytrades import TradeClassification

# Set the maximum number of rows to display
pd.set_option('display.max_rows', 500)

class TradeAnalyzer:
    def __init__(self, trades, ask, bid):
        self.trades = trades
        self.Ask = ask
        self.Bid = bid
        self.tc = None

    def classify_trades(self, method='ds_3', freq=0):
        self.tc = TradeClassification(self.trades, Ask=self.Ask, Bid=self.Bid)
        self.tc.classify(method=method, freq=freq, reduce_precision=True)
        return self.tc.df_tr