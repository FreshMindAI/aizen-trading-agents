"""Phase-1 ML layer (training doc "Phase 1 - Options Alpha ML Model").

Three outputs, one package:
    direction  P(future_return > tau)          - underlying dataset
    rv         future realized volatility      - underlying dataset
    option     y_option_profit / y_option_return - contract dataset

Every training row is read straight from SQLite views -> reproducible.
"""
