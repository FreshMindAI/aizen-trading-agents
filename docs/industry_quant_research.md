# Industry Quant Trading — Architecture & Mathematical Formulas

Research compiled 2026-08-29 for the Aizen Trading multi-agent pipeline. The
goal is to understand how world-class quant systems are built and identify
the mathematical building blocks we should ensure our ML / GNN / multi-agent
stack exploits.

## 1. Industry architectures

### Renaissance Technologies (Medallion)
- Signal pool: thousands of weak, statistically independent short-horizon
  alphas. Each alpha has a small positive expected return with low pairwise
  correlation (15.9% in WorldQuant's 101 Alphas paper).
- Combination rule: equal-weight or shrinkage-weighted ensemble of all
  alphas. No single alpha matters; the portfolio does.
- Risk: kill-switches on regime change, position-size limits per signal,
  and a global risk overlay that re-allocates across alphas each day.
- Data: alternative data (news, satellite, weather, credit card, shipping)
  mixed with L1/L2 market data, all timestamped to the microsecond.

### Two Sigma
- Layered signal factory: research -> candidate -> paper -> live with a
  strict promotion gate (out-of-sample Sharpe > X, turnover < Y, drawdown
  < Z).
- Heavy use of ML (gradient boosting, deep nets on alternative data,
  RL for execution) plus classical stat-arb signals.
- Execution: proprietary smart-order router that minimizes market impact
  using Almgren-Chriss / VWAP / Implementation-Shortfall models.

### Citadel / DE Shaw / Jump
- Multi-strategy shops: each strategy (stat-arb, fundamental, macro, credit)
  has its own signal-generation team. Cross-strategy capital allocation is
  itself a portfolio optimization problem.
- Risk parity and risk budgeting at the book level; each strategy has a
  vol target (e.g. 10% annualized) and the firm re-allocates to keep total
  book vol at target.

### QuantConnect (open framework)
- Modular algorithm framework:
  1. **Universe Selection** — filter the tradable universe.
  2. **Alpha** — generate trading insights (momentum, mean-reversion,
     sentiment, options-greeks, ML).
  3. **Portfolio Construction** — convert insights to targets
     (mean-variance, risk-parity, equal-weight, Black-Litterman).
  4. **Risk Management** — per-position and portfolio-level constraints.
  5. **Execution** — order placement (VWAP, TWAP, market-impact models).
- Data sources: US equity, options, futures, crypto, forex; news +
  fundamentals + insider/SEC data via Tiingo, Brain, Quiver.
- Sentiment models: Hugging Face FinBERT, custom transformer models.
- ML libraries: scikit-learn, XGBoost, Keras, PyTorch, TensorFlow.

## 2. Common mathematical building blocks

### 2.1 Mean reversion (OU process, pairs trading)
- Continuous-time SDE:
  `dX_t = -theta * (X_t - mu) dt + sigma * dW_t`
- Discrete AR(1):
  `X_{t+1} = X_t * exp(-theta * dt) + mu * (1 - exp(-theta * dt)) + epsilon_t`
- Half-life of mean reversion:
  `t_1/2 = ln(2) / theta`
- Pairs trading: pick a cointegrating beta, compute the spread
  `z_t = y_t - beta * x_t`, trade when `|z_t - mu| > k * sigma/sqrt(2*theta)`.

### 2.2 Momentum indicators
- EMA: `EMA_t = alpha * P_t + (1 - alpha) * EMA_{t-1}` with
  `alpha = 2 / (n + 1)`.
- RSI: `RSI = 100 - 100 / (1 + avg_gain / avg_loss)`.
- MACD: `MACD = EMA_12 - EMA_26`; signal = `EMA_9(MACD)`.

### 2.3 Volatility estimation
- GARCH(1,1):
  `sigma_t^2 = omega + alpha * epsilon_{t-1}^2 + beta * sigma_{t-1}^2`
  with `alpha + beta < 1` for stationarity.
- EWMA (RiskMetrics): `sigma_t^2 = lambda * sigma_{t-1}^2 + (1 - lambda) * r_{t-1}^2`,
  with `lambda = 0.94` for daily data.
- Parkinson range estimator:
  `sigma_hat = sqrt(1 / (4 * ln 2) * mean((ln(H_i / L_i))^2))`
- Yang-Zhang: combines overnight, open-to-open, and Rogers-Satchell
  components to give a single efficient range-based estimator.

### 2.4 Risk metrics
- Sharpe: `S = (E[R_p] - R_f) / sigma_p`.
- Sortino: replaces `sigma_p` with downside deviation only.
- VaR (parametric): `VaR_alpha = -(mu - z_alpha * sigma)`.
- CVaR (expected shortfall): `CVaR_alpha = -E[R | R <= -VaR_alpha]`.
- Kelly fraction: `f* = (p*b - q) / b = E[R] / sigma^2` for binary outcomes;
  for continuous returns `f* = E[R] / sigma^2`. Use half-Kelly in practice.

### 2.5 Option pricing (Black-Scholes-Merton)
- Geometric Brownian motion: `dS = mu * S dt + sigma * S dW`.
- PDE: `V_t + 0.5*sigma^2*S^2*V_SS + r*S*V_S - r*V = 0` with payoff
  `V(T,S) = K(S_T)`.
- Closed-form European call:
  `C = S * N(d_1) - K * exp(-r*(T-t)) * N(d_2)`
  where
  `d_1 = (ln(S/K) + (r + 0.5*sigma^2)*(T-t)) / (sigma*sqrt(T-t))`
  and `d_2 = d_1 - sigma*sqrt(T-t)`.
- Greeks: `Delta = N(d_1)`, `Gamma = N'(d_1) / (S*sigma*sqrt(T-t))`,
  `Vega = S*N'(d_1)*sqrt(T-t)`, `Theta = -S*N'(d_1)*sigma/(2*sqrt(T-t))
  - r*K*exp(-r*(T-t))*N(d_2)`, `Rho = K*(T-t)*exp(-r*(T-t))*N(d_2)`.

### 2.6 Portfolio construction
- Markowitz mean-variance: minimize `w^T * Sigma * w` subject to
  `w^T * mu = target_return` and `sum(w) = 1`.
- Capital Market Line:
  `R_P = R_f + (R_M - R_f) * sigma_P / sigma_M`.
- Black-Litterman: combine market-implied prior `Pi` with views `P` and
  `Omega` via
  `E[R] = [(tau*Sigma)^-1 + P^T * Omega^-1 * P]^-1 *
          [(tau*Sigma)^-1 * Pi + P^T * Omega^-1 * Q]`.
- Risk parity: allocate so each position contributes equal marginal risk;
  `w_i * (Sigma * w)_i = w_j * (Sigma * w)_j` for all i,j.

### 2.7 WorldQuant 101 alphas (Kakushadze 2015)
- Categories: price/volume momentum, mean reversion, volatility, volume,
  correlation, sector/industry, fundamental.
- Average holding period 0.6-6.4 days.
- Average pairwise correlation 15.9% — most are statistically independent,
  which is why the ensemble works.
- Returns are strongly correlated with volatility but NOT turnover.
- Common operator families:
  - Time-series: `ts_rank(x, d)`, `ts_delta(x, d)`, `ts_std(x, d)`,
    `ts_mean(x, d)`, `ts_decay_linear(x, d)`, `ts_corr(x, y, d)`.
  - Cross-sectional: `rank(x)`, `scale(x)`, `normalize(x)`, `IndNeutralize(x)`.
  - Logical: `If(condition, a, b)`, `Sign(x)`.

## 3. How this maps to Aizen

### Where we already align
- **3 ML heads** = WorldQuant "many weak signals" principle. The fact that
  each model has a different target (direction, realized vol, option
  opportunity) keeps the pairwise correlation low.
- **Multi-agent pipeline** = layered signal factory. The supervisor's
  `direction_prob_min = 0.55` is exactly the "promotion gate" pattern.
- **GNN on option graph** = the structural/relational signal that
  Renaissance-style systems extract from correlation matrices and
  inter-stock networks. News edges added = the alternative-data layer
  Two Sigma and Citadel rely on.
- **15-min bar grid** = L1 market data, same as the industry standard for
  intraday.
- **Backtest with as_of + v_labels (leak-safe)** = point-in-time discipline
  is the same constraint institutional quants enforce for their
  paper-trading and promotion gates.
- **v_features / v_labels views** = the same view-materialization pattern
  used to share feature pipelines across research and production.

### Where we may be under-using the math
- **Mean reversion (OU)**: not currently an explicit feature. A
  `mean_reversion_half_life` feature per symbol would let the
  direction model weight mean-reverting symbols differently from
  trending ones. Cheap to add: fit AR(1) on rolling 60-bar log-returns.
- **GARCH volatility**: we have `rv` (realized vol) but no conditional
  vol forecast. A GARCH(1,1) residual or 1-step vol forecast would
  add information beyond realized vol.
- **Cross-sectional rank/normalize**: features are absolute. WorldQuant
  uses `rank()`, `scale()`, `IndNeutralize()` to make features
  cross-sectionally comparable, which lets the model compare NVDA to
  AAPL cleanly. Adding `cs_rank` and `cs_scale` features to the ML
  pipeline would mirror the industry best practice.
- **Time-series operators**: `ts_rank`, `ts_delta`, `ts_corr` over
  lookback windows (5, 20, 60 bars) are standard. We have rolling stats
  but not rank-based ones.
- **Kelly / half-Kelly position sizing**: the supervisor's sizing
  logic could use `Kelly_fraction = edge / variance`, capped at a
  configurable max (e.g. 0.25 for half-Kelly with a 50% haircut on
  the edge estimate).
- **Risk parity at book level**: when the orchestrator produces multiple
  trade intents, allocate capital so each contributes equal vol, not
  equal dollars.
- **Black-Litterman for symbol weighting**: combine the market-implied
  "expected" score (e.g. 0.5 probability) with the model's score
  via a Bayesian shrinkage to dampen overconfidence in small samples.

### Where we may be over-reaching
- **Single-direction p(model will be right)**: the supervisor's gate
  `direction_prob > 0.55` is too narrow for short horizons (1-bar
  reversal noise can flip the sign even when the expected value is
  positive). The Sharpe of the strategy is `E[R] / sigma_R`; the
  decision threshold should be on expected edge, not on the
  probability of being right. Industry standard: trade when
  `edge / cost > 1` where `edge = E[R] - costs` and `cost` includes
  spread, slippage, and funding.
- **Decisions per 15-min bar**: a high-frequency cadence can eat
  transaction costs. Two Sigma and Citadel trade less often than the
  signal is available, specifically to manage turnover. Our backtest
  should report turnover-adjusted Sharpe to confirm this is not a
  silent drag.

## 4. What "great" looks like vs. what we have

| Dimension | Great system | Aizen today |
|---|---|---|
| Signal count | 100s-1000s of weak alphas | 3 ML + 1 GNN |
| Pairwise corr | 15% | Unknown (likely high between direction & option) |
| Data variety | L1/L2 + alt data + news + fundamentals | L1 + news (just added) + options |
| Combination | Equal-weight or shrinkage ensemble | Supervisor gate on 1 signal |
| Risk | Per-signal vol target + global vol target | Per-position only |
| Position sizing | Kelly / risk-parity | Probably equal-dollars |
| Promotion gate | Out-of-sample Sharpe + drawdown + turnover | Single `direction_prob > 0.55` |
| Backtest | Walk-forward + cross-validated | Walk-forward (US3), point-in-time (US4) |
| Leakage guard | Snapshot discipline + PIT joins | v_labels view + as_of propagation |

The biggest single upgrade is **shrinking the supervisor's gate
threshold and adding an expected-edge metric** so we trade when
`edge / cost > 1`, not when a noisy probability crosses 55%. This is
the same fix that turns a Sharpe-0.5 toy into a Sharpe-1.5 production
system.

## 5. References
- Kakushadze, Z. "101 Formulaic Alphas." arXiv:1601.00991, 2015.
- WorldQuant BRAIN tutorial. formulaic-alphas. 2014.
- Lopez de Prado, M. "Advances in Financial Machine Learning." Wiley, 2018.
- Chan, E. "Algorithmic Trading." Wiley, 2013.
- Almgren, R. and Chriss, N. "Optimal Execution of Portfolio Transactions."
  J. Risk, 2000.
- Black, F. and Litterman, R. "Asset Allocation: Combining Investor Views
  with Market Equilibrium." Goldman Sachs Fixed Income Research, 1990.
- QuantConnect Algorithm Framework documentation.
