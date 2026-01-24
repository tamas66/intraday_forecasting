## Conclusions about the data (EDA)

- Intraday price levels are **non-stationary in mean** with clear **structural regime shifts** (pre-2021 → crisis 2021–2022 → post-crisis normalization).
- First differencing reduces low-frequency persistence, but **seasonal autocorrelation remains** (notably 24h and 168h).
- Level ACF decays very slowly → **near-unit-root + deterministic seasonality**, not a simple stationary ARMA.
- PACF shows strong dependence at **lags 1–24** and **multiples of 24** → strong intraday AR structure (hour-to-hour memory).
- Weekly periodicity (168h) is consistent across ACF/PACF and decomposition → strong **weekly seasonality**.
- Rolling mean is time-varying → **mean non-stationarity** confirmed.
- Rolling std/variance is time-varying with clusters/spikes → strong **conditional heteroskedasticity** and **variance non-stationarity**.
- Returns/log-returns are closer to stationary but remain **heavy-tailed**.
- Distributions of prices, returns, log-returns, and DA–ID spread are **spiky and fat-tailed**, with extreme outliers dominating tail behavior.
- QQ plots show strong deviation from Gaussian assumptions (fat tails + asymmetry), especially for levels and spread.
- Volatility clustering is strong and persistent; variance exhibits **regime-like behavior** rather than smooth evolution.
- DA and ID prices show **very strong co-movement**, consistent with **near-cointegration** (not just correlation).
- Demand (actual and DA forecast) has a **consistent positive association** with ID price (secondary to DA price).
- Wind outturn is **negatively associated** with prices and is stronger than solar in aggregate magnitude.
- Solar effect is **highly diurnal** (time-of-day locked) and shows price suppression during midday hours.
- Cross-correlations show structured periodic patterns:
  - DA price ↔ ID price: strong, centered near 0-lag (DA information dominates).
  - Demand ↔ price: clear daily/weekly structure.
  - Wind/solar ↔ price: weaker but systematic, with diurnal components.
- Scatter plots show strong **heteroskedastic fan-out** (variance increases with level / system stress).
- Weekday vs weekend differs in distribution and outliers → **calendar effects** in both mean and variance.
- Seasonal decomposition indicates stable weekly seasonal component but unstable trend + residual variance.
- Conditional mean and variance peak together around **15–18h**, implying **price–risk co-movement** in peak hours.
- Solar and price show **opposing intraday patterns** (merit-order effect); wind is smoother/less sharply timed.
- DA–ID spread is centered near zero but dominated by rare extreme deviations → **tail-risk driven spread dynamics**.

## Effects of conclusions on further modelling steps

- Do not treat price levels as stationary without explicit deterministic seasonality; prefer **SARIMAX / ARX with seasonal terms** over plain ARIMA.
- Include deterministic **hour-of-day and day-of-week** effects (Fourier terms or dummies) in the mean equation.
- Keep **24h and 168h lag structure** in the autoregressive component; consider multiple seasonal components.
- Model DA price as a core driver:
  - As an exogenous regressor in ARX/SARIMAX, and/or
  - As an **error-correction / cointegration-style** relationship (spread-based or ECM-like structure).
- Demand and renewables should be modeled with **hour-dependent effects** (interactions with hour-of-day or separate-by-hour models).
- Use explicit volatility models:
  - **GARCH-X** (with exogenous drivers) and/or
  - **Regime-switching variance** (MS-ARX / MS-GARCH) to capture variance regimes.
- Avoid Gaussian errors; use **Student-t / skew-t** innovations to handle fat tails.
- Treat spikes as part of the data-generating process (robust likelihood / heavy-tail), not as noise to clip away.
- Consider modelling targets separately:
  - **Levels** for mean forecasting,
  - **Returns / log-returns** for volatility,
  - **DA–ID spread** for relative value and reduced mean non-stationarity.
- Use evaluation metrics sensitive to tails (e.g., quantile loss, exceedance hit rate) in addition to RMSE/MAE.
- Ensure training/validation splits respect regime changes (rolling/expanding windows); regime instability implies static parameters may drift.
