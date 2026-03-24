---
name: project_evolution_plan
description: The project is evolving to add game theory (Nash), moving average forecasting, moving average variance, and AR models into modelo_matematico/
type: project
---

The project is expanding beyond sentiment analysis. Four new mathematical modules are planned for `app/modelo_matematico/`:

1. Game Theory (Nash equilibrium, players, strategies)
2. Moving Average Forecasting (promedios móviles)
3. Moving Average Variance analysis
4. AR (Autoregressive) model

**Why:** University project evolution — integrating financial math models with the existing sentiment pipeline.
**How to apply:** New code goes in `app/modelo_matematico/`. Should integrate with existing Polars DataFrames and sentiment data from `modelo_fundamental`.
