# MachineLearningWeatherForecast

A short student report on our weather forecast project. We predict daily average temperature and dewpoint, then combine them to estimate the frost point for the coming year.

## Files in this folder
- `climate_data.csv`: daily weather history (temperature, humidity/pressure, wind, rain, dates).
- `Temperature_Pred.ipynb`: feature analysis and temperature forecasting.
- `Dewpoint_Pred.ipynb`: dewpoint feature study and forecasting.
- `FrostPoint_Pred.ipynb`: uses the two previous forecasts to compute the frost point.
- `ML_Weather_Forecast_Report.pdf`: slide-style summary of the study.

## What we are predicting
We only need two signals to get frost point: temperature and dewpoint. The frost point formula used later is:

```math
T_f = T_d + \frac{2671.02}{\frac{2954.61}{T} + 2.193665 \ln(T) - 13.3448} - T
```

## Data prep (shared across notebooks)
- Converted date columns to proper datetime, set the date as index, and filled missing days before splitting into train/test.
- Checked for missing values; none in the raw file, but added synthetic rows for absent dates and used `fillna` afterward.
- Feature groups: temperature (avg/max/min/heat index/dewpoint), humidity and pressure, wind, rain, and calendar (month/day-of-year).
- Engineering: seasonal encodings for day-of-year (`day_sin`, `day_cos`), differences like `diff_temperature` and `diff_humidity`, rainfall-per-day to avoid cumulative leakage, and scaling of inputs (targets left unscaled for SARIMAX).
- Feature pruning: kept variables with |correlation| > 0.4 but < 0.9 to avoid redundancy; dropped low-signal or duplicate features (for example rainfall, wind direction, and highly collinear pressure/humidity pairs).

## Temperature modeling (`Temperature_Pred.ipynb`)
- Final features: `Average dewpoint`, `Average humidity`, `Average windspeed`, `Average barometer`, `Dayofyear` (cyclic), `diff_temperature`, `diff_humidity`.
- Tried SVR, LightGBM, XGBoost, SARIMAX, plus baselines. Tree/boosting models scored well but overfit.
- Chosen model: SARIMAX for stability and generalization (Train RMSE 2.14 vs Test RMSE 2.51; Train MAE 1.60 vs Test MAE 1.82).

## Dewpoint modeling (`Dewpoint_Pred.ipynb`)
- Strong drivers: temperature, humidity, windspeed, barometer; gustspeed kept for its inverse seasonal link; rain features discarded.
- Time series is stationary after differencing; tested Prophet, SARIMA/ARIMA, VAR (dropped), and smoothing approaches.
- Selected ARIMA-style approach; typical error around MAPE ~ 0.36, acceptable given limited data and noise.

## Frost point computation (`FrostPoint_Pred.ipynb`)
- Reuse the two trained models to forecast next year's temperature and dewpoint.
- Apply the frost point equation day by day to get `Frost point (deg C)` and mark days where forecasted temperature is below it.
- Result: only a few frost-risk periods next year, with two short stretches worth monitoring.

## How to run it yourself
1) Install deps (Python 3.x): `pandas`, `numpy`, `scikit-learn`, `statsmodels`, `matplotlib`, `seaborn`, `lightgbm`, `xgboost`, (optional) `prophet`.
2) Place `climate_data.csv` in the same folder (already here).
3) Launch Jupyter: `jupyter notebook` and open each notebook in order: Temperature -> Dewpoint -> FrostPoint.
4) Run all cells; plots show feature selection, diagnostics (ACF/PACF), and forecast curves.

## Notes and lessons
- Seasonality dominates: day-of-year encoding and dewpoint are the most informative signals for temperature.
- Overfitting is easy; prefer SARIMAX/ARIMA for stability when future exogenous features are unknown.
- Rain-related features are mostly noise for this target; cumulative rain can leak time, so avoid it.
- Filling missing dates before modeling keeps time series methods happy and avoids calendar gaps.
