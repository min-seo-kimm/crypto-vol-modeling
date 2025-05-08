"""
Main script to run different volatility models.
"""

import time 
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from models.garch_model import GARCHModel
from models.egarch_model import EGARCHModel
from models.gjr_garch_model import GJRGARCHModel
from models.rt_garch_model import RTGARCHModel
from models.cvx_garch_model import CVXGARCHModel

warnings.filterwarnings('ignore')


# === Adjustable parameters ===
model_type = 'RT-GARCH' 
asset = 'BTC' 
dist = 'ged'
horizon = 1 
window = 1000 


# === Load residuals and realized variances ===
residuals_df = pd.read_csv('data/daily_residuals.csv', index_col=0, parse_dates=True)
realized_vars_df = pd.read_csv('data/daily_realized_vars.csv', index_col=0, parse_dates=True)
cvx_df = pd.read_csv('data/daily_cvx.csv', index_col=0, parse_dates=True)

common_idx = realized_vars_df.index.intersection(cvx_df.index)
residuals = residuals_df[asset].loc[common_idx].to_numpy()
realized_vars = realized_vars_df[asset].loc[common_idx].to_numpy()
cvx = cvx_df['CVX'].loc[common_idx].to_numpy()


# === Split data into train/test set ===
split_idx = int(len(residuals) * 0.8)

residuals_train = residuals[:split_idx]
residuals_test = residuals[split_idx:]
realized_vars_train = realized_vars[:split_idx]
realized_vars_test = realized_vars[split_idx:]
cvx_train = cvx[:split_idx]
cvx_test = cvx[split_idx:]


# === Define model mappings ===
model_classes = {
    'GARCH': GARCHModel,
    'EGARCH': EGARCHModel,
    'GJR-GARCH': GJRGARCHModel,
    'CVX-GARCH': CVXGARCHModel,
    'RT-GARCH': RTGARCHModel
}

VolatilityModel = model_classes[model_type]


# === Initialize and fit volatility model ===
print(f'Model Type: {model_type}')
print(f'Asset: {asset}')
print(f'Distribution: {dist}')

if model_type == 'CVX-GARCH':
    model = VolatilityModel(residuals_train, cvx_train, dist=dist)
else:
    model = VolatilityModel(residuals_train, dist=dist)

start = time.time()
params = model.fit()
end = time.time()

print(f'Train Time: {end - start:.2f} seconds')

print('\nEstimated Parameters:')
print(f'Omega = {params[0]:.6f}')
print(f'Alpha = {params[1]:.6f}')
print(f'Beta = {params[2]:.6f}')

if model_type in ['EGARCH', 'GJR-GARCH']:
    print(f'Gamma = {params[3]:6f}')
if model_type in ['RT-GARCH', 'CVX-GARCH']:
    print(f'Phi = {params[3]:6f}')
if dist == 't':
    print(f'Nu: {params[-1]:.6f}')

print('\nIn-Sample Model Fit:')
print(f'Log-Likelihood = {model.llk:.2f}')
print(f'AIC = {model.aic:.2f}')


# === Plot in-sample conditional variance ===
plt.figure(figsize=(10, 6))
plt.plot(model.conditional_variance, label='Fitted', linewidth=1)
plt.plot(realized_vars_train, label='Realized', linewidth=1)
plt.title(f'{model_type}(1,1) In-Sample Analysis | Distribution = {dist}')
plt.xlabel('Time')
plt.ylabel('Variance')
plt.ylim(-10, 250)
plt.legend(loc='upper right')
plt.tight_layout()
plt.show()


# === Forecast future variance ===
forecasted_vars = []
gammas = []

for t in range(len(residuals_test)):
    res_window = residuals[split_idx + t - window:split_idx + t]

    if model_type == 'CVX-GARCH':
        cvx_window = cvx[split_idx + t - window:split_idx + t]
        model = VolatilityModel(res_window, cvx_window, dist=dist)
    else:
        model = VolatilityModel(res_window, dist=dist)

    params = model.fit()
    forecasts = model.forecast(horizon=horizon)
    forecasted_vars.append(forecasts[horizon - 1])

    if model_type in ['EGARCH', 'GJR-GARCH']:
        gammas.append(params[3])

if horizon > 1:
    forecasted_vars = forecasted_vars[:-horizon + 1]
    realized_vars_adjusted = realized_vars_test[horizon - 1:]
else:
    realized_vars_adjusted = realized_vars_test


# === Plot out-of-sample predicted variance ===
plt.figure(figsize=(10, 6))
plt.plot(forecasted_vars, label=f'Predicted ({horizon}-Day)', linewidth=1)
plt.plot(realized_vars_test, label='Realized', linewidth=1)
plt.title(f'{model_type}(1,1) Out-of-Sample Analysis | Distribution = {dist}')
plt.xlabel('Time')
plt.ylabel('Variance')
plt.ylim(0, 35)
plt.legend(loc='upper right')
plt.tight_layout()
plt.show()


# === Evaluate out-of-sample performance ===
mse = np.mean((forecasted_vars - realized_vars_adjusted)**2)
mae = np.mean(np.abs(forecasted_vars - realized_vars_adjusted))
hmse = np.mean(((1 - forecasted_vars / realized_vars_adjusted)**2))

print('\nOut-of-Sample Performance:')
print(f'MSE = {mse:.4f}')
print(f'MAE = {mae:.4f}')
print(f'HMSE = {hmse:.4f}')


# === Check for leverage effect ===
if model_type in ['EGARCH', 'GJR-GARCH']:
    avg_gamma = np.mean(gammas)
    print('\nLeverage Effect:')
    print(f'Average Gamma = {avg_gamma:.6f}')
