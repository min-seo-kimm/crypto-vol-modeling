import time
import warnings
import numpy as np
import pandas as pd
from models.garch_model import GARCHModel
from models.egarch_model import EGARCHModel
from models.gjr_garch_model import GJRGARCHModel
from models.rt_garch_model import RTGARCHModel
from models.cvx_garch_model import CVXGARCHModel

warnings.filterwarnings('ignore')

# Adjustable parameters
asset = 'BTC'
horizon = 1
window = 1000

# Load data
residuals_df = pd.read_csv('data/daily_residuals.csv', index_col=0, parse_dates=True)
realized_vars_df = pd.read_csv('data/daily_realized_vars.csv', index_col=0, parse_dates=True)
cvx_df = pd.read_csv('data/daily_cvx.csv', index_col=0, parse_dates=True)
common_idx = realized_vars_df.index.intersection(cvx_df.index)
residuals = residuals_df[asset].loc[common_idx].to_numpy()
realized_vars = realized_vars_df[asset].loc[common_idx].to_numpy()
cvx = cvx_df['CVX'].loc[common_idx].to_numpy()

# Train/test split
split_idx = int(len(residuals) * 0.8)

residuals_train = residuals[:split_idx]
residuals_test = residuals[split_idx:]
realized_vars_train = realized_vars[:split_idx]
realized_vars_test = realized_vars[split_idx:]
cvx_train = cvx[:split_idx]
cvx_test = cvx[split_idx:]

# Define model mappings
model_classes = {
    'GARCH': GARCHModel,
    'EGARCH': EGARCHModel,
    'GJR-GARCH': GJRGARCHModel,
    'CVX-GARCH': CVXGARCHModel,
    'RT-GARCH': RTGARCHModel
}

# Result storage
results = []

# Run all combinations
for model_type in ['GARCH', 'CVX-GARCH', 'RT-GARCH']:
    for dist in ['normal', 'ged']:
        print(f'Running {model_type} with {dist} distribution')
        VolatilityModel = model_classes[model_type]

        if model_type == 'CVX-GARCH':
            model = VolatilityModel(residuals_train, cvx_train, dist=dist)
        else:
            model = VolatilityModel(residuals_train, dist=dist)

        # Fit in-sample
        start = time.time()
        try:
            params = model.fit()
            llk = model.llk
            aic = model.aic
        except Exception as e:
            print(f'Error: {e}')
            continue
        train_time = time.time() - start

        # Forecast out-of-sample
        forecasted_vars = []
        for t in range(len(residuals_test)):
            res_window = residuals[split_idx + t - window:split_idx + t]
            
            if model_type == 'CVX-GARCH':
                cvx_window = cvx[split_idx + t - window:split_idx + t]
                model = VolatilityModel(res_window, cvx_window, dist=dist)
            else:
                model = VolatilityModel(res_window, dist=dist)
            
            model.fit()
            forecasts = model.forecast(horizon=horizon)
            forecasted_vars.append(forecasts[horizon - 1])

        if horizon > 1:
            forecasted_vars = forecasted_vars[:-horizon + 1]
            realized_vars_adjusted = realized_vars_test[horizon - 1:]
        else:
            realized_vars_adjusted = realized_vars_test

        forecasted_vars = np.array(forecasted_vars)

        # Evaluate performance
        mse = np.mean((forecasted_vars - realized_vars_adjusted)**2)
        mae = np.mean(np.abs(forecasted_vars - realized_vars_adjusted))
        hmse = np.mean((1 - forecasted_vars / realized_vars_adjusted)**2)

        # Record results
        result_row = {
            'Model': model_type,
            'Distribution': dist,
            'Omega': params[0],
            'Alpha': params[1],
            'Beta': params[2],
            'Log_Likelihood': llk,
            'AIC': aic,
            'Train_Time': train_time,
            'MSE': mse,
            'MAE': mae,
            'HMSE': hmse
        }

        if model_type in ['EGARCH', 'GJR-GARCH']:
            result_row['Gamma'] = params[3]
        else:
            result_row['Gamma'] = np.nan

        if model_type in ['RT-GARCH', 'CVX-GARCH']:
            result_row['Phi'] = params[3]
        else:
            result_row['Phi'] = np.nan

        if dist == 't':
            result_row['Nu'] = params[-1]
        else:
            result_row['Nu'] = np.nan

        results.append(result_row)

# Save to CSV
results_df = pd.DataFrame(results)
results_df.to_csv('results.csv', index=False)
print('Results saved to results.csv')
