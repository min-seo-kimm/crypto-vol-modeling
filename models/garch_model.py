"""
Implementation of the GARCH(1,1) model.

Model:
    r_t = epsilon_t
    epsilon_t ~ N(0, sigma_t^2)
    sigma_t^2 = omega + alpha * epsilon_{t-1}^2 + beta * sigma_{t-1}^2
"""

import numpy as np
from numba import njit
from scipy.optimize import minimize
from scipy.special import gamma as gamma_fn
from models.base_model import VolatilityModel


@njit
def garch_recursion(residuals, omega, alpha, beta):
    T = len(residuals)
    var = np.empty(T)
    var[0] = np.var(residuals) # Try unconditional variance

    for t in range(1, T):
        var[t] = omega + alpha * residuals[t - 1]**2 + beta * var[t - 1]
        var[t] = np.maximum(var[t], 1e-8)

    return var


class GARCHModel(VolatilityModel):

    def __init__(self, residuals, dist='normal'):
        super().__init__(residuals, dist)
        self.conditional_variance = None
        self.llk = None
        self.aic = None


    def log_likelihood(self, params):
        if self.dist == 'normal':
            omega, alpha, beta = params
        elif self.dist == 't':
            omega, alpha, beta, nu = params
            if nu <= 2:
                return np.inf
        elif self.dist in ['laplace', 'ged']:
            omega, alpha, beta = params
        else:
            raise ValueError(f'Unsupported distribution: {self.dist}')
        
        if omega <= 0 or alpha < 0 or beta < 0 or alpha + beta >= 1: 
            return np.inf 

        var = garch_recursion(self.residuals, omega, alpha, beta)

        if self.dist == 'normal':
            ll = -0.5 * (np.log(2 * np.pi) + np.log(var) + self.residuals**2 / var)
        elif self.dist == 't':
            const = gamma_fn((nu + 1) / 2) / (gamma_fn(nu / 2) * np.sqrt((nu - 2) * np.pi))
            z2 = self.residuals**2 / var
            ll = np.log(const) - 0.5 * np.log(var) - ((nu + 1) / 2) * np.log(1 + z2 / (nu - 2))
        elif self.dist in ['laplace', 'ged']:
            beta_ged = 1 if self.dist == 'laplace' else 1.5
            gamma_1 = gamma_fn(1 / beta_ged)
            gamma_3 = gamma_fn(3 / beta_ged)
            alpha_ged = np.sqrt(var * gamma_1 / gamma_3)
            ll = np.log(beta_ged / (2 * alpha_ged * gamma_1)) - (np.abs(self.residuals) / alpha_ged)**beta_ged
    
        return -np.sum(ll)


    def fit(self, initial_guess=None): # Try different initial guesses
        if self.dist == 'normal':
            if initial_guess is None:
                initial_guess = [1e-6, 0.05, 0.9]
            bounds = [(1e-12, None), (0, 1), (0, 1)]
            method = 'L-BFGS-B'
        elif self.dist == 't':
            if initial_guess is None:
                initial_guess = [1e-6, 0.05, 0.9, 8]
            bounds = [(1e-12, None), (0, 1), (0, 1), (2.5, 100)]
            method = 'Nelder-Mead'
        elif self.dist in ['laplace', 'ged']:
            if initial_guess is None:
                initial_guess = [1e-6, 0.05, 0.9]
            bounds = [(1e-12, None), (0, 1), (0, 1)]
            method = 'L-BFGS-B'

        result = minimize(
            fun=self.log_likelihood,
            x0=initial_guess,
            bounds=bounds,
            method=method
        )

        self.fitted = True
        self.params = result.x
        self.llk = -result.fun
        self.aic = 2 * len(self.params) - 2 * self.llk
        
        if self.dist == 'normal':
            omega, alpha, beta = self.params
        elif self.dist == 't':
            omega, alpha, beta, _ = self.params
        elif self.dist in ['laplace', 'ged']:
            omega, alpha, beta = self.params

        self.conditional_variance = garch_recursion(self.residuals, omega, alpha, beta)

        return self.params


    def forecast(self, horizon=1):
        if not self.fitted:
            raise ValueError('Model must be fitted before forecasting')

        if self.dist == 'normal':
            omega, alpha, beta = self.params
        elif self.dist == 't':
            omega, alpha, beta, _ = self.params
        elif self.dist in ['laplace', 'ged']:
            omega, alpha, beta = self.params
            
        last_residual = self.residuals[-1]
        last_var = self.conditional_variance[-1]

        forecasts = np.zeros(horizon)
        forecasts[0] = omega + alpha * last_residual**2 + beta * last_var

        # For multiday forecasting
        for t in range(1, horizon):
            forecasts[t] = omega + (alpha + beta) * forecasts[t - 1]

        return forecasts
