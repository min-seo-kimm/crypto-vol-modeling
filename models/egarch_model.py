"""
Implementation of the EGARCH(1,1) model.

Model:
    r_t = epsilon_t
    z_t = epsilon_t / sigma_t
    epsilon_t ~ N(0, sigma_t^2)
    ln(sigma_t^2) = omega + alpha * (|z_{t-1}| - E|z_{t-1}|) + gamma * (z_{t-1}) + beta * ln(sigma_{t-1}^2)
"""

import numpy as np
from numba import njit
from scipy.optimize import minimize
from scipy.stats import norm
from scipy.special import gamma as gamma_fn
from models.base_model import VolatilityModel


@njit
def egarch_recursion(residuals, omega, alpha, beta, gamma, expected_abs_z):
    T = len(residuals)
    log_var = np.empty(T)
    var = np.empty(T)
    z = np.empty(T)

    log_var[0] = np.log(np.var(residuals)) # Try unconditional variance
    var[0] = np.exp(log_var[0])
    z[0] = residuals[0] / np.sqrt(var[0])

    for t in range(1, T):
        log_var[t] = omega + alpha * (np.abs(z[t - 1]) - expected_abs_z) + gamma * z[t - 1] + beta * log_var[t - 1]
        log_var[t] = min(max(log_var[t], -20), 20)
        var[t] = max(np.exp(log_var[t]), 1e-8)
        z[t] = residuals[t] / np.sqrt(var[t])

    return var


class EGARCHModel(VolatilityModel):

    def __init__(self, residuals, dist='normal'):
        super().__init__(residuals, dist)
        self.conditional_log_variance = None
        self.conditional_variance = None
        self.llk = None
        self.aic = None


    def log_likelihood(self, params):
        if self.dist == 'normal':
            omega, alpha, beta, gamma = params
            expected_abs_z = np.sqrt(2 / np.pi)
        elif self.dist == 't':
            omega, alpha, beta, gamma, nu = params
            expected_abs_z = 2 * gamma_fn((nu + 1) / 2) / (gamma_fn(nu / 2) * (nu - 1) * np.sqrt(np.pi / (nu - 2)))
            if nu <= 2:
                return np.inf
        elif self.dist in ['laplace', 'ged']:
            omega, alpha, beta, gamma = params
            expected_abs_z = 1
        else:
            raise ValueError(f'Unsupported distribution: {self.dist}')
        
        if np.abs(beta) >= 1:
            return np.inf

        var = egarch_recursion(self.residuals, omega, alpha, beta, gamma, expected_abs_z)

        if self.dist == 'normal':
            ll = -0.5 * (np.log(2 * np.pi) + np.log(var) + self.residuals**2 / var)
        elif self.dist == 't':
            const = gamma_fn((nu + 1) / 2) / (gamma_fn(nu / 2) * np.sqrt(np.pi * (nu - 2)))
            z2 = self.residuals**2 / var
            ll = np.log(const) - 0.5 * np.log(var) - ((nu + 1) / 2) * np.log(1 + z2 / (nu - 2))
        elif self.dist in ['laplace', 'ged']:
            beta_ged = 1 if self.dist == 'laplace' else 1.5
            gamma_1 = gamma_fn(1 / beta_ged)
            gamma_3 = gamma_fn(3 / beta_ged)
            alpha_ged = np.sqrt(var * gamma_1 / gamma_3)
            ll = np.log(beta_ged / (2 * alpha_ged * gamma_1)) - (np.abs(self.residuals) / alpha_ged) ** beta_ged

        return -np.sum(ll) + self.penalty_term(params)


    def penalty_term(self, params):
        penalty = 0
        if self.dist == 't':
            nu = params[-1]
            if nu < 3.5:
                penalty = 10 * (4 - nu)**2

        return penalty    


    def fit(self, initial_guess=None): # Try different initial guesses
        if self.dist == 'normal':
            if initial_guess is None:
                initial_guess = [-0.1, 0.1, 0.9, 0]
            bounds = [(-5, 5), (0, 1), (0, 0.99), (-1, 1)]
            method = 'L-BFGS-B'
        elif self.dist == 't':
            if initial_guess is None:
                initial_guess = [-0.1, 0.1, 0.9, 0, 8]
            bounds = [(-5, 5), (0, 1), (0, 0.99), (-1, 1), (2.5, 100)]
            method = 'Nelder-Mead'
        elif self.dist in ['laplace', 'ged']:
            if initial_guess is None:
                initial_guess = [-0.1, 0.1, 0.9, 0]
            bounds = [(-5, 5), (0, 1), (0, 0.99), (-1, 1)]
            method = 'L-BFGS-B'

        result = minimize(
            fun=self.log_likelihood,
            x0=initial_guess,
            bounds=bounds,
            method=method
        )

        self.fitted = True
        self.params = result.x
        self.llk = -result.fun + self.penalty_term(self.params)
        self.aic = 2 * len(self.params) - 2 * self.llk

        if self.dist == 'normal':
            omega, alpha, beta, gamma = self.params
            expected_abs_z = np.sqrt(2 / np.pi)
        elif self.dist == 't':
            omega, alpha, beta, gamma, nu = self.params
            expected_abs_z = 2 * gamma_fn((nu + 1) / 2) / (gamma_fn(nu / 2) * (nu - 1) * np.sqrt(np.pi / (nu - 2)))
        elif self.dist in ['laplace', 'ged']:
            omega, alpha, beta, gamma = self.params
            expected_abs_z = 1

        self.conditional_variance = egarch_recursion(self.residuals, omega, alpha, beta, gamma, expected_abs_z)
        self.conditional_log_variance = np.log(self.conditional_variance)

        return self.params


    def forecast(self, horizon=1):
        if not self.fitted:
            raise ValueError('Model must be fitted before forecasting')

        if self.dist == 'normal':
            omega, alpha, beta, gamma = self.params
            expected_abs_z = np.sqrt(2 / np.pi)
        elif self.dist == 't':
            omega, alpha, beta, gamma, nu = self.params
            expected_abs_z = 2 * gamma_fn((nu + 1) / 2) / (gamma_fn(nu / 2) * (nu - 1) * np.sqrt(np.pi / (nu - 2)))
        elif self.dist in ['laplace', 'ged']:
            omega, alpha, beta, gamma = self.params
            expected_abs_z = 1
            
        last_residual = self.residuals[-1]
        last_var = self.conditional_variance[-1]
        last_z = last_residual / np.sqrt(last_var)

        forecasts = np.zeros(horizon)
        forecasts[0] = np.exp(omega + alpha * (np.abs(last_z) - expected_abs_z) + gamma * last_z + beta * np.log(last_var))
        const = np.exp(omega - alpha * expected_abs_z) * (np.exp((alpha + gamma)**2 / 2) * norm.cdf(alpha + gamma) + np.exp((alpha - gamma)**2 / 2) * norm.cdf(alpha - gamma))

        # For multiday forecasting
        for t in range(1, horizon):
            forecasts[t] = const * forecasts[t - 1]**beta

        return forecasts
