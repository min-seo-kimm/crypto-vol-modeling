"""
Base class for all volatility models.

Subclasses must implement:
- log_likelihood(self, params)
- fit(self, initial_guess)
- forecast(self, horizon)
"""

class VolatilityModel:

    supported_distributions = ['normal', 't', 'ged']

    def __init__(self, residuals, dist='normal'):
        if dist not in self.supported_distributions:
            raise ValueError(f'Unsupported distribution: {dist}')

        self.residuals = residuals
        self.dist = dist
        self.params = None
        self.fitted = False


    def log_likelihood(self, params):
        raise NotImplementedError('Subclass must implement this')


    def fit(self, initial_guess):
        raise NotImplementedError('Subclass must implement this')


    def forecast(self, horizon):
        raise NotImplementedError('Subclass must implement this')
