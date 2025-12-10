import numpy as np
import matplotlib.pyplot as plt

class MAModel:
    def __init__(self, horizon):
        self.horizon = horizon
        self.coefficients = None

    def get_average(self, series, horizon):
        N = len(series)
        start = N - horizon
        end = N
        return np.mean(series[start:end])

    def get_errors(self, series, horizon):
        N = len(series)
        errors = []
        avg = self.get_average(series, horizon)
        for t in range(N - 1, N - horizon - 1, -1):
            error = series[t] - avg
            errors.append(error)
        return np.array(errors)
    
    def fit(self, series):
        return 0
    
    def predict(self, series, steps=1):
        return 0
        

def generate_time_series(samples):
    trend_func = lambda x: 2 * x * x + x - 5
    season_sin1 = lambda x: np.sin(20 * np.pi * x)
    season_sin2 = lambda x: np.sin(10 * np.pi * x + np.pi / 4)
    noise_func = lambda x: np.random.normal(0, 1, size=x.shape)
    trend = trend_func(samples)
    seasonality = 3 * season_sin1(samples) + 5 * season_sin2(samples)
    noise = noise_func(samples)
    series = trend + seasonality + noise
    return series, trend, seasonality, noise


N = 1000
x = np.linspace(0, 2, N)
series, trend, seasonal, noise = generate_time_series(x)
horizon = 10
ma_model = MAModel(horizon)