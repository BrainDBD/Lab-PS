import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

class ARModel:
    def __init__(self, lags):
        self.lags = lags
        self.coefficients = None
    
    def _get_lagged_matrix(self, series, horizon):
        N = len(series)
        if N < self.lags + horizon:
            raise ValueError(f"Series length ({N}) must be >= lags + horizon ({self.lags + horizon})")  
        lagged = np.zeros((horizon, self.lags))
        for i in range(horizon):
            start = N - self.lags - i
            end = N - i
            lagged[i, :] = series[start:end][::-1]
        return lagged
        
    def fit(self, series, horizon):
        target = series[-horizon:]
        lagged = self._get_lagged_matrix(series, horizon)
        
        normal = np.dot(lagged.T, lagged)
        correlation = np.dot(lagged.T, target)
        self.coefficients = np.linalg.solve(normal, correlation)
        return self
    
    def greedy_sparse_fit(self, series, horizon, regressor_count):
        target = series[-horizon:]
        lagged = self._get_lagged_matrix(series, horizon)
        
        selected = []
        remaining = list(range(self.lags))
        for _ in range(regressor_count):
            best_mse = float('inf')
            best_idx = -1
            for idx in remaining:
                test_idx = selected + [idx]
                test_lagged = lagged[:, test_idx]
                normal = np.dot(test_lagged.T, test_lagged)
                correlation = np.dot(test_lagged.T, target)
                test_coeffs = np.linalg.solve(normal, correlation)
                predictions = np.dot(test_lagged, test_coeffs)
                
                mse = mean_squared_error(target, predictions)
                if mse < best_mse:
                    best_mse = mse
                    best_idx = idx
                
            selected.append(best_idx)
            remaining.remove(best_idx)
            
        lagged = lagged[:, selected]
        normal = np.dot(lagged.T, lagged)
        correlation = np.dot(lagged.T, target)
        nonzero_coeffs = np.linalg.solve(normal, correlation)
        self.coefficients = np.zeros(self.lags)
        for i, idx in enumerate(selected):
            self.coefficients[idx] = nonzero_coeffs[i]
        return self, sorted(selected)
    
    def predict(self, series, horizon):        
        lagged = self._get_lagged_matrix(series, horizon)        
        predictions = np.dot(lagged, self.coefficients)
        return predictions
    

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
np.random.seed(12)
series, trend, seasonal, noise = generate_time_series(x)

p = 10
regressors = 5
train_size = int(0.8 * N)
train_series = series[:train_size]
test_series = series[train_size:]
train_horizon = len(train_series) - p
test_horizon = len(test_series) - p
model = ARModel(lags=p)
model, selected = model.greedy_sparse_fit(train_series, horizon=train_horizon, regressor_count=regressors)
print(f"Indices of the best {regressors} regressors: {selected}")

predictions = model.predict(test_series, horizon=test_horizon)
actual = test_series[-test_horizon:]
plt.figure(figsize=(12, 6))
plt.plot(actual, label='Actual', color='blue')
plt.plot(predictions, label='Predicted', color='red', linestyle='--')
plt.title('Sparse AR Model Predictions (Greedy Selection)')
plt.tight_layout()
plt.savefig("Sparse_AR_Greedy.pdf")
plt.show()