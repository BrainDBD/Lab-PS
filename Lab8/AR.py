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


def get_autocorrelation(series, max_lag):
    N = len(series)
    mean = np.mean(series)
    var = np.var(series)
    if var == 0:
        return np.zeros(max_lag)
    
    auto_corr = np.correlate(series - mean, series - mean, mode='full')[N-1:] / (var * N)
    return auto_corr[1:max_lag + 1]

# Plot Time Series & Components
N = 1000
x = np.linspace(0, 2, N)
np.random.seed(12)
series, trend, seasonal, noise = generate_time_series(x)
fig, axs = plt.subplots(4, 1, figsize=(10, 6))
axs[0].plot(x, trend, color='green', linestyle='--')
axs[0].set_title("Trend")
axs[1].plot(x, seasonal, color='purple', linestyle='--')
axs[1].set_title("Seasonal")
axs[2].plot(x, noise, color='red', linestyle='--')
axs[2].set_title("Noise")
axs[3].plot(x, series, color='blue')
axs[3].set_title("Time Series")
plt.tight_layout()
plt.savefig("1_Timeseries.pdf")
plt.show()

# Autocorrelation Analysis
p = 50
horizon = N - p
ar_model = ARModel(lags=p)
ar_model.fit(series, horizon)
fig, ax = plt.subplots(figsize=(10, 4))
auto_corr = get_autocorrelation(series=series, max_lag=p)
lags = np.arange(1, p + 1)
ax.stem(lags, auto_corr, basefmt=" ")
ax.set_xlabel("Lag")
ax.set_ylabel("Autocorrelation")
ax.set_title(f"Autocorrelation Function (p={p})")
ax.grid(True, alpha=0.3)
ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
plt.tight_layout()
plt.savefig("2_Autocorrelation.pdf")
plt.show()

# AR Model Series Recontruction
p = 100
horizon = N - p
ar_model = ARModel(lags=p)
ar_model.fit(series, horizon)
predictions = ar_model.predict(series, horizon)
fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(x, series, color='blue', label='Original Time Series')
ax.plot(x[-horizon:], predictions, color='cyan', linestyle='--', label='AR Model Predictions')
ax.set_title("AR Model Predictions vs Original Time Series")
ax.legend()
plt.tight_layout()
plt.savefig("3_AR_Predictions.pdf")
plt.show()

# Hyperparameter Tuning for one step ahead prediction
best_p = None
best_m = None
min_mse = float('inf')
N = len(series)
for p in [1, 2, 5, 10, 20, 25, 50, 100]:
    ar_model = ARModel(lags=p)
    for m in [10, 25, 50, 100, 250, 500]:
        if N - m < 2 * p:
            continue
        predictions = []
        real = []
        for i in range(N - m, N):
            train_series = series[:i]
            if len(train_series) < 2 * p:
                continue
            try:
                ar_model.fit(train_series, horizon=len(train_series) - p)
                pred = ar_model.predict(train_series, horizon=1)[0]
                predictions.append(pred)
                real.append(series[i])
            except np.linalg.LinAlgError:
                continue
            
        if len(predictions) > 0:
            mse = mean_squared_error(real, predictions)
            if mse < min_mse:
                min_mse = mse
                best_p = p
                best_m = m

print(f"Best p = {best_p} | Best m = {best_m} | MSE = {min_mse}")