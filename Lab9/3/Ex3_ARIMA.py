import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
import warnings

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

best_p, best_q = 0, 0
best_model = None
best_aic = float('inf')

warnings.filterwarnings("ignore")
for p in range(0, 21, 5):
    for q in range(0, 21, 5):
        try:
            model = ARIMA(series, order=(p, 0, q))
            model_fit = model.fit()
            # print(f"Fitted ARIMA({p}, 0, {q}) | AIC = {model_fit.aic}")
            if model_fit.aic < best_aic:
                best_p, best_q = p, q
                best_model = model_fit
                best_aic = model_fit.aic
        except:
            continue

plt.plot(x, series, label='Original Series')
plt.plot(x, best_model.fittedvalues, label=f'ARIMA({best_p}, 0, {best_q})', color='orange')
plt.title(f'Best ARIMA Model: ARIMA({best_p}, 0, {best_q})')
plt.legend()
plt.tight_layout()
plt.savefig("ARIMA.pdf")
plt.show()