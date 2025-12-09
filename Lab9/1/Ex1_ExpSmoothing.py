import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

def simple_exp_smoothing(series, alpha):
    predict = [series[0]]
    for t in range(1, len(series)):
        predict.append(alpha * series[t] + (1 - alpha) * predict[-1])
    return np.array(predict)

def double_exp_smoothing(series, alpha, beta):
    s = [series[0]]
    b = [series[1] - series[0]]
    predict = [series[0]]
    for t in range(1, len(series)):
        s_new = alpha * series[t] + (1 - alpha) * (s[-1] + b[-1])
        b_new = beta * (s_new - s[-1]) + (1 - beta) * b[-1]
        predict.append(s_new + b_new)
        s.append(s_new)
        b.append(b_new)
    return np.array(predict)

def triple_exp_smoothing(series, alpha, beta, gamma, season_length):
    n = len(series)
    s = [series[0]]
    b = [series[1] - series[0]]
    c = [series[i] - series[0] for i in range(season_length)]
    predict = [series[0]]
    for t in range(1, n):
        if t - season_length >= 0:
            c_prev = c[t - season_length]
        else:
            c_prev = 0
        s_new = alpha * (series[t] - c_prev) + (1 - alpha) * (s[-1] + b[-1])
        b_new = beta * (s_new - s[-1]) + (1 - beta) * b[-1]
        c_new = gamma * (series[t] - s_new - b[-1]) + (1 - gamma) * c_prev
        predict.append(s_new + b_new + c_new)
        s.append(s_new)
        b.append(b_new)
        c.append(c_new)
    return np.array(predict)

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
fixed_alpha = 0.7
simple = simple_exp_smoothing(series, fixed_alpha)
fixed_beta = 0.5
double = double_exp_smoothing(series, fixed_alpha, fixed_beta)
fixed_gamma = 0.3
triple = triple_exp_smoothing(series, fixed_alpha, fixed_beta, fixed_gamma, season_length=100)
fig, axs = plt.subplots(4, figsize=(15, 10))
axs[0].plot(series, label='Original Series')
axs[0].set_title("Original Time Series")
axs[1].plot(simple, color='orange', label='Simple Exponential Smoothing')
axs[1].set_title("Simple Exponential Smoothing (alpha=0.7)")
axs[2].plot(double, color='green', label='Double Exponential Smoothing')
axs[2].set_title("Double Exponential Smoothing (alpha=0.7, beta=0.5)")
axs[3].plot(triple, color='red', label='Triple Exponential Smoothing')
axs[3].set_title("Triple Exponential Smoothing (alpha=0.7, beta=0.5, gamma=0.3)")
plt.tight_layout()
plt.savefig("Fixed_Exponential_Smoothing.pdf")
plt.show()


fig, axs = plt.subplots(4, figsize=(15, 10))
axs[0].plot(series, label='Original Series')
axs[0].set_title("Original Time Series")

best_alpha = None
min_mse = float('inf')
for alpha in np.arange(0.05, 1.01, 0.05):
    simple = simple_exp_smoothing(series, alpha)
    mse = mean_squared_error(series, simple)
    if mse < min_mse:
        min_mse = mse
        best_alpha = alpha
        
simple = simple_exp_smoothing(series, best_alpha)
axs[1].plot(simple, color='orange', label='Simple Exponential Smoothing')
axs[1].set_title(f"Simple Exponential Smoothing (best alpha={best_alpha:.2f})")

best_alpha = None
best_beta = None
min_mse = float('inf')
for alpha in np.arange(0.1, 1.01, 0.1):
    for beta in np.arange(0.1, 1.01, 0.1):
        double = double_exp_smoothing(series, alpha, beta)
        mse = mean_squared_error(series, double)
        if mse < min_mse:
            min_mse = mse
            best_alpha = alpha
            best_beta = beta
            
double = double_exp_smoothing(series, best_alpha, best_beta)
axs[2].plot(double, color='green', label='Double Exponential Smoothing')
axs[2].set_title(f"Double Exponential Smoothing (best alpha={best_alpha:.2f}, best beta={best_beta:.2f})")

best_alpha = None
best_beta = None
best_gamma = None
min_mse = float('inf')
for alpha in np.arange(0.2, 1.01, 0.2):
    for beta in np.arange(0.2, 1.01, 0.2):
        for gamma in np.arange(0.2, 1.01, 0.2):
            triple= triple_exp_smoothing(series, alpha, beta, gamma, season_length=100)
            mse = mean_squared_error(series, triple)
            if mse < min_mse:
                min_mse = mse
                best_alpha = alpha
                best_beta = beta
                best_gamma = gamma
                
triple = triple_exp_smoothing(series, best_alpha, best_beta, best_gamma, season_length=100)
axs[3].plot(triple, color='red', label='Triple Exponential Smoothing')
axs[3].set_title(f"Triple Exponential Smoothing (best alpha={best_alpha:.2f}, best beta={best_beta:.2f}, best gamma={best_gamma:.2f})")
plt.tight_layout()
plt.savefig("Best_Exponential_Smoothing.pdf")
plt.show()