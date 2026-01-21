import numpy as np
import matplotlib.pyplot as plt

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

def hankelize(matrix):
    L, K = matrix.shape
    N = L + K - 1
    hankel_matrix = np.zeros(N)
    counts = np.zeros(N)

    for i in range(L):
        for j in range(K):
            hankel_matrix[i + j] += matrix[i, j]
            counts[i + j] += 1

    hankel_matrix /= counts
    return hankel_matrix


N = 1000
x = np.linspace(0, 2, N)
np.random.seed(12)
series, trend, seasonal, noise = generate_time_series(x)

L = 100
K = N - L + 1
X = np.column_stack([series[i:i+L] for i in range(K)])

eig_XXT = np.linalg.eig(X @ X.T)
eig_XTX = np.linalg.eig(X.T @ X)
U, S, VT = np.linalg.svd(X, full_matrices=False)
print(f"Do the eigen values of X * X.T equal S^2? {np.allclose(np.sort(eig_XXT[0])[::-1][:len(S)], S**2)}") 
print(f"Do the eigen values of X.T * X equal S^2? {np.allclose(np.sort(eig_XTX[0])[::-1][:len(S)], S**2)}")

X_i = [S[i] * np.outer(U[:, i], VT[i, :]) for i in range(len(S))]
X_i_hat = [hankelize(X_i[i]) for i in range(len(S))]
series_hat = sum(X_i_hat)

fig, axs = plt.subplots(2, figsize=(10, 12))
axs[0].plot(x, series, color='blue')
axs[1].plot(x, series_hat, color='purple')
axs[0].set_title('Original Time Series')
axs[1].set_title('Reconstructed Time Series')
for ax in axs:
    ax.grid()
plt.tight_layout()
plt.savefig('SSA_Reconstruction.pdf')
plt.show()

print(f"Is the reconstruction close to the original? {np.allclose(series, series_hat)}")
print(f"Max absolute difference: {np.max(np.abs(series - series_hat))}")