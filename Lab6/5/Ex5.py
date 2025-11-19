import numpy as np
import matplotlib.pyplot as plt

def rectangular_window(N):
    return np.ones(N)


def Hanning_window(N):
    n = np.arange(N)
    return 0.5 * (1 - np.cos(2 * np.pi * n / (N)))


N = 200
n = np.arange(N)
fs = 1000
t = n / fs
x = np.sin(200 * np.pi * t)
xr = x * rectangular_window(N)
xh = x * Hanning_window(N)

fig, axes = plt.subplots(1, 3, figsize=(14, 5))
axes[0].plot(n, x)
axes[0].set_title('Original Signal')
axes[0].set_xlabel('Samples')
axes[0].set_ylabel('Amplitude')
axes[0].set_ylim(-1, 1)
axes[0].grid(True)
axes[1].plot(n, xr, color='green')
axes[1].set_title('Rectangular Window')
axes[1].set_xlabel('Samples')
axes[1].set_ylabel('Amplitude')
axes[1].set_ylim(-1, 1)
axes[1].grid(True)
axes[2].plot(n, xh, color='orange')
axes[2].set_title('Hanning Window')
axes[2].set_xlabel('Samples')
axes[2].set_ylabel('Amplitude')
axes[2].set_ylim(-1, 1)
axes[2].grid(True)
plt.tight_layout()
plt.savefig("Windows.pdf")
plt.show()