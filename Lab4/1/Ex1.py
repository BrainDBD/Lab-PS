import numpy as np
import matplotlib.pyplot as plt
import math
import time


def DFT(y):
    N = len(y)
    matFour = np.zeros((N, N), dtype=complex)
    for k in range(N):
        for n in range(N):
            # x (1 / math.sqrt(N))
            matFour[k, n] = (math.e ** (-2j * math.pi * k * n / N))

    return matFour @ y


def FFT(y):
    N = len(y)
    if N <= 1:
        return y
    even = FFT(y[0::2])
    odd = FFT(y[1::2])
    T = [math.e ** (-2j * math.pi * k / N) * odd[k] for k in range(N // 2)]
    return [even[k] + T[k] for k in range(N // 2)] + [even[k] - T[k] for k in range(N // 2)]


N_vals = [128, 256, 512, 1024, 2048, 4096, 8192]
DFT_times = []
FFT_times = []

for N in N_vals:
    x = np.linspace(0, 1, N)
    y = np.sin(2 * np.pi * x[:N] + 3 * np.pi / 4)

    start_time = time.perf_counter()
    DFT_result = DFT(y)
    DFT_time = time.perf_counter() - start_time
    DFT_times.append(DFT_time)

    start_time = time.perf_counter()
    FFT_result = FFT(y)
    FFT_time = time.perf_counter() - start_time
    FFT_times.append(FFT_time)

    print(f"N={N}: DFT time = {DFT_time:.6f} s, FFT time = {FFT_time:.6f} s")

plt.figure(figsize=(8, 5))
plt.plot(N_vals, DFT_times, 'o-', label='DFT', color='red')
plt.plot(N_vals, FFT_times, 'o-', label='FFT', color='blue')
plt.xlabel('Samples (N)')
plt.ylabel('Computation Time (s)')
plt.yscale('log')
plt.title('DFT vs FFT')
plt.grid()
plt.legend()
plt.tight_layout()
plt.savefig('DFT_vs_FFT.pdf')
plt.show()
