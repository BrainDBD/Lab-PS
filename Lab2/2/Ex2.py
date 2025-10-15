import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(0, 2, 200)

y1 = np.sin(200 * np.pi * x )
y2 = np.sin(200 * np.pi * x + np.pi / 2)
y3 = np.sin(200 * np.pi * x + np.pi)
y4 = np.sin(200 * np.pi * x + 3 * np.pi / 2)

plt.plot(x, y1, label='y1 = sin(200πx)', color='blue')
plt.plot(x, y2, label='y2 = sin(200πx + π/2)', color='red')
plt.plot(x, y3, label='y3 = sin(200πx + π)', color='green')
plt.plot(x, y4, label='y4 = sin(200πx + 3π/2)', color='orange')
plt.legend(loc='upper right')
plt.savefig('Lab2/PhaseShift.pdf')
plt.show()

fig, axs = plt.subplots(5)

noise = np.random.normal(0, 1, y1.shape)
ratio = [0.1, 1, 10, 100, 1000]
signal_power = np.linalg.norm(y1)**2
noise_power = np.linalg.norm(noise)**2

for i in range(5):
    k = np.sqrt(signal_power / (ratio[i] * noise_power))
    y1_noisy = y1 + k * noise

    axs[i].plot(x, y1_noisy, label=f'y1 with Signal to Noise Ratio = {ratio[i]}', color='purple')
    axs[i].plot(x, y1, label='Original y1', color='blue', linestyle='--')
    axs[i].legend(loc='upper right')

plt.savefig('Lab2/SignalNoising.pdf')
plt.show()