import numpy as np
import matplotlib.pyplot as plt
import math

x = np.linspace(0, 1, 1000)
y = np.sin(14 * np.pi * x) + 2 * np.sin(2 * np.pi * x) + np.sin(10 * np.pi * x) + 0.5 * np.sin(30 * np.pi * x)

fig, axs = plt.subplots(2)
axs[0].plot(x, y, color='blue')
axs[0].set_xlabel('Timp (s)')
axs[0].set_ylabel('x(t)')

N = 64
matFour = np.zeros((N, N), dtype=complex)
for k in range(N):
    for n in range(N):
        matFour[k, n] = math.e ** (1j * -2 * np.pi * k * n / N)
        
    
x_samples = np.linspace(0, 1, N)
y_samples = np.sin(14 * np.pi * x_samples) + 2 * np.sin(2 * np.pi * x_samples) + np.sin(10 * np.pi * x_samples) + 0.5 * np.sin(30 * np.pi * x_samples)

prod = matFour @ y_samples
prod = prod / N

axs[1].stem(np.abs(prod), linefmt='green', markerfmt='bo', basefmt=' ')
axs[1].set_xlabel('Frecventa (Hz)')
axs[1].set_ylabel('|X(ω)|')
plt.tight_layout()
plt.savefig('Fig.pdf')
plt.savefig('Fig.png')
plt.show()