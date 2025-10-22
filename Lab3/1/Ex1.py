import numpy as np
import matplotlib.pyplot as plt
import math

x = np.linspace(0, 1, 64)
y = np.sin(2 * np.pi * x)

N = 64
matFour = np.zeros((N, N), dtype=complex)
for k in range(N):
    for n in range(N):
        matFour[k, n] = (1 / math.sqrt(N)) * np.exp(-2j * np.pi * k * n / N)

matFourH = np.conjugate(matFour.T)
prod = matFourH @ matFour

if np.allclose(prod, np.eye(N)):
    print("Matrice ortonormala")
else:
    print("Matrice care nu e ortonormala")

X = matFour @ y[:N]

fig, axs = plt.subplots(5)

axs[0].plot(x, matFour[0, :].real, color='red')
axs[0].plot(x, matFour[0, :].imag, color='red', linestyle='--')
axs[1].plot(x, matFour[1, :].real, color='green')
axs[1].plot(x, matFour[1, :].imag, color='green', linestyle='--')
axs[2].plot(x, matFour[2, :].real, color='blue')
axs[2].plot(x, matFour[2, :].imag, color='blue', linestyle='--')
axs[3].plot(x, matFour[N-2, :].real, color='blue')
axs[3].plot(x, matFour[N-2, :].imag, color='blue', linestyle='--')
axs[4].plot(x, matFour[N-1, :].real, color='green')
axs[4].plot(x, matFour[N-1, :].imag, color='green', linestyle='--')

plt.tight_layout()
plt.savefig('FourierMat.pdf')
plt.show()
    
