import numpy as np
import matplotlib.pyplot as plt

fig, axs = plt.subplots(3)

xsin = np.linspace(0, 1, 24000)
ysin = np.sin(2400 * np.pi * xsin)
xsaw = np.linspace(0, 1, 24000)
ysaw = 2 * (240 * xsaw - np.floor(240 * xsaw)) - 1

xsum = xsin
ysum = ysin + ysaw

axs[0].plot(xsin, ysin, color = 'blue')
axs[0].set_title('Sine Wave')
axs[0].grid(True, which='both', linestyle='--', color='purple', linewidth=0.5)
axs[0].set_xlim([0, 0.02])
axs[0].set_ylim([-1, 1])

axs[1].plot(xsaw, ysaw, color = 'magenta')
axs[1].set_title('Sawtooth Wave')
axs[1].grid(True, which='both', linestyle='--', color='purple', linewidth=0.5)
axs[1].set_xlim([0, 0.02])
axs[1].set_ylim([-1, 1])

axs[2].plot(xsum, ysum, color = 'orange')
axs[2].set_title('Sine + Sawtooth Wave')
axs[2].grid(True, which='both', linestyle='--', color='purple', linewidth=0.5)
axs[2].set_xlim([0, 0.02])
axs[2].set_ylim([-2, 2])

plt.tight_layout()
plt.savefig("SumWaves.pdf")
plt.show()
