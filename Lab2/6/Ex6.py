import numpy as np
import matplotlib.pyplot as plt

samplerate = 40
x = np.linspace(0, 1, 5 * samplerate)

y1 = np.sin(2 * np.pi * (samplerate / 2) * x)
y2 = np.sin(2 * np.pi * (samplerate / 4) * x)
y3 = np.sin(2 * np.pi * 0 * x)

fig, axs = plt.subplots(3)
axs[0].stem(x, y1)
axs[0].set_title('F = 1/2 Sampling Rate')
axs[1].stem(x, y2)
axs[1].set_title('F = 1/4 Sampling Rate')
axs[2].stem(x, y3)
axs[2].set_title('F = 0 Hz')

for ax in axs:
    ax.grid(True, which='both', linestyle='--', color='purple', linewidth=0.5)
plt.tight_layout()
plt.savefig("DifferentFrequencies.pdf")
plt.show()

