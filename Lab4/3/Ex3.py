import numpy as np
import matplotlib.pyplot as plt

f = 4
fs = 14 # 7 * 2 seconds
x = np.linspace(0, 2, 1000, endpoint=False)
y1 = np.sin(2 * (f + 0 * fs / 2) * np.pi * x)
y2 = np.sin(2 * (f + 1 * fs / 2) * np.pi * x)
y3 = np.sin(2 * (f + 2 * fs / 2) * np.pi * x)

x_sampled = np.linspace(0, 2, 4 * (fs + f) + 2, endpoint=False) # Increased Sampling Rate as per Nyquist rate
y1_sampled = np.sin(2 * (f + 0 * fs / 2) * np.pi * x_sampled)
y2_sampled = np.sin(2 * (f + 1 * fs / 2) * np.pi * x_sampled)
y3_sampled = np.sin(2 * (f + 2 * fs / 2) * np.pi * x_sampled)

fig, axs = plt.subplots(3, figsize=(8, 6))
axs[0].plot(x, y1, color='blue')
axs[0].stem(x_sampled, y1_sampled, markerfmt='yo', basefmt=' ', linefmt='none')
axs[1].plot(x, y2, color='purple')
axs[1].stem(x_sampled, y2_sampled, markerfmt='yo', basefmt=' ', linefmt='none')
axs[2].plot(x, y3, color='green')
axs[2].stem(x_sampled, y3_sampled, markerfmt='yo', basefmt=' ', linefmt='none')
for ax in axs:
    ax.set_xlim(0, 1)
    ax.grid()

fig.suptitle('Aliasing Removed', fontsize=16)
plt.tight_layout()
plt.savefig('Aliasing_Removed.pdf')
plt.show()
