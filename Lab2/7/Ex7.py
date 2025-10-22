import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(0, 1, 1000)
y = np.sin(4 * np.pi * x)

x1 = x[::4]
y1 = y[::4]

x2 = x1[::4]
y2 = y1[::4]

fig, axs = plt.subplots(3)
axs[0].stem(x, y)
axs[0].set_title('Original Signal')
axs[1].stem(x1, y1)
axs[1].set_title('1/4 Decimated Signal')
axs[2].stem(x2, y2)
axs[2].set_title('1/16 Decimated Signal')

plt.tight_layout()
plt.savefig("SignalDecimation.pdf")
plt.show()