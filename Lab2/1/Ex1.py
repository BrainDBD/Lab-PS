import numpy as np
import matplotlib.pyplot as plt

fig, axs = plt.subplots(2)

xsin = np.linspace(0, 1, 200)
ysin = np.sin(200 * np.pi * xsin)

xcos = np.linspace(0, 1, 200)
ycos = np.cos(200 * np.pi * xcos + np.pi / 2)

axs[0].plot(xsin, ysin, color='red')
axs[0].set_title('Sin')

axs[1].plot(xcos, ycos, color='blue')
axs[1].set_title('Cos')

plt.savefig('SinCos.pdf')
plt.show()