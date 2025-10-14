import numpy as np
import matplotlib.pyplot as plt

# a)
x = np.linspace(0, 0.03, int(0.03/0.0005))
print(x)

# b)
fig, axs = plt.subplots(3)
axs[0].plot(x, np.cos(520 * np.pi * x + np.pi / 3))
axs[1].plot(x, np.cos(280 * np.pi * x - np.pi / 3))
axs[2].plot(x, np.cos(120 * np.pi * x + np.pi / 3))
plt.savefig('1b.pdf')
plt.show()

# c)
xa = np.linspace(0, 1, 200)
fig2, axs2 = plt.subplots(3)

axs2[0].plot(x, np.cos(520 * np.pi * x + np.pi / 3))
axs2[0].stem(xa, np.cos(520 * np.pi * xa + np.pi / 3))
axs2[1].plot(x, np.cos(280 * np.pi * x - np.pi / 3))
axs2[1].stem(xa, np.cos(280 * np.pi * xa - np.pi / 3))
axs2[2].plot(x, np.cos(120 * np.pi * x + np.pi / 3))
axs2[2].stem(xa, np.cos(120 * np.pi * xa + np.pi / 3))

for ax in axs2:
    ax.set_xlim([0, 0.03])
plt.savefig('1c.pdf')
plt.show()
