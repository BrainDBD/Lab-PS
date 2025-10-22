import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(-np.pi/2, np.pi/2, 1000)
xsin = np.sin(x)

xtay = x
err_tay = np.abs(xsin - xtay)
xpade = (x - 7*x**3/60) / (1 + x**2/20)
err_pade = np.abs(xsin - xpade)


fig, axs = plt.subplots(1, 2, figsize=(12, 5))

axs[0].plot(x, xsin, label='sin(x)', color='blue')
axs[0].plot(x, xtay, label='Taylor', color='red', linestyle='--')
axs[0].set_ylim([-1, 1])
axs[0].legend()
axs[0].set_title("Taylor Approximation")

axs[1].plot(x, xsin, label='sin(x)', color='blue')
axs[1].plot(x, xpade, label='Padé', color='red', linestyle='--')
axs[1].set_ylim([-1, 1])
axs[1].legend()
axs[1].set_title("Padé Approximation")

plt.tight_layout()
plt.savefig("Lab2/8/Approximations.pdf")
plt.show()

fig, axs = plt.subplots(1, 2, figsize=(12, 5))

axs[0].plot(x, err_tay, label='|sin(x) - x|', color='purple')
axs[0].set_ylim([0, 1])
axs[0].legend()
axs[0].set_title("Taylor Approximation Error (Linear Scale)")

axs[1].plot(x, err_pade, label='|sin(x) - Padé|', color='orange')
axs[1].set_ylim([0, 1])
axs[1].legend()
axs[1].set_title("Padé Approximation Error (Linear Scale)")

plt.tight_layout()
plt.savefig("Lab2/8/Errors-Linear.pdf")
plt.show()

fig, axs = plt.subplots(1, 2, figsize=(12, 5))

axs[0].plot(x, err_tay, label='|sin(x) - x|', color='purple')
axs[0].set_yscale('log')
axs[0].legend()
axs[0].set_title("Taylor Approximation Error (Log Scale)")

axs[1].plot(x, err_pade, label='|sin(x) - Padé|', color='orange')
axs[1].set_yscale('log')
axs[1].legend()
axs[1].set_title("Padé Approximation Error (Log Scale)")

plt.tight_layout()
plt.savefig("Lab2/8/Errors-Log.pdf")
plt.show()
