import numpy as np
import matplotlib.pyplot as plt 

# a)
xa = np.linspace(0, 4, 1600)
ya = np.sin(800 * np.pi * xa)
plt.plot(xa, ya)
plt.stem(xa, ya)
plt.savefig("2a.pdf")
plt.show()

# b)
xb = np.linspace(0, 3, 4800)
yb = np.sin(1600 * np.pi * xb)
plt.plot(xb, yb)
plt.stem(xb, yb)
plt.savefig("2b.pdf")
plt.show()

# c)
xsaw = np.linspace(0, 1, 24000)
ysaw = 2 * (240 * xsaw - np.floor(240 * xsaw)) - 1
plt.plot(xsaw, ysaw, color = 'magenta')
plt.grid(True, which='both', linestyle='--', color='purple', linewidth=0.5)
plt.xlim([0, 0.01])
plt.xlabel('Time')
plt.ylim([-1, 1])
plt.ylabel('Amplitudine')
plt.title('Sawtooth Wave')
plt.savefig("2c.pdf")
plt.show()

# d) 

xsq = np.linspace(0, 1, 6000)
ysq = np.sign(np.sin(600 * np.pi * xsq))
plt.plot(xsq, ysq, color = 'green')
plt.grid(True, which='both', linestyle='--', color='purple', linewidth=0.5)
plt.xlim([0, 0.01])
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.title('Square Wave')
plt.savefig("2d.pdf")
plt.show()

# e)

arr = np.random.rand(128, 128)
plt.imshow(arr, cmap='grey')
plt.savefig("2e.pdf")
plt.show()

# f)

arr = np.zeros((128, 128))
for i in range(128):
        arr[i, :] = i % 128
plt.imshow(arr, cmap='inferno')
plt.savefig("2f.pdf")
plt.show()