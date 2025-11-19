import numpy as np
import matplotlib.pyplot as plt

def conv(x1, x2):
    N = len(x1)
    y = [0.0] * (2 * N - 1)
    for k in range(2 * N - 1):
        for i in range(N):
            j = k - i
            if 0 <= j < N:
                y[k] += x1[i] * x2[j]
    return y
    
    
x = np.random.rand(100)
signals1 = [x]
for _ in range(3):
    x = conv(x, x)
    signals1.append(x)
    
titles1 = ["x", "x1 <- x * x", "x2 <- x1 * x1", "x3 <- x2 * x2"]
    
fig1, axes1 = plt.subplots(4, 1, figsize=(10, 8))
for ax, signal, title in zip(axes1.flatten(), signals1, titles1):
    ax.plot(signal)
    ax.set_title(title)
    
plt.tight_layout()
plt.savefig("ConvRandom.pdf")
plt.show()

rect = np.zeros(100)
rect[41:60] = 1.0
signals2 = [rect]
for _ in range(3):
    rect = conv(rect, rect)
    signals2.append(rect)
    
titles2 = ["r", "r1 <- r * r", "r2 <- r1 * r1", "r3 <- r2 * r2"]

fig2, axes2 = plt.subplots(4, 1, figsize=(10, 8))
for ax, signal, title in zip(axes2.flatten(), signals2, titles2):
    ax.plot(signal, color="g")
    ax.set_title(title)
    
plt.tight_layout()
plt.savefig("ConvRectangular.pdf")
plt.show()

# Observam in ambele figuri cum graficul vectorului devine mai rotund si mai subtiat dupa fiecare convolutie, capatand o forma Gaussiana