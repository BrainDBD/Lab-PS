import numpy as np
import matplotlib.pyplot as plt

for B in [0.5, 1, 2, 4, 8, 16]:
    start = -3
    end = 3
    x = np.linspace(start, end, 2000)
    y = np.sinc(B * x) ** 2
    
    fig, axes = plt.subplots(2, 2, figsize=(8, 6))
    ax = axes.flatten()
    idx = 0
    
    for fs in [1.0, 1.5, 2.0, 4.0]:
        N = int((end - start) * fs)
        if N % 2 == 0:
            N += 1
        x_n = np.linspace(start, end, N)
        y_n = np.sinc(B * x_n) ** 2
        
        ts = 1 / fs
        x_hat = np.zeros_like(x)
        for n in range(N):
            x_hat += y_n[n] * np.sinc((x - x_n[n]) / ts)
        
        ax[idx].plot(x, y, color = 'black')
        ax[idx].stem(x_n, y_n, linefmt='orange', markerfmt='o', basefmt='k-')
        ax[idx].plot(x, x_hat, linestyle='--', color='lightgreen')
        ax[idx].set_title(r"$F_s = {}$ Hz".format(fs))
        ax[idx].set_xlabel("t[s]") 
        ax[idx].set_xlim(start, end)
        ax[idx].set_ylabel("Amplitude") 
        ax[idx].grid(True, color='gray', linestyle='--', linewidth=0.5, alpha=0.5)
        idx += 1
        
    fig.suptitle(f"B = {B}", fontsize=16)    
    plt.tight_layout() 
    plt.savefig(f"sinc(B={B}).pdf")
    plt.show()