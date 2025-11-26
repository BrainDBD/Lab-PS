from scipy import misc
import numpy as np
import matplotlib.pyplot as plt

X = misc.face(gray=True)
Y = np.fft.fft2(X)
freq_db = 20*np.log10(np.abs(Y) + 1e-10)

snr_threshold = 2.0
freq_cutoff = 50
step = 10
max_iterations = 100

for _ in range(max_iterations):
    Y_cutoff = Y.copy()
    Y_cutoff[freq_db < freq_cutoff] = 0
    X_cutoff = np.fft.ifft2(Y_cutoff)
    X_cutoff = np.real(X_cutoff)    # Avoid rounding errors in the complex domain,
    
    signal_power = np.mean(X ** 2)
    noise_power = np.mean((X - X_cutoff) ** 2)
    if noise_power == 0:
        current_snr = float('inf')
    else:
        current_snr = 10 * np.log10(signal_power / noise_power)
    
    if current_snr <= snr_threshold:
        print(f"Desired SNR reached: {current_snr:.2f} dB at cutoff frequency: {freq_cutoff}")
        break
    
    freq_cutoff += step
else:
    print(f"Max iterations reached. Final SNR: {current_snr:.2f} dB, cutoff frequency: {freq_cutoff}")
    
fig, axs = plt.subplots(1, 2, figsize=(12, 5))
axs[0].imshow(X, cmap=plt.cm.gray)
axs[0].set_title("Uncompressed Raccoon")

axs[1].imshow(X_cutoff, cmap=plt.cm.gray)
axs[1].set_title("Compressed Raccoon")
plt.tight_layout()
plt.savefig("Compression.pdf")
plt.show()