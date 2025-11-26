from scipy import misc, ndimage
import numpy as np
import matplotlib.pyplot as plt

X = misc.face(gray=True)
pixel_noise = 200
noise = np.random.randint(-pixel_noise, pixel_noise+1, size=X.shape)
X_noisy = X + noise
signal_power = np.mean(X ** 2)
noise_power = np.mean(noise ** 2)
snr = 10 * np.log10(signal_power / noise_power)
print(f"SNR before denoising: {snr:.2f} dB")

Y = np.fft.fft2(X_noisy)
Y = np.fft.fftshift(Y)
rows, cols = X_noisy.shape
u = np.arange(-cols//2, cols//2)
v = np.arange(-rows//2, rows//2)
U, V = np.meshgrid(u, v) # Create frequency grid
R = np.sqrt(U**2 + V**2) # Compute radial distances
cutoff = 200 # Frequency cutoff for low-pass filter
mask = (R < cutoff)
Y_shift_filtered = Y * mask
Y_filtered = np.fft.ifftshift(Y_shift_filtered)
X_denoised = np.real(np.fft.ifft2(Y_filtered))
X_denoised = np.clip(X_denoised, 0, 255)
removed_noise = X_noisy - X_denoised # Assume we only have the noisy image
signal_power = np.mean(X_denoised ** 2)
noise_power = np.mean(removed_noise ** 2)
snr = 10 * np.log10(signal_power / noise_power)
print(f"SNR after denoising: {snr:.2f} dB")

fig, axs = plt.subplots(1, 2, figsize=(15, 5))
axs[0].imshow(X_noisy, cmap=plt.cm.gray)
axs[0].set_title('Noisy Raccoon')
axs[1].imshow(X_denoised, cmap=plt.cm.gray)
axs[1].set_title('Denoised Raccoon')
plt.tight_layout()
plt.savefig("Denoise.pdf")
plt.show()