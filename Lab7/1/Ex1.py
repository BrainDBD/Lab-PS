import numpy as np
import matplotlib.pyplot as plt

def f1(x1, x2):
    return np.sin(2 * np.pi * x1 + 3 * np.pi * x2)

def f2(x1, x2):
    return np.sin(4 * np.pi + x1) + np.cos(6 * np.pi * x2)

window_size = (1024, 1024)

# 1
fig, axs = plt.subplots(1, 2, figsize=(12, 5))
X = np.fromfunction(f1, window_size)
axs[0].imshow(X, cmap='gray')
axs[0].set_title("Image A")
Y = np.fft.fft2(X)
freq_db = 20*np.log10(np.abs(Y) + 1e-10) # Avoid division by 0
freq_db = np.fft.fftshift(freq_db) # Center Spectrum
im1 = axs[1].imshow(freq_db, cmap='viridis', origin='lower')
axs[1].set_title("Spectrum of Image A")
plt.colorbar(im1, ax=axs[1])
plt.tight_layout()
plt.savefig("Image_&_Spectrum_1.pdf")
plt.show()

# 2
fig, axs = plt.subplots(1, 2, figsize=(12, 5))
X = np.fromfunction(f2, window_size)
axs[0].imshow(X, cmap='gray')
axs[0].set_title("Image B")
Y = np.fft.fft2(X)
freq_db = 20*np.log10(np.abs(Y) + 1e-10)
freq_db = np.fft.fftshift(freq_db)
im = axs[1].imshow(freq_db, cmap='viridis', origin='lower')
axs[1].set_title("Spectrum of Image B")
plt.colorbar(im, ax=axs[1])
plt.tight_layout()
plt.savefig("Image_&_Spectrum_2.pdf")
plt.show()

# 3
fig, axs = plt.subplots(1, 2, figsize=(12, 5))
Y = np.zeros(window_size)
Y[0][5] = 1
Y[0][window_size[1] - 5] = 1
X = np.fft.ifft2(Y)
X = np.real(X)
axs[0].imshow(X, cmap='gray')
axs[0].set_title("Image C")
Y = np.fft.fft2(X)
freq_db = 20*np.log10(np.abs(Y) + 1e-10)
freq_db = np.fft.fftshift(freq_db)
im = axs[1].imshow(freq_db, cmap='viridis', origin='lower')
axs[1].set_title("Spectrum of Image C")
plt.colorbar(im, ax=axs[1])
plt.tight_layout()
plt.savefig("Image_&_Spectrum_3.pdf")
plt.show()

# 4
fig, axs = plt.subplots(1, 2, figsize=(12, 5))
Y = np.zeros(window_size)
Y[5][0] = 1
Y[window_size[1] - 5][0] = 1
X = np.fft.ifft2(Y)
X = np.real(X)
axs[0].imshow(X, cmap='gray')
axs[0].set_title("Image D")
Y = np.fft.fft2(X)
freq_db = 20*np.log10(np.abs(Y) + 1e-10)
freq_db = np.fft.fftshift(freq_db)
im = axs[1].imshow(freq_db, cmap='viridis', origin='lower')
axs[1].set_title("Spectrum of Image D")
plt.colorbar(im, ax=axs[1])
plt.tight_layout()
plt.savefig("Image_&_Spectrum_4.pdf")
plt.show()

# 5
fig, axs = plt.subplots(1, 2, figsize=(12, 5))
Y = np.zeros(window_size)
Y[5][5] = 1
Y[window_size[0] - 5][window_size[1] - 5] = 1
X = np.fft.ifft2(Y)
X = np.real(X)
axs[0].imshow(X, cmap='gray')
axs[0].set_title("Image E")
Y = np.fft.fft2(X)
freq_db = 20*np.log10(np.abs(Y) + 1e-10)
freq_db = np.fft.fftshift(freq_db)
im = axs[1].imshow(freq_db, cmap='viridis', origin='lower')
axs[1].set_title("Spectrum of Image E")
plt.colorbar(im, ax=axs[1])
plt.tight_layout()
plt.savefig("Image_&_Spectrum_5.pdf")
plt.show()
