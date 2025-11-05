import numpy as np
import matplotlib.pyplot as plt
import scipy

fs, data = scipy.io.wavfile.read('aeiou.wav')
data = data.mean(axis=1)

N = len(data)
group_size = int(0.01 * N)
windows = []
step = group_size // 2

for start in range(0, N - group_size + 1, step):
    end = start + group_size
    window = data[start:end] * np.hanning(group_size)
    windows.append(window)
    
windows = np.array(windows)
num_windows, group_size = windows.shape
result_matrix = np.zeros((group_size // 2, num_windows))

for i, window in enumerate(windows):
    FFT_window = np.abs(np.fft.fft(window))[:group_size // 2] / group_size 
    result_matrix[:, i] = FFT_window
    
duration = N / fs 
plt.figure(figsize=(8, 4))
plt.imshow(
    20*np.log10(result_matrix + 1e-6),
    aspect='auto',
    origin='lower',
    extent=[0, duration, 0, fs/2],
    cmap='inferno'
)
plt.xlabel('Time (s)')
plt.ylabel('Frequency (Hz)')
plt.title('Custom Spectrogram')
plt.colorbar(label='Magnitude (dB)')
plt.savefig('CustomSpectro.pdf')
plt.show()