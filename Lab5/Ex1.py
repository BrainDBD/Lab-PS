import numpy as np
import matplotlib.pyplot as plt

def fft_filter(X, cutoff, fs, type = 'low'):
    N = len(X)
    fft_X = np.fft.fft(X)
    fft_freq = np.fft.fftfreq(N, d=1/fs)
    
    if type.lower() == 'low':
        mask = np.abs(fft_freq) <= cutoff
    elif type.lower() == 'high':
        mask = np.abs(fft_freq) >= cutoff
    else:
        raise ValueError("Invalid filter type!")
    
    fft_filtered = fft_X * mask
    X_filtered = np.fft.ifft(fft_filtered).real
    return X_filtered, fft_filtered


data = np.genfromtxt('Lab5/Train.csv', delimiter=',', skip_header=1, usecols=(2))

# d)
fs = 1.0/3600
N = len(data)
fft_data = np.fft.fft(data)
fft_data = np.abs(fft_data / N)
fft_data = fft_data[:N//2]
f = np.linspace(0, N//2, N//2) * fs


fig, ax = plt.subplots(2)
ax[0].plot(f, fft_data)
ax[0].set_xlabel('Frequency [cycles per hour]')
ax[0].set_ylabel('Amplitude')
ax[0].set_title('FFT of Traffic Counts')

# e)
dc_offset = fft_data[0]
print("DC offset:", dc_offset)
data_no_dc = data - np.mean(data)
print("Mean after DC removal:", np.mean(data_no_dc))

fft_data_no_dc = np.fft.fft(data_no_dc)
fft_data_no_dc = np.abs(fft_data_no_dc / N)
fft_data_no_dc = fft_data_no_dc[:N//2]

f = np.linspace(0, N//2, N//2) * fs

ax[1].plot(f, fft_data_no_dc)
ax[1].set_xlabel('Frequency [cycles per hour]')
ax[1].set_ylabel('Amplitude')
ax[1].set_title('FFT of Traffic Counts (Offset Removed)')
plt.tight_layout()
plt.savefig("OffsetRemoval.pdf")
plt.show()

# f) 
largest_4 = np.argsort(fft_data_no_dc)[-4:][::-1]
print("\nTop 4 Frequencies:")

for i, idx in enumerate(largest_4, 1):
    freq_hour = f[idx]
    freq_hz = freq_hour / 3600
    freq_day = freq_hour * 24
    freq_month = freq_day * 30
    magnitude = fft_data_no_dc[idx]
    
    print(f'{i}: {freq_hz:.9f} Hz | {freq_hour:.6f} cycles/hour | {freq_day:.2f} cycles/day | {freq_month:.2f} cycles/month | magnitude = {magnitude:.2f}')
    
# g)

start_sample = 14160
end_sample = start_sample + 24 * 30
month_data = data[start_sample:end_sample]

plt.figure(figsize=(10, 4))
plt.plot(np.arange(0, len(month_data)) / 24, month_data)
plt.xlabel('Time [days]')
plt.ylabel('Traffic count')
plt.title('One Month of Traffic Data (Starting April 7th 2014)')
plt.grid(True)
plt.xticks(np.arange(0, len(month_data)/24, 7))
plt.tight_layout()
plt.savefig("OneMonthTraffic.pdf")
plt.show()

# i)

filtered_data, filtered_fft = fft_filter(data_no_dc, 0.25 / 3600, fs, 'low')
# 0.25 -> Disregard traffic changes that happen in less than 4 hours
plt.figure(figsize=(12, 4))
plt.plot(data_no_dc, color='blue', label='Original')
plt.plot(filtered_data, color='yellow', label='Low-Pass Filtered')
plt.xlabel('Samples')
plt.ylabel('Traffic Count')
plt.title('Original vs Low-pass Filtered Traffic Signal')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("Filter.pdf")
plt.show()