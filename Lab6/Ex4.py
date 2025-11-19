import numpy as np

N = 20
d = 2
x = np.sin(np.linspace(0, 2*np.pi, N))
y = np.zeros(N)
y[d:] = x[:-d]
print(f'd = {d}')

X = np.fft.fft(x)
Y = np.fft.fft(y)

print("Estimating d using IFFT(conj(FFT(x)) * FFT(y))")
corr = np.fft.ifft(np.conj(X) * (Y)).real
d_est = np.argmax(corr)
print(f'd = {d_est}')

print("Estimating d using IFFT(FFT(y) / FFT(x))")
div = np.fft.ifft(Y / X).real
d_est = np.argmax(div)
print(f'd = {d_est}')

# A doua metoda intampina probleme cu numere ~0 deoarece foloseste impartire. Prima metoda evita aceasta problema.