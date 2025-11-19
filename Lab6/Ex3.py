import numpy as np
import time

N = 10
p = np.random.randint(-25, 25, size=N)
q = np.random.randint(-25, 25, size=N)

print("Computing r = p * q using direct product")
# start = time.perf_counter() 
r = np.convolve(p, q)
# end = time.perf_counter()  
print(r)
# print(f"Time taken: {end - start:.6f} seconds")
print()

print("Computing r = q * p using fft-based multiplication")
# start = time.perf_counter() 
p = np.pad(p, (0, N-1), 'constant')
q = np.pad(q, (0, N-1), 'constant')
P = np.fft.fft(p)
Q = np.fft.fft(q)
R = P * Q
r = np.fft.ifft(R).real
r = np.round(r).astype(int)
# end = time.perf_counter()  
print(r)
# print(f"Time taken: {end - start:.6f} seconds")

