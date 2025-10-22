import numpy as np
import scipy.io.wavfile as wav
import sounddevice as sd

# a)
xa = np.linspace(0, 1, 44100)
ya = np.sin(800 * np.pi * xa)
sd.play(ya, 44100)
sd.wait()
# b)
xb = np.linspace(0, 1, 44100)
yb = np.sin(1600 * np.pi * xb)
sd.play(yb, 44100)
sd.wait()
# c)
xsaw = np.linspace(0, 1, 44100)
ysaw = 2 * (240 * xsaw - np.floor(240 * xsaw)) - 1
sd.play(ysaw, 44100)
sd.wait()
wav.write('sawtooth.wav', 44100, ysaw)
# d) 
xsq = np.linspace(0, 1, 44100)
ysq = np.sign(np.sin(600 * np.pi * xsq))
sd.play(ysq, 44100)
sd.wait()

wav.read('sawtooth.wav')

