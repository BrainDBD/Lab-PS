import numpy as np
import sounddevice as sd
import scipy.io.wavfile as wav

xtri = np.linspace(0, 2, 48000 * 2)
ytri1 = 2 * np.abs(2 * (240 * xtri - np.floor(240 * xtri + 0.5))) - 1
ytri2 = 2 * np.abs(2 * (360 * xtri - np.floor(360 * xtri + 0.5))) - 1

ytri = np.concatenate((ytri1, ytri2))
sd.play(ytri, 48000)
sd.wait()
wav.write('trianglewaves.wav', 48000, ytri)