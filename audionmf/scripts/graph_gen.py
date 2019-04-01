import numpy
from matplotlib import pyplot as plt

from audionmf.transforms.window_func import mdct_window_mp3

time = numpy.arange(0, numpy.pi * 8, 0.1)

window = mdct_window_mp3(len(time) // 2)
amp = numpy.sin(time)
amp = window * amp

plt.plot(time, amp, label='sin(x) * MLT')
plt.plot(time, window, label='MLT')
plt.legend()

plt.xlabel('x')
plt.ylabel('windowed sin(x)')

plt.tight_layout()
plt.savefig('mlt_window.png')
