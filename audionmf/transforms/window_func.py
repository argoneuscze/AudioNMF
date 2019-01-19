import numpy


def mdct_window_mp3(N):
    def w(n):
        return numpy.sin((numpy.pi / (2 * N)) * (n + 0.5))

    window = [w(n) for n in range(2 * N)]
    return window
