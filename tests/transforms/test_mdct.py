import numpy

from audionmf.transforms.mdct import mdct, imdct
from audionmf.util.plot_util import plot_signal


def test_mdct_slow():
    x = numpy.linspace(0, 8 * numpy.pi, 1000)
    signal = [numpy.cos(y) for y in x]

    # signal = numpy.concatenate((numpy.arange(4) * -1 - 1, numpy.arange(4) + 1))

    mdct_out, padding = mdct(signal, 16, True)
    imdct_out = imdct(mdct_out, padding, True)

    plot_signal(signal, "signal.png")
    plot_signal(mdct_out, "mdct.png")

    assert numpy.allclose(imdct_out, signal)


def test_mdct_fast():
    x = numpy.linspace(0, 8 * numpy.pi, 1000)
    signal = [numpy.cos(y) for y in x]

    #signal = numpy.concatenate((numpy.arange(4) * -1 - 1, numpy.arange(4) + 1))

    mdct_fast, padding = mdct(signal, 4, False)
    mdct_slow, padding = mdct(signal, 4, True)

    plot_signal(mdct_fast, "mdct_fast.png")
    plot_signal(mdct_slow, "mdct_slow.png")

    print(mdct_slow)
    print(mdct_fast)

    assert numpy.allclose(mdct_fast, mdct_slow)

    imdct_fast = imdct(mdct_fast, padding, False)
    imdct_slow = imdct(mdct_slow, padding, True)

    assert numpy.allclose(imdct_fast, imdct_slow)
