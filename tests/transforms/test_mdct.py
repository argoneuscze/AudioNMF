import numpy

from audionmf.transforms.mdct import mdct, imdct


def test_mdct_slow():
    x = numpy.linspace(0, 8 * numpy.pi, 1000)
    signal = [numpy.cos(y) for y in x]

    mdct_out, padding = mdct(signal, 2, True)
    imdct_out = imdct(mdct_out, padding, True)

    assert numpy.allclose(imdct_out, signal)


def test_mdct_fast():
    x = numpy.linspace(0, 8 * numpy.pi, 1000)
    signal = [numpy.cos(y) for y in x]

    mdct_out, padding = mdct(signal, 2, False)
    imdct_out = imdct(mdct_out, padding, False)

    assert numpy.allclose(imdct_out, signal)


def test_mdct_both():
    x = numpy.linspace(0, 8 * numpy.pi, 1000)
    signal = [numpy.cos(y) for y in x]

    mdct_fast, padding = mdct(signal, 4, False)
    mdct_slow, padding = mdct(signal, 4, True)

    assert numpy.allclose(mdct_fast, mdct_slow)

    imdct_fast = imdct(mdct_fast, padding, False)
    imdct_slow = imdct(mdct_slow, padding, True)

    assert numpy.allclose(imdct_fast, imdct_slow)
