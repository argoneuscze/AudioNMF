import numpy

from audionmf.transforms.mdct import mdct, imdct


def test_mdct_fast():
    ...


def test_mdct_slow():
    signal = numpy.concatenate((numpy.arange(9) * -1 - 1, numpy.arange(9) + 1))
    mdct_out, padding = mdct(signal, 4, True)
    imdct_out = imdct(mdct_out, padding, True)
    assert numpy.allclose(imdct_out, signal)
