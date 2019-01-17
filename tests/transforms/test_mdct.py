import numpy

from audionmf.transforms.mdct import mdct, imdct


def test_mdct_slow():
    signal = numpy.concatenate((numpy.arange(4) * -1 - 1, numpy.arange(4) + 1))

    mdct_out, padding = mdct(signal, 4, True)
    imdct_out = imdct(mdct_out, padding, True)

    assert numpy.allclose(imdct_out, signal)


def test_mdct_fast():
    signal = numpy.concatenate((numpy.arange(4) * -1 - 1, numpy.arange(4) + 1))

    mdct_fast, padding = mdct(signal, 4, False)
    mdct_slow, padding = mdct(signal, 4, True)

    print(mdct_slow)
    print(mdct_fast)

    assert numpy.allclose(mdct_fast, mdct_slow)

    imdct_fast = imdct(mdct_fast, padding, False)
    imdct_slow = imdct(mdct_slow, padding, True)

    assert numpy.allclose(imdct_fast, imdct_slow)
