import numpy

from audionmf.util.matrix_util import array_pad


def mdct_fast(signal):
    """ Turns 2N inputs into N outputs using MDCT via a modified DCT-IV. """
    ...


def imdct_fast(mdct_ary):
    """ Turns N inputs into 2N outputs using MDCT via a modified DCT-IV. """
    ...


def mdct_slow(signal):
    """ Slow implementation from definition. Do not use in production. """

    N = len(signal) // 2

    output_ary = numpy.ndarray((N,))

    for k in range(N):
        val = 0
        for n in range(2 * N):
            val += signal[n] * numpy.cos((numpy.pi / N) * (n + 0.5 + N / 2) * (k + 0.5))
        output_ary[k] = val

    return output_ary


def imdct_slow(mdct_ary):
    """ Slow implementation from definition. Do not use in production. """

    N = len(mdct_ary)

    output_ary = numpy.ndarray((2 * N,))

    for n in range(2 * N):
        val = 0
        for k in range(N):
            val += mdct_ary[k] * numpy.cos((numpy.pi / N) * (n + 0.5 + N / 2) * (k + 0.5))
        output_ary[n] = (1 / N) * val

    return output_ary


def mdct(full_signal, block_size, slow=False):
    """ Runs overlapping MDCT on a full signal.

     Block size must be an even value.
     """

    # pad samples properly to block size
    samples, padding = array_pad(full_signal, block_size)

    # add an extra block to the start and end to fix the first and last block
    samples = numpy.pad(samples, (block_size, block_size), mode='constant', constant_values=0)

    # allocate MDCT output matrix
    mdct_matrix = numpy.ndarray((len(samples) // block_size, block_size))

    # pick the transform function R^2N -> R^N
    if not slow:
        f = mdct_fast
    else:
        f = mdct_slow

    # run MDCT for each block except the last one
    for i in range((len(samples) // block_size) - 1):
        mdct_ary = f(samples[i * block_size:i * block_size + block_size * 2])
        mdct_matrix[i] = mdct_ary

    return mdct_matrix, padding


def imdct(mdct_matrix, padding, slow=False):
    """ Reverses MDCT, returning the original signal after overlap-and-add. """

    # find block size
    block_size = mdct_matrix.shape[1]

    # allocate IMDCT output array
    imdct_ary = numpy.zeros(mdct_matrix.size)

    # pick the transform function R^2N -> R^N
    if not slow:
        f = imdct_fast
    else:
        f = imdct_slow

    # run IMDCT for each block except last one
    for i in range(mdct_matrix.shape[0] - 1):
        imdct_ary[i * block_size:i * block_size + block_size * 2] += f(mdct_matrix[i])

    return imdct_ary[block_size:-padding - block_size]
