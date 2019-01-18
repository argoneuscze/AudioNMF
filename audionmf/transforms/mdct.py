import numpy
from scipy import fftpack

from audionmf.util.matrix_util import array_pad


def mdct_fast(mdct_ary):
    """ Turns 2N inputs into N outputs using MDCT via a DCT-IV. """

    # split into fourths
    half_block_size = mdct_ary.size // 4

    # create blocks for MDCT
    # the MDCT of 2N inputs (a, b, c, d) is exactly equivalent to a DCT-IV of the N inputs: (−cR−d, a−bR)
    a = mdct_ary[:half_block_size]
    bR = mdct_ary[half_block_size:2 * half_block_size][::-1]
    cR = mdct_ary[2 * half_block_size:3 * half_block_size][::-1]
    d = mdct_ary[3 * half_block_size:4 * half_block_size]

    # fill input array
    dct_input = numpy.concatenate((-cR - d, a - bR))

    # run DCT-IV on this array of size N, producing effectively MDCT of size 2N
    dct4 = fftpack.dct(dct_input, type=4)

    return dct4 * 0.5  # TODO fix constant


def imdct_fast(mdct_ary):
    """ Turns N inputs into 2N outputs using MDCT via a DCT-IV. """

    # get a size of fourths
    half_block_size = mdct_ary.size // 2

    # allocate output array
    output_ary = numpy.ndarray((half_block_size * 4,))

    # inverse DCT-IV, obtaining back (−cR−d, a−bR)
    idct4 = fftpack.idct(mdct_ary, type=4)

    # divide by implicit scaling factor 2N
    idct4 /= 4 * half_block_size

    # extend and shift to gain the (almost) original array - still need to overlap and add with the next block
    abR = idct4[half_block_size:]
    output_ary[:half_block_size] = abR
    output_ary[half_block_size:2 * half_block_size] = -abR[::-1]

    cRd = idct4[:half_block_size]
    output_ary[2 * half_block_size:3 * half_block_size] = -cRd[::-1]
    output_ary[3 * half_block_size:] = -cRd

    return output_ary


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
    mdct_matrix = numpy.ndarray(((len(samples) // block_size) - 1, block_size))

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

    # allocate IMDCT output array along with an extra block filler at the end
    imdct_ary = numpy.zeros(mdct_matrix.size + block_size)

    # pick the transform function R^N -> R^2N
    if not slow:
        f = imdct_fast
    else:
        f = imdct_slow

    # run IMDCT for each block
    for i in range(mdct_matrix.shape[0]):
        imdct_ary[i * block_size:i * block_size + block_size * 2] += f(mdct_matrix[i])

    # remove the padding from the array and return it
    return imdct_ary[block_size:-padding - block_size]
