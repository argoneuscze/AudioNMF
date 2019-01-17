import math
import struct

import numpy


def serialize_matrix(fd, matrix):
    fd.write(struct.pack('<II', matrix.shape[0], matrix.shape[1]))
    fd.write(struct.pack('<' + 'd' * matrix.size, *matrix.flat))


def deserialize_matrix(fd):
    dt = numpy.dtype(numpy.float64)
    dt = dt.newbyteorder('<')
    rows, cols = struct.unpack('<II', fd.read(8))
    matrix = numpy.frombuffer(fd.read(rows * cols * 8), dtype=dt)
    matrix = numpy.reshape(matrix, newshape=(rows, cols))
    return matrix


def array_pad(ary, n):
    padding = (-len(ary)) % n
    new_ary = numpy.pad(ary, (0, padding), mode='constant', constant_values=0)
    return new_ary, padding


def array_pad_split(ary, n):
    padding = (-len(ary)) % n
    new_ary = numpy.pad(ary, (0, padding), mode='constant', constant_values=0)
    ary_count = len(new_ary) // n
    split_ary = numpy.split(new_ary, ary_count)
    return split_ary, padding


def array_to_positive_matrix(ary):
    # convert 1D array into a square 2D matrix, pad appropriately with 0s
    sample_cnt = ary.size
    square_dim = math.ceil(math.sqrt(sample_cnt))
    padding = square_dim ** 2 - sample_cnt

    padded_samples = numpy.pad(ary, (0, padding), mode='constant', constant_values=0)
    final_matrix = padded_samples.reshape((square_dim, square_dim))

    # incrementing all values by 2^15, need to have a large enough datatype
    final_matrix = final_matrix.astype(numpy.int32) + 2 ** 15

    # return matrix and padding
    return final_matrix, padding


def array_to_fft(array):
    # runs RFFT on an array and returns the real and imaginary arrays separately
    fft = numpy.fft.rfft(array)

    # we get FFT_SIZE / 2 + 1 samples instead of FFT_SIZE / 2, but the first and last
    # only have real components, so we can put the last one's real component
    # into the first's complex component to save a value
    fft[0] += fft[-1].real * 1j

    # remove the last element altogether
    fft = fft[:-1]

    # return the arrays
    return fft.real, fft.imag


def fft_to_array(real_ary, imag_ary):
    # reverses the process above
    # join the arrays back together
    fft_ary = real_ary + imag_ary * 1j

    # fix the values
    fft_ary = numpy.append(fft_ary, fft_ary[0].imag)
    fft_ary[0] = fft_ary[0].real

    # run inverse FFT
    ifft = numpy.fft.irfft(fft_ary)

    # return the original array
    return ifft


def increment_by_min(matrix):
    # increments matrix by its lowest value and returns the structure and the absolute value
    min_val = abs(numpy.amin(matrix))
    return matrix + min_val, min_val
