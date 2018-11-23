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
