import math

import numpy


def serialize_matrix(fd, matrix):
    ...  # todo write to open file


def deserialize_matrix(fd):
    ...


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
