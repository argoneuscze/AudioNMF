import struct

import numpy


def serialize_matrix(fd, matrix, dtype='f'):
    dt = numpy.dtype(dtype)
    matrix = matrix.astype(dt)
    fd.write(struct.pack('<II', matrix.shape[0], matrix.shape[1]))
    fd.write(struct.pack('<' + dtype * matrix.size, *matrix.flat))


def deserialize_matrix(fd, dtype='f'):
    dt = numpy.dtype(dtype)
    dt = dt.newbyteorder('<')
    rows, cols = struct.unpack('<II', fd.read(8))
    matrix = numpy.frombuffer(fd.read(rows * cols * dt.itemsize), dtype=dt)
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


def matrix_split(matrix, N):
    """ Splits an X*Y matrix into X // N chunks sized N*Y without any padding at the end. """
    # if N is None, return the original matrix as one chunk
    if N is None:
        return [matrix]
    submatrices = list()
    for i in range((matrix.shape[0] // N) + 1):
        submatrix = matrix[i * N:i * N + N]
        submatrices.append(submatrix)
    return submatrices


def increment_by_min(matrix):
    # increments matrix by its lowest value and returns the structure and the absolute value
    min_val = abs(numpy.amin(matrix))
    return matrix + min_val, min_val
