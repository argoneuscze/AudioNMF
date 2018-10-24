import math

import numpy


class Channel:
    def __init__(self):
        self.samples = numpy.array([], dtype=numpy.int16)

    def add_sample_array(self, data):
        self.samples = numpy.append(self.samples, data)

    def to_positive_matrix(self):
        # naive square matrix
        sample_cnt = self.samples.size
        square_dim = math.ceil(math.sqrt(sample_cnt))
        padding = square_dim ** 2 - sample_cnt

        padded_samples = numpy.pad(self.samples, (0, padding), mode='constant', constant_values=0)
        final_matrix = padded_samples.reshape((square_dim, square_dim))

        # incrementing all values by 2^15, need to convert back
        final_matrix = final_matrix.astype(numpy.int32)
        return final_matrix + 2 ** 15, padding
