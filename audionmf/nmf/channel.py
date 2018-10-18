import numpy


class Channel:
    def __init__(self):
        self.samples = list()

    def add_sample_array(self, data):
        self.samples = numpy.append(self.samples, data)

    def to_matrix(self):
        return numpy.asarray(self.samples)
        # TODO make 2d matrix and return actual size minus padding, use sqrt
