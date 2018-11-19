import numpy


class Channel:
    def __init__(self):
        self.samples = numpy.array([], dtype=numpy.int16)

    def add_sample_array(self, data):
        self.samples = numpy.append(self.samples, data)
