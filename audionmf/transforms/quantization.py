import numpy


def scale_val(x, old_min, old_max, new_min, new_max):
    old_range = abs(old_max - old_min)
    new_range = abs(new_max - new_min)
    val = (((x - old_min) / old_range) * new_range) + new_min
    return val


def mu_law_compand(x, mu=255):
    """ Assumes values -1 <= x <= 1 """
    return numpy.sign(x) * numpy.log(1 + mu * abs(x)) / numpy.log(1 + mu)


def mu_law_expand(y, mu=255):
    """ Assumes values -1 <= y <= 1 """
    return numpy.sign(y) * (1 / mu) * (((1 + mu) ** abs(y)) - 1)


class UniformQuantizer:
    """ A uniform quantizer of N levels. """

    def __init__(self, min_val, max_val, levels):
        val_range = max_val - min_val
        self.step = val_range / (levels - 1)
        self.min_val = min_val

    def quantize_value(self, x):
        """ Quantizes a value to the according level. """
        quant_val_idx = numpy.round((x - self.min_val) / self.step)
        return int(quant_val_idx)

    def dequantize_index(self, idx):
        """ Returns the original value based on the index. """
        val = self.min_val + idx * self.step
        return val
