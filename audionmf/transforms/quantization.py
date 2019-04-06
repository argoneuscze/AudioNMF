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


def init_unif_quant(min_val, max_val, levels):
    """ Initializes a uniform quantizer of N levels. """
    val_range = max_val - min_val
    step = val_range / (levels - 1)

    def unif_quant_val(x):
        """ Quantizes a value to the according level. """
        quant_val_idx = round((x - min_val) / step)
        return quant_val_idx

    return unif_quant_val, step
