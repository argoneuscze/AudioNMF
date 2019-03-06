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
