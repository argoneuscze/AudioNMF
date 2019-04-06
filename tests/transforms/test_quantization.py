import numpy

from audionmf.transforms.quantization import scale_val, mu_law_compand, mu_law_expand, init_unif_quant


def test_scale_val_positive():
    assert scale_val(15, 10, 20, 2, 6) == 4


def test_scale_val_negative():
    assert scale_val(-15, -20, -10, -6, -2) == -4


def test_scale_val_negpos():
    omin = 10
    omax = 20
    nmin = -1
    nmax = 1

    assert scale_val(10, omin, omax, nmin, nmax) == -1
    assert scale_val(15, omin, omax, nmin, nmax) == 0
    assert scale_val(20, omin, omax, nmin, nmax) == 1


def test_mu_law():
    val1 = 0.5
    val2 = -0.5
    val3 = 0

    assert numpy.isclose(mu_law_expand(mu_law_compand(val1, 255), 255), val1)
    assert numpy.isclose(mu_law_expand(mu_law_compand(val2, 255), 255), val2)
    assert numpy.isclose(mu_law_expand(mu_law_compand(val3, 255), 255), val3)


def test_unif_quant_val():
    min_val = 0
    max_val = 1
    levels = 32

    quantize, step = init_unif_quant(min_val, max_val, levels)

    ref_step = max_val / (levels - 1)
    assert ref_step == step

    assert quantize(0) == 0
    assert quantize(1) == 31
    assert quantize(0.671) == 21
