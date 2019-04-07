import numpy

from audionmf.transforms.quantization import scale_val, mu_law_compand, mu_law_expand, UniformQuantizer


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

    quantizer = UniformQuantizer(min_val, max_val, levels)

    assert quantizer.quantize_value(0) == 0
    assert quantizer.quantize_value(1) == 31
    assert quantizer.quantize_value(0.671) == 21


def test_unif_quant_matrix():
    matrix = numpy.array([
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9]
    ])

    quantizer = UniformQuantizer(1, 9, 64)
    quantize_vec = numpy.vectorize(quantizer.quantize_value)
    dequantize_vec = numpy.vectorize(quantizer.dequantize_index)

    quant_matrix = quantize_vec(matrix)
    mult_matrix = dequantize_vec(quant_matrix)

    assert numpy.allclose(matrix, mult_matrix, atol=0.2)
