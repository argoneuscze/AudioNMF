import numpy

from audionmf.transforms.nmf import NMF
from audionmf.util.matrix_util import increment_by_min


def nmf_matrix(matrix, max_iter=100, rank=30):
    # increment the matrix to make sure it's positive
    matrix_inc, min_val = increment_by_min(matrix)

    # TODO save
    # use Kullback-Leibler divergence
    # nmf = nimfa.Nmf(matrix_inc, max_iter=max_iter, rank=rank, objective='div', update='divergence')()
    # W = nmf.basis()
    # H = nmf.coef()

    # calculate NMF
    nmf = NMF(matrix, max_iter=max_iter, rank=rank)
    W, H = nmf.factorize()

    return W, H, min_val


def nmf_matrix_original(W, H, min_val):
    # get the original matrix
    matrix = numpy.matmul(W, H) - min_val

    return matrix
