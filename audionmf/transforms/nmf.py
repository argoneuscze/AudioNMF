import numpy


class NMFError(Exception):
    pass


class NMF:
    def __init__(self, matrix, max_iter=100, rank=30, initialize='random', cost_func='euclidean', update='euclidean'):
        self.V = matrix
        self.W = numpy.zeros((matrix.shape[0], rank))
        self.H = numpy.zeros((rank, matrix.shape[1]))
        self.initialize = self.init_func[initialize]
        self.eval_cost = self.cost_func[cost_func]
        self.update = self.update_func[update]
        self.max_iter = max_iter

        self.validate()

    def validate(self):
        """ Makes sure the matrix can be factorized. """
        if self.W.shape[1] >= min(self.V.shape):
            raise NMFError('Rank larger than the original dimensions.')
        if numpy.amin(self.V) < 0:
            raise NMFError('Original matrix contains negative values.')

    def factorize(self):
        """ Factorizes the matrix and returns W (basis) and H (coefficients) """
        self.initialize(self)
        last_cost = None
        for i in range(self.max_iter):
            self.update(self)
            cost = self.eval_cost(self)
            if cost == last_cost:
                break
            last_cost = cost
        return self.W, self.H

    def init_random(self):
        """ Randomly initializes both matrices with a uniform distribution between [0, max(original)). """
        W = numpy.random.rand(*self.W.shape)
        H = numpy.random.rand(*self.H.shape)
        max_val = numpy.amax(self.V)
        W *= max_val
        H *= max_val
        self.W = W
        self.H = H

    def cost_euclidean(self):
        current = numpy.matmul(self.W, self.H)
        return numpy.linalg.norm(self.V - current)

    def update_euclidean(self):
        W = self.W
        H = self.H
        V = self.V
        m = numpy.matmul

        W = W * ((m(V, numpy.transpose(H))) / (m(m(W, H), numpy.transpose(H))))
        H = H * ((m(numpy.transpose(W), V)) / (m(m(numpy.transpose(W), W), H)))

        self.W = W
        self.H = H

    init_func = {
        'random': init_random
    }

    cost_func = {
        'euclidean': cost_euclidean
    }

    update_func = {
        'euclidean': update_euclidean
    }
