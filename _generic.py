import theano
import theano.tensor as T
import theano.tensor.nnet as nnet


floatX = theano.config.floatX


class _CostBase:
    def __init__(self, graph, l1_term, l2_term):
        self.graph = graph + l1_term + l2_term

    def __call__(self, outputs, targets):
        return self.graph(outputs, targets)

    def derivative(self, wrt):
        return T.grad(self.graph, wrt=wrt)


class Xent_cost(_CostBase):
    def __init__(self, outputs, targets, l1_term, l2_term):
        graph = nnet.categorical_crossentropy(outputs, targets)
        _CostBase.__init__(self, graph, l1_term, l2_term)


class MSE_cost(_CostBase):
    def __init__(self, outputs, targets, l1_term, l2_term):
        graph = T.exp2(outputs - targets)
        _CostBase.__init__(self, graph, l1_term, l2_term)


class NLL_cost(_CostBase):
    def __init__(self, outputs, targets, l1_term, l2_term):
        graph = -T.sum(T.log(outputs)[T.arange(targets.shape[0]), targets])
        _CostBase.__init__(self, graph, l1_term, l2_term)
