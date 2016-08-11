import numpy as np
import theano
import theano.tensor as T
import theano.tensor.nnet as nnet

from csxnet.model import NeuralNetworkBase


class ThNetDynamic(NeuralNetworkBase):
    def __init__(self, data, eta, lmbd1, lmbd2, mu, cost):
        NeuralNetworkBase.__init__(self, data, eta, lmbd1, lmbd2, mu)

        inshape, _ = data.neurons_required()

        if isinstance(inshape, int):
            inshape = (inshape,)

        self.inputs = T.matrix("Inputs") if len(inshape) == 1 else T.tensor4("Inputs")
        self.targets = T.matrix("Targets")
        self.m = T.scalar("m", dtype="float32")
        self.mint = T.scalar("mint", dtype="int32")

        # Final computational graphs go here
        self.cost = cost
        self.output = None
        self.prediction = None

        # Compiled Theno functions will go here
        self._fit = None
        self._evaluate = None
        self._predict = None

        self.age = 0
        self.architecture = []
        self.finalized = False

    def add_convpool(self, conv, filters, pool):
        from layers import ThConvPoolLayer
        pos, fanin = self._layeradd_prepare()

        self.architecture.append("Conv({}x{}x{}); Pool({})".format(filters, conv, conv, pool))
        self.layers.append(ThConvPoolLayer(conv, filters, pool, fanin, pos))

    def add_fc(self, neurons, activation="sigmoid"):
        from layers import ThFCLayer
        if activation == "softmax":
            if neurons != self.outsize:
                print("Warning! Assumed creation of output layer, but got wrong number of neurons!")
                print("Adjusting neuron number to correct output size:", self.outsize)
            self.finalize()

        pos, fanin = self._layeradd_prepare()

        self.architecture.append("FC({}): {}".format(neurons, activation[:4]))
        self.layers.append(ThFCLayer(neurons, fanin, pos, activation))

    def add_rlayer(self, neurons):
        from .layers import ThRLayer
        pos, fanin = self._layeradd_prepare()

        self.architecture.append("Rlayer({}): {}".format(neurons, "tanh"))
        self.layers.append(ThRLayer(neurons, fanin, pos))

    def add_lstm(self, neurons):
        from .layers import ThLSTM
        pos, fanin = self._layeradd_prepare()

        self.architecture.append("LSTM({}): {}".format(neurons, "tanh"))
        self.layers.append(ThLSTM(neurons, fanin, pos))

    def _layeradd_prepare(self):
        pos = len(self.layers)
        fanin = self.fanin if not pos else self.layers[-1].outshape
        return pos, fanin

    def finalize(self):
        from .layers import ThOutputLayer

        def define_output_layer():
            pos, fanin = self._layeradd_prepare()
            self.architecture.append("Out({}): {}".format(self.outsize, "softmax"))
            return ThOutputLayer(self.outsize, fanin, pos)

        def define_feedforward():
            self.mint = self.m.astype("int32")
            forward_pass_rules = [self.inputs]
            for layer in self.layers:
                forward_pass_rules.append(layer.output(forward_pass_rules[-1], self.mint))
            return forward_pass_rules[-1]

        def define_cost():
            # Define regularization terms
            # l1 = sum([T.abs_(layer.weights).sum() for layer in self.layers])
            # l1 *= self.lmbd1 / (self.data.N / 2)
            # l2 = sum([T.exp2(layer.weights).sum() for layer in self.layers])
            # l2 *= self.lmbd2 / (self.data.N * 2)

            # Build string for architecture display
            chain = "Cost: " + self.cost
            reg = ""
            if self.lmbd1 or self.lmbd2:
                reg += " + "
            if self.lmbd1:
                reg += "L1"
                if self.lmbd2:
                    reg += " + L2 reg."
                else:
                    reg += " reg."
            if self.lmbd2:
                reg += "L2 reg."
            self.architecture.append(chain + reg)

            # Define cost function
            if self.cost.lower() == "xent":
                cost = nnet.categorical_crossentropy(
                    coding_dist=self.output,
                    true_dist=self.targets).sum()
            elif self.cost.lower() == "mse":
                cost = T.exp2(self.targets - self.output).sum()
            else:
                raise RuntimeError("Cost function {} not supported!".format(self.cost))

            # cost = cost + l1 + l2

            return cost

        def define_update_rules():
            rules = []
            for params in [layer.params for layer in self.layers]:
                rules.extend([(param, param - (self.eta / self.m) * T.grad(self.cost, param))
                              for param in params])
            return rules

        self.layers.append(define_output_layer())
        self.output = define_feedforward()
        self.cost = define_cost()
        self.prediction = T.argmax(self.output, axis=1)
        updates = define_update_rules()

        self._fit = theano.function(inputs=[self.inputs, self.targets, self.m],
                                    updates=updates,
                                    name="_fit")

        self._evaluate = theano.function(inputs=[self.inputs, self.targets, self.m],
                                         outputs=[self.cost, self.prediction],
                                         name="_evaluate")
        self._predict = theano.function(inputs=[self.inputs, self.m],
                                        outputs=[self.prediction],
                                        name="_predict")
        self.finalized = True

    def learn(self, batch_size, verbose=False):
        if not self.finalized:
            raise RuntimeError("Unfinalized network!")
        for i, (questions, targets) in enumerate(self.data.batchgen(batch_size)):
            m = float(questions.shape[0])
            self._fit(questions, targets, m)
            if verbose:
                print("Done {} batches!".format(i))
        self.age += 1

    def evaluate(self, on="tesing"):
        if not self.finalized:
            raise RuntimeError("Unfinalized network!")
        m = self.data.n_testing
        tab = self.data.table(on)
        qst, idp = tab[0][:m], tab[1][:m]
        cost, pred = self._evaluate(qst, idp, m)
        acc = np.mean(np.equal(np.argmax(idp, axis=1), pred))
        return cost, acc

    def predict(self, questions: np.ndarray):
        m = questions.shape[0]
        return self._predict(questions, m)

    def describe(self, verbose=1):
        chain = "---------------\n"
        chain += "Dynamic Theano-based Artificial Neural Network.\n"
        chain += "Age: " + str(self.age) + "\n"
        chain += "Architecture: " + "; ".join(self.architecture) + "\n"
        if not self.finalized:
            chain += "!!! UNFINALIZED !!!"
        chain += "---------------"
        if verbose:
            print(chain)
        else:
            return chain
