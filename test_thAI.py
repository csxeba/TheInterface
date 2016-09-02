import sys

from TheInterface.thmodel import ThNetDynamic
from csxdata import CData
from csxdata.utilities.parsers import mnist_tolearningtable as mnist_to_lt

dataroot = "D:/Data/" if sys.platform == "win32" else "/data/Prog/data/"
miscroot = dataroot + "misc/"
miscpath = miscroot + "mnist.pkl.gz"


def teach(net, epochs=10, bsize=10):
    tacs, lacs = [], []
    for epoch in range(1, epochs+1):
        net.learn(bsize)
        tacc = net.evaluate("testing")
        lacc = net.evaluate("learning")
        tacs.append(tacc)
        lacs.append(lacc)
        print("E: {} T: {} L: {}".format(epoch, tacc, lacc))

print("Pulling data")
data = CData(mnist_to_lt(miscpath))
data.transformation = "std"

print("Building network")
thnet = ThNetDynamic(data, 0.1, 0.0, 5.0, 0.0, "xent")
# csxnet = Network(data, 0.1, 0.0, 0.0, 0.9, "xent")
del data

thnet.add_fc(60, activation="tanh")
thnet.finalize()
print("Network built!")
thnet.describe(1)

# csxnet.add_fc(60)
# csxnet.finalize_architecture()
# print("Network built!")
# csxnet.describe(1)

print("Staring learning phase...")
teach(thnet, 30)


