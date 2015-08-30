#----------
# build the dataset
#----------
import sys
sys.path.append('/users/math/ms/dhwanit/local/lib/python2.6/site-packages')
from pybrain.datasets import SupervisedDataSet
import numpy, math
import pylab


def disp_network(n):
    for mod in n.modules:
        for conn in n.connections[mod]:
            print conn
            for cc in range(len(conn.params)):
                print conn.whichBuffers(cc), conn.params[cc]

xvalues = numpy.linspace(0,2 * math.pi, 1001)
yvalues = 5 * numpy.sin(xvalues)

ds = SupervisedDataSet(1, 1)
for x, y in zip(xvalues, yvalues):
    ds.addSample((x,), (y,))

#----------
# build the network
#----------
from pybrain.structure import SigmoidLayer, LinearLayer
from pybrain.tools.shortcuts import buildNetwork

net = buildNetwork(1,
                   3, # number of hidden units
                   1
                   )
#----------
# train
#----------
from pybrain.supervised.trainers import BackpropTrainer
trainer = BackpropTrainer(net, ds, verbose=True)
trainer.trainUntilConvergence(maxEpochs = 20)
print 'Network details:'
disp_network(net)
print net.params
#----------
# evaluate
#----------

# neural net approximation
pylab.plot(xvalues,
           [ net.activate([x]) for x in xvalues ], linewidth = 2,
           color = 'blue', label = 'NN output')

# target function
pylab.plot(xvalues,
           yvalues, linewidth = 2, color = 'red', label = 'target')

pylab.grid()
pylab.legend()
pylab.show()
