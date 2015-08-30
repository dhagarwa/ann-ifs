import sys
sys.path.append('/users/math/bs/dhwanit/local/lib/python2.6/site-packages')
from pybrain.datasets import SupervisedDataSet
import numpy, math
import pylab
from pybrain.structure import SigmoidLayer, LinearLayer
from pybrain.tools.shortcuts import buildNetwork
from pybrain.supervised.trainers import BackpropTrainer
from random import randint

#Predetermined constants
start = 0
end = 4 * math.pi
subintervals = 1001
N = 4 
step = (end - start)/ N
maxIter = 10000

#Create sample data

def F(x):
	#return x**2
	return math.sin(x)
	
	
	
xvalues = numpy.linspace(start, end, subintervals)
#yvalues = 5 * numpy.sin(xvalues)
yvalues = [F(x) for x in xvalues]
zvalues = [F(x) + 1 for x in xvalues]


#Display neural network weights

'''def disp_network(n):
	i = 0
	w = [[0 for x in range(5)] for x in range(2)]
	b = [0 for x in range(5)]
	v = [0 for x in range(6)]
	
	for mod in n.modules:
		for conn in n.connections[mod]:
			for cc in range(len(conn.params)):
				if i < 10:
					if i % 2 == 0:
						w[0][int(i/2)] = conn.params[cc]
					else:
						w[1][int(i/2)] = conn.params[cc]
						
				elif i < 15:
					v[i - 9] = conn.params[cc]
					
				elif i < 16:
					v[0] = conn.params[cc]
					
				elif i < 21:
					b[i - 16] = conn.params[cc]
					
                i += 1
	
	#print [w, b, v]
	return [w, b, v]'''

#Function to save appropriate contraction map i.e l_{n} and neural network
list_of_maps = []

def calculate_a_e(left, right):
	a = (right - left)/ (end - start)
	e = (end*left - start*right) / (end - start)
	return [a, e]
	
	
def add_map(left, right, net):
	a = (right - left)/ (end - start)
	e = (end*left - start*right) / (end - start)
	list_of_maps.append([a, e, net])




#Function to check if a neural network is contractive
'''def is_contractive(net):
	weights = disp_network(net)
	w = weights[0]
	b = weights[1]
	v = weights[2]
	sum = 0
	for i in range(1, 6):
		sum += abs(v[i]* w[1][i-1])
		
	if sum < 4:
		return True
	else:
		return False'''


#Function to plot attractor
def plot_attractor(contraction_maps):
	num = len(contraction_maps)
	xlist = []
	ylist = []
	x = 0
	y = 0
	for i in range(maxIter):
		r = randint(0,num-1)
		a = contraction_maps[r][0]
		e = contraction_maps[r][1]
		net = contraction_maps[r][2]
		x = a * x + e
		y = net.activate([x, y])
		xlist.append(x)
		ylist.append(y)
		#print "plotting point"
		
	pylab.plot(xlist, ylist, 'ro')	

#Start training neural networks 

left = start
right = start + step

while right <= end:

	param = calculate_a_e(left, right)
	a = param[0]
	e = param[1]
	zvalues = []
	for x in xvalues:
		zvalues.append(F(a*x + e))
		
	ds = SupervisedDataSet(2, 1)
	for x, y, z in zip(xvalues, yvalues, zvalues):
		ds.addSample((x, y), (z,))		
	
	net = buildNetwork(2,
					   5, # number of hidden units
					   1
					   )
	#----------
	# train
	#----------
	trainer = BackpropTrainer(net, ds, verbose=False)
	trainer.trainUntilConvergence(maxEpochs = 20)


	if is_contractive(net):
		print 'Yippee! Contractive'
		add_map(left, right, net)
		left = right
		right = right + step
		
	else:
		print 'Not contractive'
		right = right + step

#print list_of_maps

#target function
pylab.plot(xvalues, yvalues, linewidth = 2, color = 'blue', label = 'target')
           
#plot attractor
plot_attractor(list_of_maps)
pylab.grid()
pylab.legend()
pylab.show()
