import numpy
from boltzmann_machines.trbm import TRBM

data = []
for i in xrange(1000):
    vector = []
    for j in xrange(10):
        measurement = numpy.random.randint(0,2)
        vector.append(measurement)
    data.append(vector)

data = numpy.array(data)
network = TRBM(10,15)
network.train(data, epochs=10, learning_rate=0.1)

#test data
data = []
for i in xrange(11):
    vector = []
    for j in xrange(10):
        measurement = numpy.random.randint(0,2)
        vector.append(measurement)
    data.append(vector)
data = numpy.array(data)

initial_data = numpy.zeros((10,1))
for i in xrange(10):
    initial_data[i] = data[0,i]
sample = network.sample_network(data[1,:], initial_data)
print sample

for i in xrange(1,11):
    sample = network.sample_network(data[2,:], initial_data)
    print sample