import numpy
from boltzmann_machines.dbn import DBN
from boltzmann_machines.dbn_continuous import DBNContinuous

data = []
for i in xrange(1000):
    vector = []
    for j in xrange(10):
        measurement = numpy.random.randint(0,2)
        vector.append(measurement)
    data.append(vector)

data = numpy.array(data)
network = DBN(10,[15])
network.train(data, epochs=10, learning_rate=0.1)

network = DBNContinuous(10,[15])
network.train(data, epochs=10, learning_rate=0.1)