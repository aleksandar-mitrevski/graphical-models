import numpy
from learning.mode_graph import ModeGraph
import matplotlib.pyplot as plt

import json

from libpgm.nodedata import NodeData
from libpgm.graphskeleton import GraphSkeleton
from libpgm.dyndiscbayesiannetwork import DynDiscBayesianNetwork

from inference.sensor_dbn_inference import SensorDbnInference

# test_data = numpy.load('c_means_test_data.npy')
# graph_manager = ModeGraph()
# means, memberships = graph_manager.learn_graph(test_data, 3, 2, 0.1, 100)
# max_membership_indices = numpy.argmax(memberships, axis=1)

# cluster_1 = numpy.where(max_membership_indices==0)[0]
# cluster_2 = numpy.where(max_membership_indices==1)[0]
# cluster_3 = numpy.where(max_membership_indices==2)[0]

# figure = plt.figure(1)
# figure.add_subplot(1,2,1)
# plt.scatter(test_data[:,0], test_data[:,1])
# figure.add_subplot(1,2,2)
# plt.scatter(test_data[cluster_1,0], test_data[cluster_1,1], c= 'r', marker='o')
# plt.scatter(test_data[cluster_2,0], test_data[cluster_2,1], c='b', marker='o')
# plt.scatter(test_data[cluster_3,0], test_data[cluster_3,1], c='k', marker='o')
# plt.scatter(means[:,0], means[:,1], c='c', marker='o')
# plt.show()

means = numpy.array([[  2.00755688e-04,   1.65181639e-01], [  8.37884753e-01,   9.99778286e-01], [  9.75317567e-01,   2.46051178e-02]])

# sensor network
sensor_network_file = open('test_bayesian_networks/graph_sensor_dbn.txt', 'r')
sensor_network_file_data = eval(sensor_network_file.read())

sensor_network_skeleton = GraphSkeleton()
sensor_network_skeleton.V = sensor_network_file_data["V"]
sensor_network_skeleton.E = sensor_network_file_data["E"]

sensor_network = DynDiscBayesianNetwork()
sensor_network.V = sensor_network_skeleton.V
sensor_network.E = sensor_network_skeleton.E
sensor_network.initial_Vdata = sensor_network_file_data["initial_Vdata"]
sensor_network.twotbn_Vdata = sensor_network_file_data["twotbn_Vdata"]

# observation_network
observation_network_file = open('test_bayesian_networks/graph_transition_dbn.txt', 'r')
observation_network_file_data = eval(observation_network_file.read())

observation_network_skeleton = GraphSkeleton()
observation_network_skeleton.V = observation_network_file_data["V"]
observation_network_skeleton.E = observation_network_file_data["E"]

observation_network = DynDiscBayesianNetwork()
observation_network.V = observation_network_skeleton.V
observation_network.E = observation_network_skeleton.E
observation_network.initial_Vdata = observation_network_file_data["initial_Vdata"]
observation_network.twotbn_Vdata = observation_network_file_data["twotbn_Vdata"]

sensor_inference_engine = SensorDbnInference(sensor_network)
obs_inference_engine = SensorDbnInference(observation_network)

previous_observation = observation_network.initial_Vdata['state']['vals'][numpy.argmax(observation_network.initial_Vdata['state']['cprob'])]
transition = observation_network.initial_Vdata['state']['vals'][numpy.argmax(observation_network.initial_Vdata['state']['cprob'])]
state = numpy.zeros(2)
state[0] = numpy.random.randint(0,2)
print sensor_inference_engine.get_current_belief()
for i in xrange(1,10):
    state[1] = numpy.random.randint(0,2)
    print 'Measurements -> ', state
    cluster = numpy.argmin(numpy.apply_along_axis(numpy.linalg.norm, 1, state - means))
    obs_inference_engine.filter(str(cluster))
    transition = transition + observation_network.initial_Vdata['state']['vals'][numpy.argmax(observation_network.initial_Vdata['state']['cprob'])]
    sensor_inference_engine.filter(transition)

    print 'Likely transition -> ', transition
    print sensor_inference_engine.get_current_belief()
    print

    transition = '' + observation_network.initial_Vdata['state']['vals'][numpy.argmax(observation_network.initial_Vdata['state']['cprob'])]
    state[0] = state[1]