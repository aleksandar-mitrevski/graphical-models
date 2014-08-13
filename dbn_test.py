import json

from libpgm.nodedata import NodeData
from libpgm.graphskeleton import GraphSkeleton
from libpgm.dyndiscbayesiannetwork import DynDiscBayesianNetwork

from inference.sensor_dbn_inference import SensorDbnInference

network_file = open('sensor_dbn.txt', 'r')
network_file_data = eval(network_file.read())

network_skeleton = GraphSkeleton()
network_skeleton.V = network_file_data["V"]
network_skeleton.E = network_file_data["E"]

network = DynDiscBayesianNetwork()
network.V = network_skeleton.V
network.E = network_skeleton.E
network.initial_Vdata = network_file_data["initial_Vdata"]
network.twotbn_Vdata = network_file_data["twotbn_Vdata"]

inference_engine = SensorDbnInference(network)
print 'Initial belief: ', inference_engine.get_current_belief()
inference_engine.filter('1')
print 'Measurement = 1: ', inference_engine.get_current_belief()
inference_engine.filter('0')
print 'Measurement = 0: ', inference_engine.get_current_belief()
inference_engine.filter('0')
print 'Measurement = 0: ', inference_engine.get_current_belief()