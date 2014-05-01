from libpgm.nodedata import NodeData
from libpgm.graphskeleton import GraphSkeleton
from libpgm.discretebayesiannetwork import DiscreteBayesianNetwork

from inference.exact_inference import ExactInferenceEngine

node_data = NodeData()
network_skeleton = GraphSkeleton()
node_data.load('network.txt')
network_skeleton.load('network.txt')
network = DiscreteBayesianNetwork(network_skeleton, node_data)

inference_engine = ExactInferenceEngine(network)
query_variable = 'Burglary'
evidence_variables = {'MaryCalls': 'true', 'JohnCalls': 'true'}
resulting_distribution = inference_engine.perform_inference(query_variable, evidence_variables)
print resulting_distribution

query_variable = 'JohnCalls'
evidence_variables = {'MaryCalls': 'true'}
resulting_distribution = inference_engine.perform_inference(query_variable, evidence_variables)
print resulting_distribution