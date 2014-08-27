from libpgm.nodedata import NodeData
from libpgm.graphskeleton import GraphSkeleton
from libpgm.discretebayesiannetwork import DiscreteBayesianNetwork

from inference.exact_inference import ExactInferenceEngine
from inference.approximate_inference import ApproximateInferenceEngine

node_data = NodeData()
network_skeleton = GraphSkeleton()
node_data.load('network.txt')
network_skeleton.load('network.txt')
network = DiscreteBayesianNetwork(network_skeleton, node_data)

exact_inference_engine = ExactInferenceEngine(network)
approximate_inference_engine = ApproximateInferenceEngine(network)

query_variable = 'Burglary'
evidence_variables = {'MaryCalls': 'true', 'JohnCalls': 'true'}
resulting_distribution = exact_inference_engine.perform_inference(query_variable, evidence_variables)
print 'P(B|m,j) - enumeration: ', resulting_distribution
resulting_distribution = exact_inference_engine.perform_ve_inference(query_variable, evidence_variables)
print '(B|m,j) - variable elimination: ', resulting_distribution
resulting_distribution = approximate_inference_engine.perform_rs_inference(query_variable, evidence_variables, 100000)
print 'P(B|m,j) - approximate - rejection sampling: ', resulting_distribution
resulting_distribution = approximate_inference_engine.perform_lw_inference(query_variable, evidence_variables, 100000)
print 'P(B|m,j) - approximate - likelihood weighting: ', resulting_distribution
resulting_distribution = approximate_inference_engine.perform_gibbs_inference(query_variable, evidence_variables, 100000)
print 'P(B|m,j) - approximate - Gibbs: ', resulting_distribution
print

query_variable = 'JohnCalls'
evidence_variables = {'MaryCalls': 'true'}
resulting_distribution = exact_inference_engine.perform_inference(query_variable, evidence_variables)
print 'P(j|m) - enumeration: ', resulting_distribution
resulting_distribution = exact_inference_engine.perform_ve_inference(query_variable, evidence_variables)
print '(j|m) - variable elimination: ', resulting_distribution
resulting_distribution = approximate_inference_engine.perform_rs_inference(query_variable, evidence_variables, 100000)
print 'P(j|m) - approximate - rejection sampling: ', resulting_distribution
resulting_distribution = approximate_inference_engine.perform_lw_inference(query_variable, evidence_variables, 100000)
print 'P(j|m) - approximate - likelihood weighting: ', resulting_distribution
resulting_distribution = approximate_inference_engine.perform_gibbs_inference(query_variable, evidence_variables, 100000)
print 'P(j|m) - approximate - Gibbs: ', resulting_distribution