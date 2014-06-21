import numpy
from decision.mdp import MDP

world = numpy.array([[False, False, False, False], [False, True, False, False], [False, False, False, False]])
rewards = -0.04 * numpy.ones(world.shape)
rewards[0,3] = 1.
rewards[1,3] = -1.
actions = dict({'up': [-1,0], 'down': [1,0], 'left': [0,-1], 'right': [0,1]})
transition_function = dict({'up': dict({'up': 0.8, 'left': 0.1, 'right': 0.1}), 'down': dict({'down': 0.8, 'left': 0.1, 'right': 0.1}), 'left': dict({'left': 0.8, 'up': 0.1, 'down': 0.1}), 'right': dict({'right': 0.8, 'up': 0.1, 'down': 0.1})})
terminal_states = dict({(0,3): 1., (1,3): -1.})

decision_process = MDP(world, actions, transition_function, rewards, terminal_states)
utilities, policy, iterations = decision_process.value_iteration()