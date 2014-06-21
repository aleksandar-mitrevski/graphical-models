import numpy

class MDP(object):
    def __init__(self, world, actions, transition_function, rewards, terminal_states):
        """Defines a Markov decision process where the actions represent movement in a 2D grid.

        Keyword arguments:
        world -- A 2D list or numpy array of boolean values, such that a value 'True' at position [i,j] means that the (i,j)th cell is an obstacle.
        actions -- A dictionary in which the keys represent action names and the values action results.
        transition_function -- A dictionary in which the keys are actions, while the values are dictionaries of possible outcomes and their probabilities
        rewards -- A 2D list or numpy array of rewards for each of the states.
        terminal_states - A dictionary in which the keys are tuples of (row,column) indices, while the values are utilities of the corresponding terminal states.

        """
        self.world = numpy.array(world)
        self.actions = dict(actions)
        self.transition_function = dict(transition_function)
        self.rewards = numpy.array(rewards)
        self.terminal_states = dict(terminal_states)

        self.rows = self.world.shape[0]
        self.columns = self.world.shape[1]

    def value_iteration(self, initial_utilities=None, discount=1., change_tolerance=1e-5, max_iterations=100000):
        """Performs value iteration on the given MDP.

        Keyword arguments:
        initial_utilities -- A 2D array of initial state utilities. If None, zero initial utilities are assumed.
        discount -- Reward discount.
        change_tolerance -- Tolerance value for the change of utilities between two consecutive iterations.
        max_iterations -- The maximum allowed number of iterations.

        Returns:
        current_utilities -- A 2D array of utilities for each of the states.
        current_policy -- A 2D array representing the policy for each state.
        iterations -- Number of iterations of the algorithm.

        """
        if initial_utilities == None:
            initial_utilities = numpy.zeros(self.world.shape)
        current_utilities = numpy.array(initial_utilities)
        current_policy = numpy.empty(self.world.shape, dtype='|S10')

        #we assign utility values to the terminal states
        for key,value in self.terminal_states.iteritems():
            current_utilities[key[0],key[1]] = value
            current_policy[key[0], key[1]] = 'x'

        #we add an 'x' to the policy array in those states where we have obstacles
        for i in xrange(self.rows):
            for j in xrange(self.columns):
                if self.world[i,j]:
                    current_policy[i,j] = 'x'

        utilities_updated = True
        iterations = 0
        while utilities_updated and iterations < max_iterations:
            utilities_updated = False
            iterations = iterations + 1
            old_utilities = numpy.array(current_utilities)
            maximum_change = 0.

            for i in xrange(self.rows):
                for j in xrange(self.columns):
                    #we don't update the utility of the current state if it is an obstacle
                    if self.world[i,j]:
                        continue

                    #we also don't update the utility of the current state if it is a terminal state
                    state_terminal = False
                    for key,value in self.terminal_states.iteritems():
                        if key[0] == i and key[1] == j:
                            state_terminal = True
                            break

                    if state_terminal:
                        continue

                    #we calculate expected utilities for all actions
                    expected_utilities = dict()
                    for action in self.actions.keys():
                        expected_utility = 0.
                        for potential_outcome,probability in self.transition_function[action].iteritems():
                            new_state_row = i + self.actions[potential_outcome][0]
                            new_state_column = j + self.actions[potential_outcome][1]
                            new_state_row, new_state_column = self.__correct_state_indices(new_state_row, new_state_column, new_state_row-i, new_state_column-j)
                            expected_utility = expected_utility + old_utilities[new_state_row, new_state_column] * probability

                        expected_utilities[action] = expected_utility

                    #we want to maximise the utility, so we look for the action that gives the maximum expected utility
                    max_action, max_action_utility = self.__find_dict_max(expected_utilities)

                    #we update the utility of the current state and save the best action in the policy
                    current_utilities[i,j] = self.rewards[i,j] + discount * max_action_utility
                    current_policy[i,j] = max_action

                    utility_delta = abs(current_utilities[i,j] - old_utilities[i,j])
                    if utility_delta > maximum_change:
                        maximum_change = utility_delta

            if maximum_change > (change_tolerance * (1-discount) / discount):
                utilities_updated = True

        return current_utilities, current_policy, iterations

    def __correct_state_indices(self, row, column, row_update, column_update):
        """Corrects the row and column indices of a state so that they are not outside the grid or in an obstacle field.

        Keyword arguments:
        row -- The row value of a state.
        column -- The column value of a state.
        row_update -- Difference between the new and the old row value.
        column_update -- Difference between the new and the old column value.

        Returns:
        row -- An updated version of the row index.
        column -- An updated version of the column index.

        """
        #we update the row index if it is outside the grid
        if row < 0:
            row = 0
        elif row >= self.rows:
            row = self.rows - 1

        #we update the column index if it is outside the grid
        if column < 0:
            column = 0
        elif column >= self.columns:
            column = self.columns - 1

        #we update both the row and the column index if the state they describe is an obstacle
        if self.world[row, column] == True:
            row = row - row_update
            column = column - column_update

        return row, column

    def __find_dict_max(self, dictionary):
        """Finds the maximum value in a dictionary and its key.

        Keyword arguments:
        dictionary -- A dictionary where the values are numbers.

        Returns:
        max_key -- The key corresponding to the maximum value in the dictionary.
        max_val -- The maximum value in the dictionary.

        """
        keys = numpy.array(dictionary.keys())
        values = numpy.array(dictionary.values())

        max_val = numpy.max(values)
        max_val_arg = numpy.argmax(values)

        max_key = keys[max_val_arg]
        return max_key, max_val