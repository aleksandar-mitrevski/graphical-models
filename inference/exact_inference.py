class ExactInferenceEngine(object):
    def __init__(self, network):
        """Defines an engine for performing exact inference in a discrete Bayesian network.

        Keyword arguments:
        network -- A 'libpgm.discretebayesiannetwork' object representing a discrete Bayesian network.

        """
        self.network = network

    def perform_inference(self, query_variable, evidence_variables):
        """Calculates the probability distribution P(query_variable|evidence_variables).
        Assumes that we have only one query variable.

        Keyword arguments:
        query_variable -- The name of the query variable (as called in the network).
        evidence_variables -- A dictionary containing variable names as keys and observed values as values.

        Returns:
        distribution -- A dictionary containing the values of the query variable as keys and the probabilities as values.

        """
        distribution = dict()
        known_variables = list(evidence_variables.keys())
        known_variables.append(query_variable)
        hidden_variables = list(set(self.network.V) - set(known_variables))
        normalizer = 0.0

        #we assign values to the evidence variables
        variable_assignments = {query_variable: ''}
        for variable in self.network.V:
            if variable in evidence_variables.keys():
                variable_assignments[variable] = evidence_variables[variable]
            else:
                variable_assignments[variable] = ''

        #we calculate the probability of each value of the query variable
        for _,value in enumerate(self.network.Vdata[query_variable]['vals']):
            variable_assignments[query_variable] = value
            distribution[value] = self.__sum_and_enumerate(hidden_variables, variable_assignments)
            normalizer = normalizer + distribution[value]

        for _,key in enumerate(distribution.keys()):
            distribution[key] = distribution[key] / normalizer

        return distribution

    def __sum_and_enumerate(self, hidden_variables, variable_assignments):
        """Recursively calculates the sum of those entries in the joint probability distribution
        which are necessary for finding the probability of the query variable.

        Keyword arguments:
        hidden_variables -- A list containing the names of the hidden variables that have not been assigned yet.
        variable_assignments -- A dictionary containing the current assignments to all variables.

        Returns:
        probability -- In the base case, returns the probability of a term calculated
                       after all variables in the network have been assigned.
                       Returns the sum of such terms in the recursive call.

        """

        #--------- Base case ---------
        #we calculate a term using the chain rule
        if len(hidden_variables) == 0:
            probability = 1.0
            for _,variable in enumerate(variable_assignments.keys()):
                value_index = self.network.Vdata[variable]['vals'].index(variable_assignments[variable])
                parents = self.network.Vdata[variable]['parents']

                if parents == None:
                    probability = probability * self.network.Vdata[variable]['cprob'][value_index]
                else:
                    parent_values = list()
                    for _,parent in enumerate(parents):
                        parent_values.append(variable_assignments[parent])
                    parent_values_string = "[" + ", ".join("'" + x + "'" for x in parent_values) + "]"
                    probability = probability * self.network.Vdata[variable]['cprob'][parent_values_string][value_index]
            return probability
        #--------- Recursive case ---------
        #we take one of the hidden variables, assign a value to it, perform a recursive call, and sum the results
        else:
            variable_to_assign = hidden_variables[0]
            new_hidden_variables = list(hidden_variables)
            new_hidden_variables.remove(variable_to_assign)

            probability = 0.0
            for _,value in enumerate(self.network.Vdata[variable_to_assign]['vals']):
                variable_assignments[variable_to_assign] = value
                probability = probability + self.__sum_and_enumerate(new_hidden_variables, variable_assignments)

            return probability