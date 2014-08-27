import numpy

class ApproximateInferenceEngine(object):
    def __init__(self, network):
        """Defines an engine for performing approximate inference in a discrete Bayesian network.

        Keyword arguments:
        network -- A 'libpgm.discretebayesiannetwork.DiscreteBayesianNetwork' object representing a discrete Bayesian network.

        """
        self.network = network

        #a topological order will be needed in the process of generating variable assignments
        self.network.toporder()

    def perform_rs_inference(self, query_variable, evidence_variables, number_of_samples):
        """Calculates the probability distribution P(query_variable|evidence_variables)
        using rejection sampling. Assumes that we have only one query variable.

        Keyword arguments:
        query_variable -- The name of the query variable (as called in the network).
        evidence_variables -- A dictionary containing variable names as keys and observed values as values.
        number_of_samples -- The number of samples that should be used in the sampling process.

        Returns:
        distribution -- A dictionary containing the values of the query variable as keys and the probabilities as values.

        """
        evidence_supporting_sample_counter = 0.
        distribution = dict()
        number_of_variable_values = len(self.network.Vdata[query_variable]['vals'])
        for i in xrange(number_of_variable_values):
            distribution[self.network.Vdata[query_variable]['vals'][i]] = 0.

        for i in xrange(number_of_samples):
            sample_assignments = self.generate_rs_sample()
            supports_evidence = True
            for key in evidence_variables.keys():
                if evidence_variables[key] != sample_assignments[key]:
                    supports_evidence = False
                    break

            if supports_evidence:
                evidence_supporting_sample_counter = evidence_supporting_sample_counter + 1.
                distribution[sample_assignments[query_variable]] = distribution[sample_assignments[query_variable]] + 1.

        if evidence_supporting_sample_counter > 1e-10:
            for key in distribution.keys():
                distribution[key] = distribution[key] / evidence_supporting_sample_counter
        return distribution

    def perform_lw_inference(self, query_variable, evidence_variables, number_of_samples):
        """Calculates the probability distribution P(query_variable|evidence_variables)
        using likelihood weighting. Assumes that we have only one query variable.

        Keyword arguments:
        query_variable -- The name of the query variable (as called in the network).
        evidence_variables -- A dictionary containing variable names as keys and observed values as values.
        number_of_samples -- The number of samples that should be used in the sampling process.

        Returns:
        distribution -- A dictionary containing the values of the query variable as keys and the probabilities as values.

        """
        distribution = dict()
        number_of_variable_values = len(self.network.Vdata[query_variable]['vals'])
        for i in xrange(number_of_variable_values):
            distribution[self.network.Vdata[query_variable]['vals'][i]] = 0.

        for i in xrange(number_of_samples):
            sample_assignments, weight = self.generate_lw_sample(evidence_variables)
            distribution[sample_assignments[query_variable]] = distribution[sample_assignments[query_variable]] + weight

        normaliser = 0.
        for key in distribution.keys():
            normaliser = normaliser + distribution[key]

        if normaliser > 1e-10:
            for key in distribution.keys():
                distribution[key] = distribution[key] / normaliser
        return distribution

    def perform_gibbs_inference(self, query_variable, evidence_variables, number_of_samples):
        """Calculates the probability distribution P(query_variable|evidence_variables)
        using Gibbs sampling. Assumes that we have only one query variable.

        Keyword arguments:
        query_variable -- The name of the query variable (as called in the network).
        evidence_variables -- A dictionary containing variable names as keys and observed values as values.
        number_of_samples -- The number of samples that should be used in the sampling process.

        Returns:
        distribution -- A dictionary containing the values of the query variable as keys and the probabilities as values.

        """
        distribution = dict()
        number_of_variable_values = len(self.network.Vdata[query_variable]['vals'])
        for i in xrange(number_of_variable_values):
            distribution[self.network.Vdata[query_variable]['vals'][i]] = 0.

        variable_assignments = dict()

        #we initialise the variables randomly before generating samples
        for _,variable in enumerate(self.network.V):
            if variable in evidence_variables.keys():
                variable_assignments[variable] = evidence_variables[variable]
            else:
                value_index = numpy.random.randint(0,len(self.network.Vdata[variable]['vals']))
                variable_assignments[variable] = self.network.Vdata[variable]['vals'][value_index]

        for i in xrange(number_of_samples):
            variable_assignments = self.generate_gibbs_sample(variable_assignments, evidence_variables)
            distribution[variable_assignments[query_variable]] = distribution[variable_assignments[query_variable]] + 1.

        normaliser = 0.
        for key in distribution.keys():
            normaliser = normaliser + distribution[key]

        if normaliser > 1e-10:
            for key in distribution.keys():
                distribution[key] = distribution[key] / normaliser
        return distribution

    def generate_rs_sample(self):
        """Generates a random assignment for the variables in the network.
        The assignment respects the conditional probabilities in the network.

        Returns:
        assigned_values -- A dictionary containing variable names and their assigned values.

        """
        number_of_variables = len(self.network.V)
        assigned_values = dict()
        for i in xrange(number_of_variables):
            assigned_values[self.network.V[i]] = ''

        for i in xrange(number_of_variables):
            current_variable = self.network.V[i]
            parent_values = self.get_parent_values(current_variable, assigned_values)

            cumulative_distribution = [0.]
            if parent_values == None:
                number_of_values = len(self.network.Vdata[current_variable]['cprob'])
                for i in xrange(number_of_values):
                    cumulative_distribution.append(cumulative_distribution[i] + self.network.Vdata[current_variable]['cprob'][i])
            else:
                number_of_values = len(self.network.Vdata[current_variable]['cprob'][parent_values])
                for i in xrange(number_of_values):
                    cumulative_distribution.append(cumulative_distribution[i] + self.network.Vdata[current_variable]['cprob'][parent_values][i])

            value_index = 1
            number_of_values = len(cumulative_distribution)
            random_number = numpy.random.rand()
            while value_index < number_of_values and random_number > cumulative_distribution[value_index]:
                value_index = value_index + 1

            #we decrease the index by 1 because of the additional value in the cumulative distribution array
            value_index = value_index - 1
            assigned_values[current_variable] = self.network.Vdata[current_variable]['vals'][value_index]

        return assigned_values

    def generate_lw_sample(self, evidence_variables):
        """Generates a random assignment for the variables in the network.
        The assignment respects the conditional probabilities in the network.

        Keyword arguments:
        evidence_variables -- A dictionary containing variable names as keys and observed values as values.

        Returns:
        assigned_values -- A dictionary containing variable names and their assigned values.

        """
        number_of_variables = len(self.network.V)
        assigned_values = dict()
        for i in xrange(number_of_variables):
            if self.network.V[i] in evidence_variables.keys():
                assigned_values[self.network.V[i]] = evidence_variables[self.network.V[i]]
            else:
                assigned_values[self.network.V[i]] = ''

        weight = 1.
        for i in xrange(number_of_variables):
            current_variable = self.network.V[i]
            parent_values = self.get_parent_values(current_variable, assigned_values)

            #we update the weight if we are sampling an evidence variable
            if self.network.V[i] in evidence_variables.keys():
                value_index = self.network.Vdata[current_variable]['vals'].index(evidence_variables[self.network.V[i]])
                if parent_values == None:
                    weight = weight * self.network.Vdata[current_variable]['cprob'][value_index]
                else:
                    weight = weight * self.network.Vdata[current_variable]['cprob'][parent_values][value_index]
            else:
                cumulative_distribution = [0.]
                if parent_values == None:
                    number_of_values = len(self.network.Vdata[current_variable]['cprob'])
                    for i in xrange(number_of_values):
                        cumulative_distribution.append(cumulative_distribution[i] + self.network.Vdata[current_variable]['cprob'][i])
                else:
                    number_of_values = len(self.network.Vdata[current_variable]['cprob'][parent_values])
                    for i in xrange(number_of_values):
                        cumulative_distribution.append(cumulative_distribution[i] + self.network.Vdata[current_variable]['cprob'][parent_values][i])

                value_index = 1
                number_of_values = len(cumulative_distribution)
                random_number = numpy.random.rand()
                while value_index < number_of_values and random_number > cumulative_distribution[value_index]:
                    value_index = value_index + 1

                #we decrease the index by 1 because of the additional value in the cumulative distribution array
                value_index = value_index - 1
                assigned_values[current_variable] = self.network.Vdata[current_variable]['vals'][value_index]

        return assigned_values, weight

    def generate_gibbs_sample(self, variable_assignments, evidence_variables):
        """Generates a random assignment for the non-evidence variables in the network,
        sampling each of them given their Markov blanket.

        Keyword arguments:
        variable_assignments -- A dictionary containing variable names as keys and variable assignments as values.
        evidence_variables -- A dictionary containing variable names as keys and observed values as values.

        Returns:
        variable_assignments -- A dictionary containing variable names as keys and variable assignments as values.

        """
        variables_to_sample = list(set(self.network.V) - set(evidence_variables.keys()))

        for _,variable in enumerate(variables_to_sample):
            #we calculate the product of the probabilities of the children
            #given their parents if the variable has any children
            value_probabilities = dict()
            for _,value in enumerate(self.network.Vdata[variable]['vals']):
                value_probabilities[value] = 1.

            if self.network.Vdata[variable]['children'] != None:
                for _,value in enumerate(self.network.Vdata[variable]['vals']):
                    alternative_assignments = dict(variable_assignments)
                    alternative_assignments[variable] = value
                    children_probability_product = 1.

                    for _,child in enumerate(self.network.Vdata[variable]['children']):
                        value_index = self.network.Vdata[child]['vals'].index(variable_assignments[child])
                        parent_values = self.get_parent_values(child, alternative_assignments)
                        children_probability_product = children_probability_product * self.network.Vdata[child]['cprob'][parent_values][value_index]
                    value_probabilities[value] = children_probability_product

            parent_values = self.get_parent_values(variable, variable_assignments)
            normaliser = 0.

            #we multiply the children probability by the probability of the
            #current variable given its parents (or by its prior if it has no parents)
            if parent_values == None:
                for i,value in enumerate(self.network.Vdata[variable]['vals']):
                    value_probabilities[value] = value_probabilities[value] * self.network.Vdata[variable]['cprob'][i]
                    normaliser = normaliser + value_probabilities[value]
            else:
                for i,value in enumerate(self.network.Vdata[variable]['vals']):
                    value_probabilities[value] = value_probabilities[value] * self.network.Vdata[variable]['cprob'][parent_values][i]
                    normaliser = normaliser + value_probabilities[value]

            for _,key in enumerate(value_probabilities.keys()):
                value_probabilities[key] = value_probabilities[key] / normaliser

            cumulative_distribution = [0.]
            for i,value in enumerate(self.network.Vdata[variable]['vals']):
                cumulative_distribution.append(cumulative_distribution[i] + value_probabilities[value])

            value_index = 1
            number_of_values = len(cumulative_distribution)
            random_number = numpy.random.rand()
            while value_index < number_of_values and random_number > cumulative_distribution[value_index]:
                value_index = value_index + 1

            #we decrease the index by 1 because of the additional value in the cumulative distribution array
            value_index = value_index - 1
            variable_assignments[variable] = self.network.Vdata[variable]['vals'][value_index]

        return variable_assignments

    def get_parent_values(self, variable, assigned_values):
        """Returns the assigned values to the parent variables of a given variable.

        Keyword arguments:
        variable -- A string representing a variable in the network.
        assigned_values -- A dictionary containing value assignments to variables.

        Returns:
        parent_values_string -- A string representing the values assigned to the parents of the given variable.
        The string is in a format compatible with the network representation.

        """
        if self.network.Vdata[variable]['parents'] == None:
            return None
        else:
            number_of_parents = len(self.network.Vdata[variable]['parents'])
            parent_values = []
            for i in xrange(number_of_parents):
                current_parent = self.network.Vdata[variable]['parents'][i]
                parent_values.append(assigned_values[current_parent])
            parent_values_string = "[" + ", ".join("'" + x + "'" for x in parent_values) + "]"

        return parent_values_string