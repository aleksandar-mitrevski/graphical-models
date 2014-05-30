import numpy
import ast
from factor import Factor

class ExactInferenceEngine(object):
    def __init__(self, network):
        """Defines an engine for performing exact inference in a discrete Bayesian network.

        Keyword arguments:
        network -- A 'libpgm.discretebayesiannetwork' object representing a discrete Bayesian network.

        """
        self.network = network
        self.network.toporder()

    def perform_inference(self, query_variable, evidence_variables):
        """Calculates the probability distribution P(query_variable|evidence_variables) using enumeration.
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
        hidden_variables = []
        for _,var in enumerate(self.network.V):
            if var not in known_variables:
                hidden_variables.append(var)
        normaliser = 0.0

        #given the variable ordering represented by 'self.network' and 'hidden_variables',
        #we look for the CPTs that need each of the variables;
        #this will allow us to make use of situations where we can extract terms in front of a summation
        variables, dependency_levels = self.__find_dependency_levels(known_variables, hidden_variables)

        #we start with an initial assignment of variables: the assignment of evidence variables is fixed,
        #while the other ones (including the query variable) will have to be enumerated
        variable_assignments = dict()
        for _,variable in enumerate(self.network.V):
            if variable in evidence_variables.keys():
                variable_assignments[variable] = evidence_variables[variable]
            else:
                variable_assignments[variable] = ''

        #we take the variables whose CPTs are independent of the hidden variables
        #because they can be taken in front of the summation
        sum_independent_variables = variables[numpy.where(dependency_levels==-1)[0]]

        #we calculate the probability of each value of the query variable
        for _,value in enumerate(self.network.Vdata[query_variable]['vals']):
            variable_assignments[query_variable] = value

            #we calculate a product in front of the summation if we have independent variables
            term_product = 1.0
            if len(sum_independent_variables) > 0:
                term_product = self.__calculate_term_product(sum_independent_variables, variable_assignments)

            distribution[value] = term_product * self.__sum_and_enumerate(hidden_variables, variable_assignments, variables, dependency_levels, 0)
            normaliser = normaliser + distribution[value]

        for _,key in enumerate(distribution.keys()):
            distribution[key] = distribution[key] / normaliser

        return distribution

    def perform_ve_inference(self, query_variable, evidence_variables):
        """Calculates the probability distribution P(query_variable|evidence_variables) using variable elimination.
        Assumes that we have only one query variable.

        Keyword arguments:
        query_variable -- The name of the query variable (as called in the network).
        evidence_variables -- A dictionary containing variable names as keys and observed values as values.

        Returns:
        distribution -- A dictionary containing the values of the query variable as keys and the probabilities as values.

        """
        known_variables = list(evidence_variables.keys())
        known_variables.append(query_variable)
        hidden_variables = []
        for _,var in enumerate(self.network.V):
            if var not in known_variables:
                hidden_variables.append(var)

        variables, dependency_levels = self.__find_dependency_levels(known_variables, hidden_variables)

        #we don't want the query variable to be considered as known
        #in the factor creation process
        known_variables.remove(query_variable)
        factors = self.__create_factors(evidence_variables, variables, dependency_levels)

        current_factor = Factor()
        for i in xrange(len(hidden_variables)-1, -1, -1):
            current_factor = self.__multiply_factors(factors, current_factor, dependency_levels, i)
            current_factor.sum_out(hidden_variables[i])

        current_factor = self.__multiply_factors(factors, current_factor, dependency_levels, -1)
        alpha = sum(numpy.array(current_factor.probabilities))

        distribution = dict()
        for i,value in enumerate(current_factor.values):
            distribution[value[0]] = current_factor.probabilities[i] / alpha

        return distribution

    def __sum_and_enumerate(self, hidden_variables, variable_assignments, variables, dependency_levels, current_dependency_level):
        """Recursively calculates the sum of those entries in the joint probability distribution
        which are necessary for finding the probability of the query variable.

        Keyword arguments:
        hidden_variables -- A list containing the names of the hidden variables that have not been assigned yet.
        variable_assignments -- A dictionary containing the current assignments to all variables.
        variables -- Variables in the network with indices aligned to those of 'dependency_levels'.
        dependency_levels -- A 'numpy.array' containing zero-based indices indicating the level at which we can
                             extract terms in front of an inner summation.
        current_dependency_level -- An integer denoting the index of the current inner summation.

        Returns:
        probability -- In the base case, returns the probability of a term calculated
                       after all variables in the network have been assigned.
                       Returns the sum of such terms in the recursive call.

        """

        #--------- Base case ---------
        #we calculate the product of the appropriate variables for the rightmost summation
        if len(hidden_variables) == 0:
            # relevant_variables = variables[numpy.where(dependency_levels==current_dependency_level-1)[0]]
            # probability = self.calculate_term_product(relevant_variables, variable_assignments)
            # return probability
            return 1.0
        #--------- Recursive case ---------
        #we take one of the hidden variables, assign a value to it, perform a recursive call, and sum the results
        else:
            variable_to_assign = hidden_variables[0]
            new_hidden_variables = list(hidden_variables)
            new_hidden_variables.remove(variable_to_assign)

            probability = 0.0
            relevant_variables = variables[numpy.where(dependency_levels==current_dependency_level)[0]]
            for _,value in enumerate(self.network.Vdata[variable_to_assign]['vals']):
                variable_assignments[variable_to_assign] = value

                #we calculate a product of terms in case we have CPTs that are independent
                #of the summation over the hidden variables that are not assigned yet
                term_product = 1.0
                if len(relevant_variables) > 0:
                    term_product = self.__calculate_term_product(relevant_variables, variable_assignments)

                term_product = term_product * self.__sum_and_enumerate(new_hidden_variables, variable_assignments, variables, dependency_levels, current_dependency_level + 1)
                probability = probability + term_product

            return probability

    def __find_dependency_levels(self, known_variables, hidden_variables):
        """Looks for the level at which a term can be extracted in front of an inner summation.
        Doing this for each of the variables allows us to decompose the summation over the hidden variables appropriately.

        Keyword arguments:
        known_variables -- A list containing evidence variables and the query variable.
        hidden_variables -- A list of hidden variables.

        Returns:
        variables -- A 'numpy.array' of variables in the network.
        dependency_levels -- A 'numpy.array' containing zero-based indices that indicate the level at which we can
                             extract a certain term in front of an inner summation. The indices of this array and
                             the array 'variables' are aligned, such that 'dependency_level[i]' denotes the
                             dependency level of 'variable[i]'.

        """
        variables = numpy.array(self.network.V)
        dependency_levels = numpy.zeros(variables.shape, dtype=int)

        #we look for dependencies between the CPTs of the known variables
        #and the hidden variables and assign an appropriate level to them
        for _,variable in enumerate(known_variables):
            variable_index = numpy.where(variables==variable)[0][0]
            dependency_levels[variable_index] = -1
            for i in xrange(len(hidden_variables)):
                hidden = hidden_variables[i]
                variable_is_child = self.network.Vdata[hidden]['children'] != None and variable in self.network.Vdata[hidden]['children']
                if variable_is_child:
                    dependency_levels[variable_index] = i
                    break

        #we look for dependencies between the CPTs of the hidden variables
        #and the other hidden variables and assign an appropriate level to them
        for i,variable in enumerate(hidden_variables):
            variable_index = numpy.where(variables==variable)[0][0]
            dependency_levels[variable_index] = i
            for j in xrange(i+1,len(hidden_variables)):
                hidden = hidden_variables[j]
                variable_is_child = self.network.Vdata[hidden]['children'] != None and variable in self.network.Vdata[hidden]['children']
                if variable_is_child:
                    dependency_levels[variable_index] = j
                    break

        return variables, dependency_levels

    def __calculate_term_product(self, relevant_variables, variable_assignments):
        """Calculates the product of CPT entries that correspond to the
        assignments given by 'variable_assignments' and contain the variables in 'relevant_variables'.

        Keyword arguments:
        relevant_variables -- A 'numpy.array' containing variable names.
        variable_assignments -- A dictionary containing the current variable assignments.

        Returns:
        probability -- The calculated product of terms.

        """
        probability = 1.0

        for _,variable in enumerate(relevant_variables):
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

    def __create_factors(self, evidence_variables, variables, dependency_levels):
        """

        Keyword arguments:
        evidence_variables -- A dictionary containing evidence variables as keys and the observed values as values.
        variables -- A list of variables in the network.
        dependency_levels -- A 'numpy.array' returned by 'self.__find_dependency_levels'.

        Returns:
        factors -- A list of 'Factor' objects ordered in reverse order of the values in 'dependency_levels'.

        """

        #we are going to create factors from the highest level of dependency
        sorting_indices = numpy.argsort(dependency_levels)[::-1]
        number_of_variables = len(variables)
        known_variables = list(evidence_variables.keys())
        factors = []

        for i in xrange(number_of_variables):
            factor_variables = []
            factor_values = []
            factor_probabilities = []

            variable = variables[sorting_indices[i]]
            values = numpy.array(self.network.Vdata[variable]['vals'])
            parents = self.network.Vdata[variable]['parents']
            if parents == None:
                #the factor has an empty scope if we have
                #an evidence variable that doesn't have any parents
                if variable in known_variables:
                    evidence_value = evidence_variables[variable]
                    value_index = numpy.where(values == evidence_value)[0]
                    cpt = self.network.Vdata[variable]['cprob']

                    factor_values.append(values[value_index])
                    factor_probabilities.append(cpt[value_index])
                else:
                    cpt = self.network.Vdata[variable]['cprob']
                    factor_variables.append(variable)

                    for i,val in enumerate(values):
                        factor_values.append([val])
                        factor_probabilities.append(cpt[i])
            else:
                #if the variable is one of the evidence variables,
                #the factor will only contain the probabilities
                #that correspond to the observed value
                if variable in known_variables:
                    evidence_value = evidence_variables[variable]
                    value_index = numpy.where(values == evidence_value)[0]
                    cpt = self.network.Vdata[variable]['cprob']

                    #we check if any of the variable's parents is an evidence variable
                    known_parents = []
                    for _,parent in enumerate(parents):
                        if parent in known_variables:
                            known_parents.append(parent)

                    #if none of the parents are known, we just take the probabilities
                    #that correspond to the variable's observed value
                    if len(known_parents) == 0:
                        factor_variables = list(parents)
                        for _,cpt_values in enumerate(cpt.keys()):
                            cpt_values_list = ast.literal_eval(cpt_values)
                            factor_values.append(cpt_values_list)
                            factor_probabilities.append(cpt[cpt_values][value_index])
                    #if some of the parents are also evidence variables,
                    #we take the probabilities that correspond to the observed values
                    else:
                        factor_variables = list(set(parents) - set(known_parents))
                        parent_indices = [x for x,val in enumerate(parents) if val in known_parents]
                        for _,cpt_values in enumerate(cpt.keys()):
                            cpt_values_list = ast.literal_eval(cpt_values)

                            evidence_values_ok = True
                            for _,index in enumerate(parent_indices):
                                if cpt_values_list[index] != evidence_variables[parents[index]]:
                                    evidence_values_ok = False
                                    break

                            if evidence_values_ok:
                                factor_values_list = []
                                for _,var in enumerate(factor_variables):
                                    variable_index = parents.index(var)
                                    factor_values_list.append(cpt_values_list[variable_index])
                                factor_values.append(factor_values_list)
                                factor_probabilities.append(cpt[cpt_values][value_index])
                else:
                    cpt = self.network.Vdata[variable]['cprob']

                    #we check if any of the variable's parents is an evidence variable
                    known_parents = []
                    for _,parent in enumerate(parents):
                        if parent in known_variables:
                            known_parents.append(parent)

                    if len(known_parents) == 0:
                        factor_variables = list(parents)
                        factor_variables.insert(0, variable)
                        for _,cpt_values in enumerate(cpt.keys()):
                            cpt_values_list = ast.literal_eval(cpt_values)

                            for i,val in enumerate(values):
                                values_list = list(cpt_values_list)
                                values_list.insert(0, val)
                                factor_values.append(values_list)
                                factor_probabilities.append(cpt[cpt_values][i])
                    #if some of the parents are also evidence variables,
                    #we take the probabilities that correspond to the observed values
                    else:
                        factor_variables = list(set(parents) - set(known_parents))
                        factor_variables.insert(0, variable)
                        parent_indices = [x for x,val in enumerate(parents) if val in known_parents]
                        for _,cpt_values in enumerate(cpt.keys()):
                            cpt_values_list = ast.literal_eval(cpt_values)

                            evidence_values_ok = True
                            for _,index in enumerate(parent_indices):
                                if cpt_values_list[index] != evidence_variables[parents[index]]:
                                    evidence_values_ok = False
                                    break

                            if evidence_values_ok:
                                factor_values_list = []
                                for _,var in enumerate(factor_variables):
                                    variable_index = parents.index(var)
                                    factor_values_list.append(cpt_values_list[variable_index])

                                for i,val in enumerate(values):
                                    values_list = list(factor_values)
                                    values_list.insert(0, val)
                                    factor_values.append(values_list)
                                    factor_probabilities.append(cpt[cpt_values][i])

            new_factor = Factor(factor_variables, factor_values, factor_probabilities)
            factors.append(new_factor)

        return factors

    def __multiply_factors(self, factors, current_factor, dependency_levels, current_dependency_level):
        """Multiplies the factors that are at the distribution summation's current level.

        Keyword arguments:
        factors -- A list of factors.
        current_factor -- A factor storing partially computed values; if its list of variables is empty,
                          we are just starting the computations in the network.
        dependency_levels -- A 'numpy.array' returned by 'self.__find_dependency_levels'.
        current_dependency_level -- The distribution summation's current level.

        Returns:
        current_factor -- A factor storing partially computed updated values.

        """
        number_of_factors_to_multiply = len(numpy.where(dependency_levels==current_dependency_level)[0])
        if number_of_factors_to_multiply == 1:
            if len(current_factor.probabilities) == 0:
                current_factor = factors[0]
            else:
                current_factor = current_factor.multiply(factors[0])
            factors.pop(0)
        else:
            for i in xrange(number_of_factors_to_multiply):
                if len(current_factor.probabilities) == 0:
                    current_factor = factors[0]
                else:
                    current_factor = current_factor.multiply(factors[0])
                factors.pop(0)

        return current_factor