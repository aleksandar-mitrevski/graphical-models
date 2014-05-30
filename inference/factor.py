import numpy

class Factor(object):
    def __init__(self, variables=[], values=[], probabilities=[]):
        """Defines a new factor of the variables in 'variables'.

        Keyword arguments:
        variables -- A list of variables in the factor.
        values -- A 2D list of values of the factor variables.
        probabilities -- A 2D list of probabilities corresponding to the factor variables;
                         'probabilities' should be aligned with 'values', i.e.
                         'probabilities[i]' should result from 'values[i]'.

        """
        self.variables = variables
        self.values = values
        self.probabilities = probabilities

    def multiply(self, other):
        """Multiplies 'self' with 'other'.

        Keyword arguments:
        other -- A 'Factor' object.

        Returns:
        new_factor -- A 'Factor' object representing the product of the factors 'self' and 'other'.

        """

        #we create the set of variables in the new factor
        new_variables = list(set(self.variables).union(set(other.variables)))
        new_values = []
        new_probabilities = []

        if len(self.variables) == 0:
            for i,values in enumerate(other.values):
                new_values.append(values)
                new_probabilities.append(self.probabilities[0] * other.probabilities[i])
        elif len(other.variables) == 0:
            for i,values in enumerate(self.values):
                new_values.append(values)
                new_probabilities.append(self.probabilities[i] * other.probabilities[0])
        else:
            #we look for the common variables in the factors and their indices in both factors
            common_variables = list(set(self.variables).intersection(set(other.variables)))
            f1_common_variables_indices = []
            f2_common_variables_indices = []
            for _,variable in enumerate(common_variables):
                f1_common_variables_indices.append(self.variables.index(variable))
                f2_common_variables_indices.append(other.variables.index(variable))

            #we multiply the factors
            for i,values in enumerate(self.values):
                for j,other_values in enumerate(other.values):
                    #we multiply 'self.probabilities[i]' and 'other.probabilities[i]'
                    #only if the values of the common variables are equal
                    common_variable_values_equal = True
                    for k in xrange(len(common_variables)):
                        if values[f1_common_variables_indices[k]] != other_values[f2_common_variables_indices[k]]:
                            common_variable_values_equal = False
                            break

                    if not common_variable_values_equal:
                        continue

                    #we create a list corresponding to the assignment of values
                    #to the factor variables; the assignment follows the
                    #variable order in 'new_variables'
                    new_value_list = []
                    for _,var in enumerate(new_variables):
                        if var in self.variables:
                            var_index = self.variables.index(var)
                            new_value_list.append(values[var_index])
                        else:
                            var_index = other.variables.index(var)
                            new_value_list.append(other_values[var_index])

                    new_values.append(new_value_list)
                    new_probabilities.append(self.probabilities[i] * other.probabilities[j])

        new_factor = Factor(new_variables, new_values, new_probabilities)
        return new_factor

    def sum_out(self, variable):
        """Sums out 'variable' from the factor.

        Keyword arguments:
        variable -- Name of a variable in the factor.

        Returns:
        self

        """
        variable_index = self.variables.index(variable)
        new_values = []
        new_probabilities = []

        #we take the value of the variable that we are summing out
        #in the first row of 'self.values' and then sum the probability of this row
        #with the probability of the rows that have different values for the variable,
        #but where the values of the other variables are equal
        while len(self.values) > 0:
            variable_value = self.values[0][variable_index]
            other_variable_values = list(self.values[0])
            other_variable_values.pop(variable_index)
            new_values.append(other_variable_values)

            probability_sum = self.probabilities[0]
            indices_to_remove = [0]

            for i in xrange(1, len(self.values)):
                if self.values[i][variable_index] != variable_value:
                    other_variable_values_i = list(self.values[i])
                    other_variable_values_i.pop(variable_index)

                    if other_variable_values == other_variable_values_i:
                        probability_sum = probability_sum + self.probabilities[i]
                        indices_to_remove.append(i)

            new_probabilities.append(probability_sum)
            for i in xrange(len(indices_to_remove)-1, -1, -1):
                self.values.pop(indices_to_remove[i])
                self.probabilities.pop(indices_to_remove[i])

        self.variables.pop(variable_index)
        self.values = new_values
        self.probabilities = new_probabilities

        return self