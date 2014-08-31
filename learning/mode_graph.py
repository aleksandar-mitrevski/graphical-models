import numpy

class ModeGraph(object):
    def __init__(self):
        self.initial_probabilities = None
        self.transition_model = None
        self.observation_model = None

    def learn_graph(self, data, number_of_means, m, epsilon, max_iterations):
        means, memberships = self.cluster_data(data, number_of_means, m, epsilon, max_iterations)
        self.initial_probabilities, self.transition_model = self.learn_transition_model(data, number_of_means, means, memberships)
        self.observation_model = self.learn_observation_model(data, number_of_means, means, memberships)
        return means, memberships

    def cluster_data(self, data, number_of_means, m, epsilon, max_iterations):
        number_of_vectors, data_dimensionality = data.shape
        means = numpy.zeros((number_of_means, data_dimensionality))
        memberships = numpy.zeros((number_of_vectors, number_of_means))

        for i in xrange(number_of_vectors):
            memberships[i,i%number_of_means] = 1.

        selected_mean_vectors = []
        for i in xrange(number_of_means):
            vector_index = -1
            vector_selected = False
            while not vector_selected:
                vector_index = numpy.random.randint(0,number_of_vectors)
                if vector_index not in selected_mean_vectors:
                    vector_selected = True
            means[i,:] = data[vector_index,:] + numpy.random.rand()
            selected_mean_vectors.append(vector_index)

        error = 1e10
        iterations = 0
        while error > epsilon and iterations < max_iterations:
            previous_memberships = numpy.array(memberships)

            #mean vector update
            for i in xrange(number_of_means):
                mean_vector = numpy.zeros(data_dimensionality)
                for j in xrange(number_of_vectors):
                    mean_vector = mean_vector + pow(memberships[j,i],m) * data[j,:]
                normaliser = numpy.sum(pow(memberships[:,i],2))
                means[i,:] = mean_vector / normaliser

            #membership update
            for i in xrange(number_of_vectors):
                membership_sum = 0.
                for j in xrange(number_of_means):
                    membership_normaliser = 0.
                    for k in xrange(number_of_means):
                        membership_normaliser = membership_normaliser + pow(numpy.linalg.norm(data[i,:] - means[j,:]) / numpy.linalg.norm(data[i,:] - means[k,:]), 2/(m-1))
                    memberships[i,j] = 1. / membership_normaliser
                    membership_sum = membership_sum + memberships[i,j]

                for j in xrange(number_of_means):
                    memberships[i,j] = memberships[i,j] / membership_sum

            membership_differences = memberships - previous_memberships
            error = numpy.linalg.norm(membership_differences)
            iterations = iterations + 1

        return means, memberships

    def learn_transition_model(self, data, number_of_means, means, memberships):
        number_of_vectors, data_dimensionality = data.shape
        max_membership_indices = numpy.argmax(memberships, axis=1)
        initial_probabilities = numpy.zeros(number_of_means)
        for i in xrange(number_of_means):
            i_th_cluster_vector_count = len(numpy.where(max_membership_indices==i)[0]) * 1.0
            initial_probabilities[i] = i_th_cluster_vector_count / number_of_vectors

        transition_count = self.count_transitions(data, number_of_means, max_membership_indices)
        transition_probabilities = numpy.zeros((number_of_means, number_of_means))
        for i in xrange(number_of_means):
            i_th_cluster_vectors = numpy.where(max_membership_indices==i)[0]
            i_th_cluster_vector_count = len(i_th_cluster_vectors) * 1.0
            if (number_of_vectors-1) in i_th_cluster_vectors:
                i_th_cluster_vector_count = i_th_cluster_vector_count - 1

            for j in xrange(number_of_means):
                transition_probabilities[i,j] = transition_count[i,j] / i_th_cluster_vector_count

        return initial_probabilities, transition_probabilities

    def learn_observation_model(self, data, number_of_means, means, memberships):
        """I am not happy with the calculations here.
        """
        observation_probabilities = numpy.zeros((number_of_means, number_of_means))
        max_membership_indices = numpy.argmax(memberships, axis=1)
        for i in xrange(number_of_means):
            sum_p_i = numpy.sum(memberships[:,i])
            i_th_cluster_indices = numpy.where(max_membership_indices==i)[0]
            i_th_sum = 0.
            for j in xrange(number_of_means):
                sum_p_i_j = numpy.sum(memberships[i_th_cluster_indices,j])
                observation_probabilities[i,j] = sum_p_i_j / sum_p_i
                i_th_sum = i_th_sum + observation_probabilities[i,j]

            for j in xrange(number_of_means):
                observation_probabilities[i,j] = observation_probabilities[i,j] / i_th_sum

        return observation_probabilities

    def count_transitions(self, data, number_of_means, max_membership_indices):
        number_of_vectors = data.shape[0]
        counts = numpy.zeros((number_of_means, number_of_means))
        for i in xrange(number_of_vectors-1):
            i_th_vector_cluster = max_membership_indices[i]
            iplus1_th_vector_cluster = max_membership_indices[i+1]
            counts[i_th_vector_cluster,iplus1_th_vector_cluster] = counts[i_th_vector_cluster,iplus1_th_vector_cluster] + 1
        return counts