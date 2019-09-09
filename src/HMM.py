import utils
import numpy as np

from itertools import chain 
from collections import Counter

class Baum_Welch():

    def __init__(self, obs_seq, hid_seq):

        self.epsilon = 0.0001
        self.iterations = len(obs_seq)

        self.obs_seq = obs_seq
        self.hid_seq = hid_seq

        self.num_obs = len(set(obs_seq))
        self.num_hids = len(set(hid_seq))

        self.pi_shape = [1, self.num_hids]
        self.pi = torch.zeros(size = self.pi_shape, dtype = torch.float64)

        self.emiss_shape = [self.num_hids, self.num_obs]
        self.emission = torch.zeros(size = self.emiss_shape, dtype = torch.float64)

        self.A_shape = [self.num_hids, self.num_hids]
        self.A = torch.zeros(size = self.trans_shape, dtype = torch.float64)

        self.alpha_beta_shape = [self.num_hids, len(self.obs_seq)]
        self.alpha = torch.zeros(size = self.alpha_beta_shape, dtype = torch.float64)
        self.beta = torch.zeros(size = self.alpha_beta_shape, dtype = torch.float64)

    def expected_output_occurrence(self, index, step_gammas, summed_gammas):
        filtered_gamma = step_gammas.index_select(1, torch.LongTensor(index))
        sum_filtered_gamma = filtered_gamma.sum(dim = 1)
        new_obs_prob = torch.div(sum_filtered_gamma, summed_gammas)
        return new_obs_prob
    
    def forward(self):  
        alpha_initial = self.pi * self.emission[:, self.obs_seq[0]]
        self.alpha[:, 0] = torch.div(alpha_initial, alpha_initial.sum())
        for i, obs in enumerate(self.obs_seq[1:]):
            current_probability = torch.matmul(self.alpha[:, i], A)
            forward_probability = torch.mul(current_probability, self.emission[:, obs])
            self.alpha[:, i+1] = torch.div(forward_probability, forward_probability.sum())

    def backward(self):  
        self.beta[:, -1] = torch.from_numpy(np.array([1.0, 1.0]))
        for i, obs in enumerate(self.obs_seq[:0:-1]):
            x = torch.mul(self.emission[:, obs], self.A)
            _beta = torch.matmul(x, self.beta[:, -(i+1)])
            self.beta[:, -(i+2)] = torch.div(_beta, _beta.sum())

    def calculate_gammas(self):
        num = torch.mul(self.alpha, self.beta)
        denom = torch.sum(num, dim = 0)
        gamma_i = torch.div(num, denom)
        return gamma_i

    def calculate_zetas(self):
        zetas = []
        A_T = torch.transpose(self.A, 1, 0)
        for t, fwd in enumerate(self.alpha[:, :-1].transpose(1, 0)):
            x = torch.mul(fwd, A_T)
            numerator = torch.transpose(x, 1, 0) * self.emission[:, self.obs_seq[t+1]] * self.beta[:, t+1]
            denomenator = torch.sum(numerator, dim = 0).sum(dim = 0)
            zeta = torch.div(numerator, denomenator)
            zetas.append(zeta)
        summed_zetas = torch.stack(zetas, dim = 0).sum(dim = 0)
        return summed_zetas

    def re_estimate_parameters(self):

        step_gammas = self.calculate_gammas()

        ##################################################
        #       Re-estimate initial probabilities
        ##################################################

        new_pi = step_gammas[:, 0]

        ##################################################
        #         Re-estimate transition matrix
        ##################################################

        summed_zetas = self.calculate_zetas()
        summed_gammas = torch.sum(step_gammas[:, :-1], dim = 1)
        new_transition_matrix = torch.div(summed_zetas, summed_gammas.view(-1, 1))

        ##################################################
        #         Re-estimate emission matrix
        ##################################################

        summed_gammas = torch.sum(step_gammas, dim = 1)
        state_indices = [np.where(self.obs_seq == searchval)[0] for searchval in set(self.obs_seq)]
        new_emission_matrix = [self.expected_output_occurrence(value, step_gammas, summed_gammas) for value in state_indices]   
        new_emission_matrix = torch.stack(new_emission_matrix, dim = 0).transpose(1, 0)

        return new_pi, new_transition_matrix, new_emission_matrix

    def check_convergence(self, new_pi, new_transition_matrix, new_emission_matrix):
        delta_pi = torch.max(torch.abs(self.pi - new_pi)).item() < self.epsilon
        delta_transition_mat = torch.max(torch.abs(self.A - new_transition_matrix)).item() < self.epsilon
        delta_emission_mat = torch.max(torch.abs(self.emission - new_emission_matrix)).item() < self.epsilon
        converged =  [delta_pi, delta_transition_mat, delta_emission_mat]
        if not all(converged):
            return False
        else:
            return True

    def expectation_maximization(self):
        # Expectation
        self.forward()
        self.backward()

        # Maximization 
        new_pi, new_transition_matrix, new_emission_matrix = self.re_estimate_parameters()

        converged = self.check_convergence(new_pi, new_transition_matrix, new_emission_matrix)

        self.pi = new_pi
        self.A = new_transition_matrix
        self.emission = new_emission_matrix

        return converged

    def baum_welch(self):

        for i in range(self.iterations):
            converged = self.expectation_maximization()
            if converged:
                print("Converged at iteration: {}".format(i))
                break
                
        return self.pi, self.A, self.emission

if __name__ == "__main__":

    bm = Baum_Welch()
