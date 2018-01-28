import numpy as np
from scipy.stats import norm
import time as time

class IOOM:
    def __init__(self, Type = 'Binaries'):

        self.Type = Type

    def P_x_i(self, x, z, Theta):

        if self.Type == 'Binaries':

            return np.prod([self.P_x_id(x[d], z, Theta[:, d]) for d in range(np.shape(Theta)[1])])

        if self.Type == 'Gaussian':

            return np.prod([self.P_x_id(x[d], z, Theta[:, [d, int(np.shape(Theta)[1]/2) + d]]) for d in range(int(np.shape(Theta)[1]/2))])


    def P_x_d(self, x, z, Theta):

        if self.Type == 'Binaries':

            return np.prod([self.P_x_id(x[i], z[i, :], Theta) for i in range(np.shape(x)[0])])

        if self.Type == 'Gaussian':

            return np.prod([self.P_x_id(x[i], z[i, :], Theta) for i in range(np.shape(x)[0])])


    def P_x_id(self, x, z, Theta):

        if self.Type == 'Binaries':
            p_1 = np.power(Theta, z).prod()
            p_0 = np.power((1 - Theta), z).prod()
            return (p_1/(p_0 + p_1))**x*(p_0/(p_0 + p_1))**(1-x)

        if self.Type == 'Gaussian':

            return norm.pdf(x, loc=(Theta[:, 0]*1/Theta[:, 1]*z).sum(), scale=1/(1/Theta[:, 1]*z).sum())

    def rate(self, x, i, k, z_0, z_1, Theta):

        m = sum(z_1[:, k])-1
        p_0 = (np.shape(x)[0]-m)*self.P_x_i(x[i, :], z_0[i, :], Theta)
        p_1 = m*self.P_x_i(x[i, :], z_1[i, :], Theta)
        rate = p_1/(p_1 + p_0)

        return rate


    def init_Theta(self, K, d):

        if self.Type == 'Binaries' :
            Theta = np.random.uniform(low=1e-10, high=1-(1e-10), size=(K, d))

        if self.Type == 'Gaussian':
            Theta =  np.random.uniform(low=1e-10, high=1-(1e-10), size=(K, 2*d))

        return Theta

    def sample_Theta(self, x, z, Theta, K, d, omega):

        Theta_out = Theta.copy()
        Theta_temp = Theta.copy()

        for i, j in zip(range(K), range(d)):
            Theta_temp[i, j] = np.random.uniform(high=max(Theta[i, j] - omega, 1e-10), low=min(Theta[i, j] + omega, 1))

        for i,j in zip(range(K), range(d)):
            t_ratio = (max(Theta_temp[i, j]-omega, 0)-min(Theta_temp[i, j]+omega, 1))/(max(Theta[i, j]-omega, 0)-min(Theta[i, j]+omega, 1))
            if self.Type == 'Binaries':
                p_ratio = self.P_x_d(x[:, j], z, Theta_temp[:, j]) / self.P_x_d(x[:, j], z, Theta_temp[:, j])

            elif self.Type == 'Gaussian':
                p_ratio = self.P_x_d(x[:, j], z, Theta_temp[:, [j, d+j]]) / self.P_x_d(x[:, j], z, Theta_temp[:, [j, d+j]])

            accept_ratio = t_ratio*p_ratio
            u = np.random.uniform(low=0, high=1)
            if u < accept_ratio:
               Theta_out[i, j] = Theta_temp[i, j]

            Theta_temp[i, j] = Theta[i, j]

        return Theta_out


    def discard_clusters(self, z, dict_z, Theta):

        for i in range

    def fit(self, x, K_init, Niter, alpha=1, omega=0.01, prop_new_clusts=True, stochastic = False):
        '''
        Computes the estimated clusters and Theta
        :param x: (np array) The unlabeled data
        :param K_init: (int) Number of initial clusters
        :param Niter: (int) Number of iterations
        :param alpha: (float) parameter of the prior
        :param omega: (float) parameter of the proposition
        :param prop_new_clusts: (Boolean) allow new clusters
        :return: (np array) clusters (np array) theta
        '''

        Z_out = []
        U_out = []
        Theta_out = []

        n, d = np.shape(x)
        Theta = self.init_Theta(K_init, d)

        z = np.random.binomial(n=1, p=0.5, size=(n, K_init))

        for j in range(Niter):

            K = (z.sum(axis=0) > 0).sum()
            z_t = z[:, z.sum(axis=0) > 0].copy()
            Theta_t = Theta[z.sum(axis=0) > 0, :].copy()
            for i in range(n):
                if stochastic:
                    i_ind = np.random.choice(n)

                else:
                    i_ind = i

                for k in range(K):
                    z_temp_1 = z_t.copy()
                    z_temp_1[i_ind, k] = 1
                    z_temp_0 = z_t.copy()
                    z_temp_0[i_ind, k] = 0
                    p = self.rate(x, i_ind, k, z_temp_0, z_temp_1, Theta_t)
                    if np.isnan(p):
                        z_t[i_ind, k] = 0

                    else:
                        z_t[i_ind, k] = np.random.binomial(n=1, p=p)

                if prop_new_clusts:
                    if stochastic:
                        k_new = np.random.poisson(alpha/(i+1))

                    else :
                        k_new = np.random.poisson(alpha / n)

                    if k_new > 0:
                        Theta_prop = self.init_Theta(k_new, d)
                        Theta_prop_t = np.concatenate([Theta_t, Theta_prop], axis=0)
                        z_prop = np.zeros((n, k_new))
                        z_prop_t = np.concatenate([z_t, z_prop], axis=1)
                        z_prop[i_ind, :] = 1
                        z_prop_t[i_ind, K:(K + k_new)] = 1
                        accept_ratio = np.prod([self.P_x_i(x[_, :], z_prop_t[_,:], Theta_prop_t)/self.P_x_i(
                                x[_, :], z_t[_,:], Theta_t) for _ in range(n)])
                        u = np.random.uniform(low=0, high=1)
                        if u < accept_ratio or np.isnan(accept_ratio):
                            K += k_new
                            Theta_t =  Theta_prop_t
                            z_t =  z_prop_t
                            Theta = np.concatenate([Theta, Theta_prop], axis=0)
                            z = np.concatenate([z, z_prop], axis=1)

            #Theta[z.sum(axis=0) > 0, :] = Theta_t
            #z[:, z.sum(axis=0) > 0] = z_t

            Theta = Theta_t
            z = z_t

            Theta = self.sample_Theta(x, z, Theta, K, d, omega)
            #Z_out.append(z)
            U_out.append(np.dot(z[:, z.sum(axis=0) > 0],np.transpose(z[:, z.sum(axis=0) > 0])))
            Theta_out.append(Theta)

            if j%10 == 0:
                print('Iteration {} done'.format(j))


        return Z_out, U_out, Theta_out



