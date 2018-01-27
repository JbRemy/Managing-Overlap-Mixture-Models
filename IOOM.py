import numpy as np
import time as time

class IOOM:
    def __init__(self, Type = 'Binaries'):

        self.Type = Type

    def P_x_i(self, x, z, Theta):

        if self.Type == 'Binaries':

            return np.prod([self.P_x_id(x[d], z, Theta[:,d]) for d in range(np.shape(Theta)[1])])


    def P_x_d(self, x, z, Theta):

        if self.Type == 'Binaries':

            return np.prod([self.P_x_id(x[i], z[i, :], Theta) for i in range(np.shape(x)[0])])


    def P_x_id(self, x, z, Theta):

        if self.Type == 'Binaries':
            p_1 = np.power(Theta, z).prod()
            p_0 = np.power((1 - Theta), z).prod()

            return (p_1/(p_0 + p_1))**x*(p_0/(p_0 + p_1))**(1-x)


    def rate(self, x, i, k, z_0, z_1, Theta):

        if self.Type == 'Binaries':
            m = sum(z_1[:, k])-1
            p_0 = (np.shape(x)[0]-m)*self.P_x_i(x[i, :], z_0[i, :], Theta)
            p_1 = m*self.P_x_i(x[i, :], z_1[i, :], Theta)
            rate = p_1/(p_1 + p_0)

        return rate


    def init_Theta(self, K, d):

        if self.Type == 'Binaries' :
            Theta = np.random.uniform(low=1e-10, high=1-(1e-10), size=(K, d))

        return Theta

    def sample_Theta(self, x, z, Theta, K, d, omega):

        Theta_out = Theta.copy()
        Theta_temp = Theta.copy()

        if self.Type == 'Binaries':
            for i, j in zip(range(K), range(d)):
                Theta_temp[i, j] = np.random.uniform(high=max(Theta[i, j] - omega, 0), low=min(Theta[i, j] + omega, 1))

            for i,j in zip(range(K), range(d)):
                t_ratio = (max(Theta_temp[i, j]-omega, 0)-min(Theta_temp[i, j]+omega, 1))/(max(Theta[i, j]-omega, 0)-min(Theta[i, j]+omega, 1))
                p_ratio = self.P_x_d(x[:, j], z, Theta_temp[:, j]) / self.P_x_d(x[:, j], z, Theta_temp[:, j])
                accept_ratio = t_ratio*p_ratio
                u = np.random.uniform(low=0, high=1)
                if u < accept_ratio:
                   Theta_out[i, j] = Theta_temp[i, j]

                Theta_temp[i, j] = Theta[i, j]

        return Theta_out


    def fit(self, x, K_init, Niter, alpha=1, omega=0.01, prop_new_clusts=True, Z_true=None):
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
        K = K_init

        Knew = 0

        z = np.random.binomial(n=1, p=0.5, size=(n, K_init))

        for j in range(Niter):

            K_t = (z.sum(axis=0) > 0).sum()
            for i in range(n):
                for k in range(K_t):
                    z_temp_1 = z[:, z.sum(axis=0) > 0].copy()
                    z_temp_1[i, k] = 1
                    z_temp_0 = z[:, z.sum(axis=0) > 0].copy()
                    z_temp_0[i, k] = 0
                    p = self.rate(x, i, k, z_temp_0, z_temp_1, Theta[z.sum(axis=0) > 0, :])
                    z[:, z.sum(axis=0) > 0][i, k] = np.random.binomial(n=1, p=p)

                if prop_new_clusts:
                    k_new = np.random.poisson(alpha/n)
                    if k_new > 0:
                        Theta_prop = np.concatenate([Theta, self.init_Theta(k_new, d)], axis=0)
                        z_prop = np.concatenate([z, np.zeros((n, k_new))], axis=1)
                        z_prop[i, K:(K + k_new)] = 1
                        accept_ratio = np.prod([self.P_x_i(x[i, :], z_prop[i,:], Theta_prop)/self.P_x_i(x[i, :], z[i,:], Theta) for i in range(n)])
                        u = np.random.uniform(low=0, high=1)
                        if u < accept_ratio:
                            Knew += 1
                            Theta = Theta_prop
                            z = z_prop
                            K += k_new

            Theta = self.sample_Theta(x, z, Theta, K, d, omega)
            Z_out.append(z)
            U_out.append(np.dot(z[:, z.sum(axis=0) > 0],np.transpose(z[:, z.sum(axis=0) > 0])))
            Theta_out.append(Theta)

            if j%10 == 0:
                print('Iteration {} done'.format(j))


        return Z_out, U_out, Theta_out

