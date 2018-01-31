import numpy as np
from scipy.stats import invwishart

def normpdf(x, mu=0, sigma=1):
    '''
    computes the pdf of a normal distribution
    :param x: (float)
    :param mu: (float) mean
    :param sigma: (float) std
    :return: (float)
    '''

    u = float((x-mu) / abs(sigma))
    y = np.exp(-u*u/2) / (np.sqrt(2*np.pi) * abs(sigma))
    return y

class IOMM:
    def __init__(self, Type = 'Binaries'):

        self.Type = Type

    def P_x_i(self, x, z, Theta):
        '''
        computes the probability of the vector x conditionally to its clusters assignations z and the parameters Theta
        :param x: (np array)
        :param z: (np array)
        :param Theta: (np array)
        :return: (float)
        '''

        if self.Type == 'Binaries':

            return np.prod([self.P_x_id(x[d], z, Theta[:, d]) for d in range(np.shape(Theta)[1])])

        if self.Type == 'Gaussian':

            return np.prod([self.P_x_id(x[d], z, Theta[:, [d, int(np.shape(Theta)[1]/2) + d]]) for d in range(int(np.shape(Theta)[1]/2))])


    def P_x_d(self, x, z, Theta):
        '''
        compute the probability of an entire column given the clusters attributions z and the parameters Theta
        :param x: (np array)
        :param z: (np array)
        :param Theta: (np array)
        :return: (float)
        '''
        if self.Type == 'Binaries':

            return np.prod([self.P_x_id(x[i], z[i, :], Theta) for i in range(np.shape(x)[0])])

        if self.Type == 'Gaussian':

            return np.prod([self.P_x_id(x[i], z[i, :], Theta) for i in range(np.shape(x)[0])])


    def P_x_id(self, x, z, Theta):
        '''
        computes the probability of obtaining x on the d th variable given the clusters and parameters
        :param x: (float)
        :param z: (np array)
        :param Theta: (np array)
        :return: (float)
        '''

        if self.Type == 'Binaries':
            p_1 = np.power(Theta, z).prod()
            p_0 = np.power((1 - Theta), z).prod()
            return (p_1/(p_0 + p_1))**x*(p_0/(p_0 + p_1))**(1-x)

        if self.Type == 'Gaussian':
            sigma = 1/(np.multiply(1 / Theta[:, 1], z)).sum()
            mu = np.multiply(Theta[:, 0], np.multiply(1/Theta[:, 1], z)).sum()
            prob = normpdf(x, mu=sigma*mu, sigma=np.sqrt(sigma))
            if np.isnan(prob):
                return 0

            else:
                return prob


    def rate(self, x, i, k, z_0, z_1, Theta):
        '''
        Probability of z[i, k] = 1 knowing Theta
        :param x: (np array)
        :param i: (int)
        :param k: (int)
        :param z_0: (np array) with z[i, k] = 0
        :param z_1: (np array) with z[i, k] = 1
        :param Theta: (np array)
        :return: (float)
        '''

        m = sum(z_1[:, k])-1
        p_0 = (np.shape(x)[0]-m)*self.P_x_i(x[i, :], z_0[i, :], Theta)
        p_1 = m*self.P_x_i(x[i, :], z_1[i, :], Theta)
        rate = p_1/(p_1 + p_0)

        return rate


    def init_Theta(self, K, d):
        '''
        initializes Theta
        :param K: (int) number of cluusters
        :param d: (int) number of variables
        :return: (np array)
        '''

        if self.Type == 'Binaries' :
            Theta = np.random.uniform(low=1e-10, high=1-(1e-10), size=(K, d))

        if self.Type == 'Gaussian':
            #Theta = np.zeros((K, 2*d))
            # for i in range(K):
            #     for j in range(d):
            #         sigma = invwishart.rvs(df=1, scale=0.5, size=1, random_state=None)
            #         mu = np.random.normal(loc=0, scale=3*sigma)
            #         Theta[i, j] = mu
            #         Theta[i, j+d] = sigma

            Theta =  np.concatenate([np.random.uniform(low=-1, high=1, size=(K, d)),
                                   np.random.uniform(low=1e-10, high=1-(1e-10), size=(K, d))], axis=1)

        return Theta


    def sample_Theta(self, x, z, Theta, K, d, omega):
        '''
        Proposition of a new Theta with respect to the current Theta, the clusters assignations z, and the data x
        :param x: (np array)
        :param z: (np array)
        :param Theta: (np array)
        :param K: (int)
        :param d: (int)
        :param omega: (float) width of the proposition window
        :return: (np array)
        '''

        Theta_out = Theta.copy()
        Theta_temp = Theta.copy()

        for i in range(K):
            for j in range(d):
                if self.Type == 'Binaries':
                    Theta_temp[i, j] = np.random.uniform(high=max(Theta[i, j] - omega, 1e-10),
                                                         low=min(Theta[i, j] + omega, 1))

                elif self.Type == 'Gaussian':
                    Theta_temp[i, j] = np.random.uniform(high=Theta[i, j] - omega,
                                                         low=Theta[i, j] + omega)
                    Theta_temp[i, j+d] = np.random.uniform(high=max(Theta[i, j + d] - omega, 1e-10),
                                                         low=min(Theta[i, j + d] + omega, 1))

        for i in range(K):
            for j in range(d):
                if self.Type == 'Binaries':
                    t_ratio = (max(Theta_temp[i, j] - omega, 0) - min(Theta_temp[i, j] + omega, 1)) / (
                            max(Theta[i, j] - omega, 0) - min(Theta[i, j] + omega, 1))
                    p_ratio = self.P_x_d(x[:, j], z, Theta_temp[:, j]) / self.P_x_d(x[:, j], z, Theta[:, j])

                elif self.Type == 'Gaussian':
                    t_ratio = (max(Theta_temp[i, d+j] - omega, 0) - min(Theta_temp[i, d+j] + omega, 1)) / \
                              (max(Theta[i, d+j] - omega, 0) - min(Theta[i, d+j] + omega, 1))
                    p_ratio = self.P_x_d(x[:, j], z, Theta_temp[:, [j, d+j]]) / self.P_x_d(x[:, j], z, Theta[:, [j, d+j]])

                accept_ratio = t_ratio*p_ratio
                u = np.random.uniform(low=0, high=1)
                if u < accept_ratio:
                   Theta_out[i, j] = Theta_temp[i, j]

                Theta_temp[i, j] = Theta[i, j]

        return Theta_out


    def fit(self, x, K_init, Niter, alpha=1, omega=0.01, prop_new_clusts=True, stochastic = False, burn_in=0):
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

        n_clusters = []

        n, d = np.shape(x)
        Theta = self.init_Theta(K_init, d)

        z = np.random.binomial(n=1, p=0.5, size=(n, K_init))

        for j in range(Niter):

            ind = z.sum(axis=0) > 0
            K = ind.sum()
            z_t = z[:, ind].copy()
            Theta_t = Theta[ind, :].copy()
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
                        z_prop[i_ind, :] = 1
                        z_prop_t = np.concatenate([z_t, z_prop], axis=1)
                        accept_ratio = np.prod([self.P_x_i(x[_, :], z_prop_t[_,:], Theta_prop_t)/self.P_x_i(
                                x[_, :], z_t[_,:], Theta_t) for _ in range(n)])
                        u = np.random.uniform(low=0, high=1)
                        if (u < accept_ratio) or (np.isnan(accept_ratio) == True):
                            K += k_new
                            Theta_t =  Theta_prop_t.copy()
                            z_t =  z_prop_t.copy()
                            Theta = np.concatenate([Theta, Theta_prop], axis=0)
                            z = np.concatenate([z, z_prop], axis=1)

            ind = z.sum(axis=0) > 0
            Theta[ind, :] = Theta_t.copy()
            z[:, ind] = z_t.copy()

            ind = z.sum(axis=0) > 0
            Theta[ind, :] = self.sample_Theta(x, z[:, ind], Theta[ind, :], ind.sum(), d, omega)

            if j == burn_in:
                U_out = np.dot(z[:, ind], np.transpose(z[:, ind]))
                Theta_out = Theta
                Z_out = z

            if j > burn_in:
                U_out = np.add(U_out, np.dot(z_t, np.transpose(z_t)))
                if self.Type == 'Binaries':
                    Theta_out = np.concatenate([Theta_out, np.zeros((np.shape(Theta)[0]-np.shape(Theta_out)[0], d))],
                                               axis=0)

                elif self.Type=='Gaussian':
                    Theta_out = np.concatenate([Theta_out, np.zeros((np.shape(Theta)[0] - np.shape(Theta_out)[0], 2*d))],
                                               axis=0)

                Theta_out = np.add(Theta_out, Theta)
                Z_out = np.concatenate([Z_out, np.zeros((n, np.shape(z)[1]-np.shape(Z_out)[1]))], axis=1)
                Z_out = np.add(Z_out, z)
                n_clusters.append(K)

            if j%10 == 0:
                print('Iteration {} done'.format(j))

        U_out = U_out / (Niter-burn_in)
        Theta_out = Theta_out / (Niter-burn_in)
        Z_out = Z_out / (Niter-burn_in)

        return Z_out, U_out, Theta_out, n_clusters



