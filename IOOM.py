import numpy as np
from scipy.stats import beta

class IOOM:
    def __init__(self, Type = 'Binaries'):

        self.Type = Type

    def rate(self, x, z, Theta):

        if self.Type == 'Binaries':
            rate = 1
            for d in range(np.shape(Theta)[1]):
                theta_temp = np.prod(np.power(Theta[:,d], z))/(np.prod(np.power(1-Theta[:,d], z))+ np.prod(np.power(Theta[:,d], z)))
                rate = rate*(theta_temp**x[d])*((1-theta_temp)**(1-x[d]))

        return rate

    def rate_ratio(self, x, i, k, z_1, z_2, Theta):

        if self.Type == 'Binaries':
            rate = 1
            m = sum(z_1[:, k])-1
            rate_m = (np.shape(x)[0]-m)/m
            for k in range(np.shape(Theta)[0]):
                for d in range(np.shape(Theta)[1]):
                    rate = rate*np.exp(z_2[i, k]*x[i, d]*np.log(Theta[k, d]/(1+Theta[k, d])))/np.exp(z_1[i, k]*x[i, d]*np.log(Theta[k, d]/(1+Theta[k, d])))
            rate = rate*rate_m
            # for i in range(np.shape(x)[1]):
            #     for d in range(np.shape(Theta)[1]):
            #         theta_temp_1 = np.prod(np.power(Theta[:,d], z_1[i,:]))/(np.prod(np.power(1-Theta[:,d], z_1[i,:]))+ np.prod(np.power(Theta[:,d], z_1[i,:])))
            #         theta_temp_0 = np.prod(np.power(Theta[:,d], z_2[i,:]))/(np.prod(np.power(1-Theta[:,d], z_2[i,:]))+ np.prod(np.power(Theta[:,d], z_2[i,:])))
            #         rate_1 = (theta_temp_1**x[i,d])*((1-theta_temp_1)**(1-x[i,d]))
            #         rate_0 = (theta_temp_0 ** x[i,d]) * ((1 - theta_temp_0) ** (1 - x[i,d]))
            #         rate = rate*rate_0/rate_1

        return rate


    def init_Theta(self, K, d):

        if self.Type == 'Binaries' :
            Theta = np.random.uniform(low=1e-10, high=1-(1e-10), size=(K, d))

        return Theta

    def sample_Theta(self, x, z, Theta, K, d, omega):

        Theta_out = Theta.copy()
        Theta_temp = Theta.copy()

        if self.Type == 'Binaries':
            for i,j in zip(range(K), range(d)):
                Theta_temp[i, j] = np.random.uniform(high=max(Theta[i, j]-omega, 0), low=min(Theta[i, j]+omega, 1))
                t_ratio = (max(Theta_temp[i, j]-omega, 0)-min(Theta_temp[i, j]+omega, 1))/(max(Theta[i, j]-omega, 0)-min(Theta[i, j]+omega, 1))
                p_ratio = self.rate(np.reshape(x[i,j], (1,1)), z[i,:], np.reshape(Theta_temp[:,j], (K,1)))/self.rate(np.reshape(x[i,j], (1,1)), z[i,:], np.reshape(Theta[:,j], (K,1)))
                a = t_ratio*p_ratio
                u = np.random.uniform(low=0, high=1)
                if (u < a):
                   Theta_out[i, j] = Theta_temp[i, j]

                Theta_temp = Theta

        return Theta_out


    def fit(self, x, K_init, Niter, alpha=1, omega=0.01, prop_new_clusts=True):
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

        n, d = np.shape(x)
        Theta = self.init_Theta(K_init, d)
        K = K_init

        z = np.random.binomial(n=1, p=0.5, size=(n, K_init))

        for j in range(Niter):
            for i in range(n):
                for k in range(K):
                    z_temp = z.copy()
                    z_temp[i, k] = 1
                    z_temp_2 = z.copy()
                    z_temp[i, k] = 0
                    #p_1 = (sum(z_temp[:,k])-1)/n*self.rate(x[i,:], z_temp[i,:], Theta)
                    #p_0 = (n-sum(z_temp_2[:,k]))/n*self.rate(x[i,:], z_temp_2[i,:], Theta)
                    p = self.rate_ratio(x, i, k, z_temp, z_temp_2, Theta)
                    print(1/(p+1))
                    z[i, k] = np.random.binomial(n=1, p=1/(p+1))

                if prop_new_clusts:
                    k_new = np.random.poisson(alpha/n)
                    Theta = np.concatenate([Theta, self.init_Theta(k_new, d)], axis=0)
                    z = np.concatenate([z, np.zeros((n,k_new))], axis=1)
                    z[i, K:(K+k_new)]=1
                    K += k_new

            Theta = self.sample_Theta(x, z, Theta, K, d, omega)



        return z, Theta

