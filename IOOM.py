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
                rate = rate*(theta_temp**x[d]*(1-theta_temp)**(1-x[d]))

        return rate


    def init_Theta(self, K, d):

        if self.Type == 'Binaries' :
            Theta = np.random.uniform(low=0, high=1, size=(K, d))

        return Theta

    def sample_Theta(self, x, z, Theta, K, d, omega):

        Theta_temp = Theta
        if self.Type == 'Binaries':
            for i,j in zip(range(K), range(d)):
                Theta_temp[i, j] = np.random.beta(omega*Theta[i, j], omega*(1-Theta[i, j]))

            for i, j in zip(range(K), range(d)):
                t_ratio = beta._pdf(Theta[i, j], a= omega*Theta_temp[i, j], b=omega*Theta_temp[i, j])/beta._pdf(Theta_temp[i, j], a= omega*Theta[i, j], b=omega*Theta[i, j])
                p_ratio = self.rate(x[i,:], z[i,:], np.reshape(Theta_temp[:,j], (K,1)))/self.rate(x[i,:], z[i,:], np.reshape(Theta[:,j], (K,1)))
                a = t_ratio*p_ratio
                u = np.random.uniform(low=0, high=1)
                if u < a:
                    Theta[i, j] = Theta_temp[i, j]

        return Theta


    def fit(self, x, K_init, Niter, alpha=1, omega=0.01, prop_new_clusts=True):

        n, d = np.shape(x)
        Theta = self.init_Theta(K_init, d)
        K = K_init

        z = np.zeros((n, K_init))
        z[:,0] = np.reshape(np.random.binomial(n=1, p=0.5, size=(n, 1)), (n,))

        for j in range(Niter):
            for i in range(n):
                z_temp = z
                z_temp[i,:] = 1
                for k in range(K):
                    z[i, k] = np.random.binomial(n=1, p=(sum(z[:,k])-1)/n*self.rate(x[i,:], z_temp[i,:], Theta))

                if prop_new_clusts:
                    k_new = np.random.poisson(alpha/n)
                    Theta = np.concatenate([Theta, self.init_Theta(k_new, d)], axis=0)
                    z = np.concatenate([z, np.zeros((n,k_new))], axis=1)
                    z[i, K:(K+k_new)]=1
                    K += k_new

            Theta = self.sample_Theta(x, z, Theta, K, d, omega)

        return z, Theta

