import numpy as np

from IOMM import IOMM
import utils

#Testing IOMM on a toy data set with only two non overlapping clusters
d = 2 #Dimension of the data (number of features)
K = 2 #Number of clusters
n = 100 #Number of individuals

#We opposite set parameters for the two clusters
params_1 = [0, 1]
params_2 = [1-i for i in params_1]

X = np.zeros((n, d))
for i in range(d):
    X_1 = np.random.binomial(n=1, p=params_1[i], size=(int(n/K),1))
    X_2 = np.random.binomial(n=1, p=params_2[i], size=(int(n/K),1))
    X[:, i] = np.reshape(np.concatenate((X_1, X_2), axis=0), (n,))

Z_true = np.transpose(np.array([[1]*50+[0]*50,[0]*50+[1]*50]))
U_true = np.dot(Z_true,np.transpose(Z_true))
utils.plot_similarity(U_true, title='', save_path='/media/sf_Debian-shared-folder/', name='fig_test_1_true_U')

IOMM_classifier = IOMM(Type='Binaries')

Z, U, Theta, n_clusters = IOMM_classifier.fit(X, K_init=1, Niter=3000, alpha=1, omega=0.1, prop_new_clusts=True, burn_in=2000)

utils.Compute_accuracy(U, U_true, thresh=0)

utils.plot_similarity(U, title='', save_path='/media/sf_Debian-shared-folder/', name='fig_test_1_simulated_U')

utils.plot_n_clusters(n_clusters, True_n_clusters = 2, title='', save_path='/media/sf_Debian-shared-folder/', name='fig_test_1_n_clusters')

#Testing IOMM on a toy data set with more clusters and dimension
d = 50 #Dimension of the data (number of features)
K = 5 #Number of clusters
n = 100 #Number of individuals

#We opposite set parameters for the two clusters
params = np.random.uniform(low=0, high=1, size=(K, d))

X = np.zeros((n, d))
Z_true = np.zeros((n, K))
for k in range(K):
    Z_true[20*k:20*(k+1), k] = 1
    for i in range(d):
        X[k*20:(k+1)*20, i] = np.random.binomial(n=1, p=params[k, i], size=(20,))

U_true = np.dot(Z_true,np.transpose(Z_true))
utils.plot_similarity(U_true, title='', save_path='/media/sf_Debian-shared-folder/', name='fig_test_2_true_U')

IOMM_classifier = IOMM(Type='Binaries')

Z, U, Theta, n_clusters = IOMM_classifier.fit(X, K_init=1, Niter=9000, alpha=1, omega=0.05, prop_new_clusts=True, stochastic=True, burn_in=5000)

utils.Compute_accuracy(U, U_true, thresh=0)
utils.Compute_accuracy(U, U_true, thresh=1)
utils.Compute_accuracy(U, U_true, thresh=2)
utils.Compute_accuracy(U, U_true, thresh=3)

utils.plot_similarity(U, title='', save_path='/media/sf_Debian-shared-folder/', name='fig_test_2_simulated_U')

utils.plot_n_clusters(n_clusters, True_n_clusters = 5, title='', save_path='/media/sf_Debian-shared-folder/', name='fig_test_2_n_clusters')



import numpy as np

from IOMM import IOMM
import utils

#Testing IOMM on a toy data set with gaussian clusters
d = 15 #Dimension of the data (number of features)
K = 4 #Number of clusters
n = 100 #Number of individuals

params = np.concatenate([np.random.uniform(low=-1, high=1, size=(K, d)),
                                    np.random.uniform(low=1e-10, high=1-(1e-10), size=(K, d))], axis=1)

X = np.zeros((n, d))
Z_true = np.zeros((n, K))
for k in range(K):
    Z_true[25*k:25*(k+1), k] = 1
    Z_true[25*k+10:25*(k + 1), min(k+1, K-1)] = 1

for i in range(n):
    for j in range(d):
        sigma = 1 / (1 / params[:, d + j] * Z_true[i, :]).sum()
        mu = (params[:, j]* 1/params[:, d+j]* Z_true[i, :]).sum()
        X[i, j] = np.random.normal(loc=sigma*mu, scale=np.sqrt(sigma))

U_true = np.dot(Z_true,np.transpose(Z_true))
utils.plot_similarity(U_true, title='', save_path='/media/sf_Debian-shared-folder/', name='fig_test_4_true_U')

IOMM_classifier = IOMM(Type='Gaussian')

Z, U, Theta, n_clusters = IOMM_classifier.fit(X, K_init=1, Niter=9000, alpha=1, omega=0.1, prop_new_clusts=True, stochastic=False, burn_in=5000)

utils.Compute_accuracy(U, U_true, thresh=0)
utils.Compute_accuracy(U, U_true, thresh=1)
utils.Compute_accuracy(U, U_true, thresh=2)
utils.Compute_accuracy(U, U_true, thresh=3)

utils.plot_similarity(U, title='', save_path='/media/sf_Debian-shared-folder/', name='fig_test_4_simulated_U')

utils.plot_n_clusters(n_clusters, True_n_clusters = 5, title='', save_path='/media/sf_Debian-shared-folder/', name='fig_test_4_n_clusters')