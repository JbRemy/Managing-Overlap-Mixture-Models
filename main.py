import numpy as np

from IOOM import IOOM
import utils

#Testing IOOM on a toy data set with only two non overlapping clusters
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

ioom_classifier = IOOM(Type='Binaries')

list_Z, list_U, list_Theta = ioom_classifier.fit(X, K_init=2, Niter=2000, alpha=1, omega=0.1, prop_new_clusts=True, Z_true=Z_true)

utils.Compute_accuracy(list_U, U_true, thresh=0, burn_in=1000)

burn_in=1000
U = sum(list_U[burn_in:len(list_U)])/(len(list_U)-burn_in)
U = np.vectorize(round)(U)
utils.plot_similarity(U, title='', save_path='/media/sf_Debian-shared-folder/', name='fig1')
utils.plot_similarity(U_true, title='', save_path='/media/sf_Debian-shared-folder/', name='fig2')

utils.plot_n_clusters(list_Z, True_n_clusters = 2, burn_in=1000, title='', save_path='/media/sf_Debian-shared-folder/', name='fig')

#Testing IOOM on a toy data set with more clusters and dimension
d = 15 #Dimension of the data (number of features)
K = 5 #Number of clusters
n = 100 #Number of individuals

#We opposite set parameters for the two clusters
params = np.random.binomial(n=1, p=0.15, size=(K, d))

X = np.zeros((n, d))
Z_true = np.zeros((n, K))
for k in range(K):
    Z_true[10*k:10*(k+1), k] = 1
    for i in range(d):
        X[k*10:(k+1)*10, i] = np.random.binomial(n=1, p=params[k, i], size=(10,))

ioom_classifier = IOOM(Type='Binaries')

Z, Theta = ioom_classifier.fit(X, K_init=2, Niter=5000, alpha=1, omega=0.1, prop_new_clusts=True, Z_true=Z_true)

print((np.abs(np.dot(Z,np.transpose(Z))-np.dot(Z_true,np.transpose(Z_true))) <= 0).sum()/(100**2))
print((np.abs(np.dot(Z,np.transpose(Z))-np.dot(Z_true,np.transpose(Z_true))) <= 1).sum()/(100**2))
print((np.abs(np.dot(Z,np.transpose(Z))-np.dot(Z_true,np.transpose(Z_true))) <= 2).sum()/(100**2))

print((np.dot(Z_true,np.transpose(Z_true)) <= 0).sum()/(100**2))




