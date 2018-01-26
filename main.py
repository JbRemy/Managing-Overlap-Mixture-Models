import numpy as np

from IOOM import IOOM

#Testing IOOM on a toy data set
d = 10 #Dimension of the data (number of features)
K = 2 #Number of clusters
n = 100 #Number of individuals

#We opposite set parameters for the two clusters
params_1 = [0, 0, 0, 0.2, 0.4, 0.6, 0.8, 1, 1, 1]
params_2 = [1-i for i in params_1]

X = np.zeros((n, d))
for i in range(d):
    X_1 = np.random.binomial(n=1, p=params_1[i], size=(int(n/K),1))
    X_2 = np.random.binomial(n=1, p=params_2[i], size=(int(n/K),1))
    X[:, i] = np.reshape(np.concatenate((X_1, X_2), axis=0), (n,))

ioom_classifier = IOOM()

Z_true = np.transpose(np.array([[1]*50+[0]*50,[0]*50+[1]*50]))

Z, Theta = ioom_classifier.fit(X, 2, 100, alpha=1, omega=0.1, prop_new_clusts=False)

((np.dot(Z,np.transpose(Z))-np.dot(Z_true,np.transpose(Z_true)))!=0).sum()/(100**2)
