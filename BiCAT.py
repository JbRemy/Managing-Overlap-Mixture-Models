import pandas as pd

from IOMM import IOMM
import utils


df = pd.read_csv('data2.csv')

X = df.as_matrix()[:,1:71]

IOMM_classifier = IOMM(Type='Gaussian')

Z, U, Theta, n_clusters = IOMM_classifier.fit(X, K_init=1, Niter=3000, alpha=1, omega=0.1, prop_new_clusts=True, stochastic=False, burn_in=1000)

Z_2 = Z.copy()
Z_2[Z_2 > 0.5] = 1
Z_2[Z_2 <= 0.5] = 0
U_2 = np.dot(Z_2, np.transpose(Z_2))

utils.plot_similarity(U_2, title='', save_path='/media/sf_Debian-shared-folder/', name='fig_Bicat_simulated_U')

utils.plot_n_clusters(n_clusters, True_n_clusters = None, title='', save_path='/media/sf_Debian-shared-folder/', name='fig_Bicat_n_clusters')