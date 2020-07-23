import itertools
import time
import warnings

import numpy as np
import matplotlib.pyplot as plt

from sklearn import cluster, datasets
from sklearn.preprocessing import StandardScaler
from itertools import cycle, islice

np.random.seed(0)
import matplotlib as mpl
from scipy import linalg
from sklearn import mixture
from models.CMGMM import CMGMM
from models.IGMM import IGMM

color_iter = itertools.cycle([ 'red','navy', 'c', 'cornflowerblue', 'gold', 'darkorange',u'#ffed6f',u'#81b1d2',u'#0072B2',u'#FFB5B8',u'#8dd3c7',u'#E5E5E5','blue'])


def plot_results(X, Y, means, covariances, index, title):
    splot = plt.subplot(2, 1, 1 + index)
    for i, (mean, covar, color) in enumerate(zip(
            means, covariances, color_iter)):
        v, w = linalg.eigh(covar)
        v = 2. * np.sqrt(2.) * np.sqrt(v)
        u = w[0] / linalg.norm(w[0])
        # as the DP will not use every component it has access to
        # unless it needs it, we shouldn't plot the redundant
        # components.
        if not np.any(Y == i):
            continue
        plt.scatter(X[Y == i, 0], X[Y == i, 1], .8, color=color)

        # Plot an ellipse to show the Gaussian component
        angle = np.arctan(u[1] / u[0])
        angle = 180. * angle / np.pi  # convert to degrees
        ell = mpl.patches.Ellipse(mean, v[0], v[1], 180. + angle, color=color)
        ell.set_clip_box(splot.bbox)
        ell.set_alpha(0.5)
        splot.add_artist(ell)

    plt.xlim(-2., 2)
    plt.ylim(-2., 2.)
    plt.title(title)
    plt.xticks(())
    plt.yticks(())


def plot_samples(X, Y, n_components, index, title):
    plt.subplot(2, 1, 2 + index)
    for i, color in zip(range(n_components), color_iter):
        # as the DP will not use every component it has access to
        # unless it needs it, we shouldn't plot the redundant
        # components.
        if not np.any(Y == i):
            continue
        plt.scatter(X[Y == i, 0], X[Y == i, 1], .8, color=color)

    plt.xlim(-6., 4. * np.pi - 6.)
    plt.ylim(-5., 5.)
    plt.title(title)
    plt.xticks(())
    plt.yticks(())


# Parameters
n_samples = 6000

# Generate random sample following a sine curve
np.random.seed(0)
n_samples = 500
Xcircle, y = datasets.make_blobs(n_samples=1750, centers=3, cluster_std=0.25,
                            random_state=2,center_box=(0,0))

plt.figure(figsize=(2, 1))


print(Xcircle.shape)
# Fit a Gaussian mixture with EM using ten components
cmgmm = IGMM(min_components=3, max_components=12)
ALGO ="IGMM"
cmgmm.fit(Xcircle)
plot_results(Xcircle, cmgmm.predict(Xcircle), cmgmm.means_, cmgmm.covariances_, 0, ALGO+' INITIAL')
#plt.show()

Xmoon,y = datasets.make_moons(n_samples=n_samples, noise=0.05)

cmgmm.fit(Xmoon)
X = np.concatenate((Xcircle , Xmoon), axis=0)
plot_results(X, cmgmm.predict(X), cmgmm.means_, cmgmm.covariances_, 1, ALGO+" ADAPTED")



plt.show()