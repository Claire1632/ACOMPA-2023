from math import sqrt, floor
import numpy as np


def random(ds, k, random_state=42):
    """
    Create random cluster centroids.
    
    Parameters
    ----------
    ds : numpy array
        The dataset to be used for centroid initialization.
    k : int
        The desired number of clusters for which centroids are required.
    Returns
    -------
    centroids : numpy array
        Collection of k centroids as a numpy array.
    """

    np.random.seed(random_state)
    centroids = []
    m = np.shape(ds)[0]

    for _ in range(k):
        r = np.random.randint(0, m-1)
        centroids.append(ds[r])

    return np.array(centroids)


def plus_plus(ds, k, random_state=42):
    """
    Create cluster centroids using the k-means++ algorithm.

    Parameters
    ----------
    ds : numpy array
        The dataset to be used for centroid initialization.
    k : int
        The desired number of clusters for which centroids are required.
    Returns
    -------
    centroids : numpy array
        Collection of k centroids as a numpy array.

    Inspiration from here: https://stackoverflow.com/questions/5466323/how-could-one-implement-the-k-means-algorithm
    """

    np.random.seed(random_state)
    centroids = [ds[0]]

    for _ in range(1, k):
        dist_sq = np.array([min([np.inner(c-x,c-x) for c in centroids]) for x in ds])
        probs = dist_sq/dist_sq.sum()
        cumulative_probs = probs.cumsum()
        r = np.random.rand()
        
        for j, p in enumerate(cumulative_probs):
            if r < p:
                i = j
                break
        
        centroids.append(ds[i])

    return np.array(centroids)


def naive_sharding(ds, k):
    """
    Create cluster centroids using deterministic naive sharding algorithm.
    
    Parameters
    ----------
    ds : numpy array
        The dataset to be used for centroid initialization.
    k : int
        The desired number of clusters for which centroids are required.
    Returns
    -------
    centroids : numpy array
        Collection of k centroids as a numpy array.
    """

    def _get_mean(sums, step):
        """Vectorizable ufunc for getting means of summed shard columns."""
        return sums/step

    n = np.shape(ds)[1]
    m = np.shape(ds)[0]
    centroids = np.zeros((k, n))

    composite = np.mat(np.sum(ds, axis=1))
    ds = np.append(composite.T, ds, axis=1)
    ds.sort(axis=0)

    step = floor(m/k)
    vfunc = np.vectorize(_get_mean)

    for j in range(k):
        if j == k-1:
            centroids[j:] = vfunc(np.sum(ds[j*step:,1:], axis=0), step)
        else:
            centroids[j:] = vfunc(np.sum(ds[j*step:(j+1)*step,1:], axis=0), step)

    return centroids

import pandas as pd
data = pd.read_csv("C:/Users/ADMIN/OneDrive/Desktop/data_test_Km.csv")
data = np.array(data)
print(plus_plus(data, k=3, random_state=42))

# import numpy as np 
# # df = pd.DataFrame({'a':[4.3,4.5,np.nan,np.nan], 
# #  'b':[5.5,5.1,np.nan,0.2], 
# #  'x':[7.5,3.2,2.1,np.nan], 
# #  'y':[7.7,3.1,np.nan,0.5],
# #  'z':[8.1,2.9,2.4,np.nan],
# #  't':[8.5,np.nan,2.6,0.7],
# #  'p':[np.nan,np.nan,3.0,np.nan]}) 
import pandas as pd
df = pd.read_csv("C:/Users/ADMIN/OneDrive/Desktop/data_test.csv")
# df = np.array(df)

print(df)
	
print(df.fillna(df.median()))
print(df)