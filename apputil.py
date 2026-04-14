import numpy as np
import seaborn as sns
from time import time
from sklearn.cluster import KMeans

# Load diamonds dataset and extract numerical columns
diamonds = sns.load_dataset('diamonds')
numerical_diamonds = diamonds.select_dtypes(include=[np.number])

def kmeans(X, k):
    """
    Perform k-means clustering on data.

    Parameters
    ----------
    X : ndarray
        Input data array of shape (n_samples, n_features)
    k : int
        Number of clusters

    Returns
    -------
    tuple
        centroids : ndarray of shape (k, n_features)
            Cluster centroids
        labels : ndarray of shape (n_samples,)
            Cluster assignment for each sample
    """
    km = KMeans(n_clusters=k)
    km.fit(X)
    centroids = km.cluster_centers_
    labels = km.labels_

    return centroids, labels


def kmeans_diamonds(n, k):
    """
    Perform k-means clustering on the diamonds dataset.

    Parameters
    ----------
    n : int
        Number of rows from the diamonds dataset to use
    k : int
        Number of clusters

    Returns
    -------
    tuple
        centroids : ndarray of shape (k, n_features)
            Cluster centroids
        labels : ndarray of shape (n_samples,)
            Cluster assignment for each sample
    """
    # Get first n rows of numerical diamonds data
    data = numerical_diamonds.iloc[:n].values

    # Run kmeans on this subset
    centroids, labels = kmeans(data, k)

    return centroids, labels


def kmeans_timer(n, k, n_iter):
    """
    Measure average runtime of k-means clustering on diamonds dataset.

    Parameters
    ----------
    n : int
        Number of rows from the diamonds dataset to use
    k : int
        Number of clusters
    n_iter : int
        Number of times to run the clustering

    Returns
    -------
    float
        Average runtime in seconds across all iterations
    """
    times = []

    for _ in range(n_iter):
        start = time()
        kmeans_diamonds(n, k)
        elapsed = time() - start
        times.append(elapsed)

    average_time = np.mean(times)

    return average_time
