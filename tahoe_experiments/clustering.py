# clustering dans espace latent
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import numpy as np

def cluster_input(arr: np.ndarray, 
                  min_k=2, 
                  max_k=10,
                  clustering_algo=KMeans,
                  clustering_params={"random_state":6262, 
                                  "n_init":10}): 
    """
    Cluster any input across a range of number of clusters.
    Any clustering algo accepting a 'n_clusters' argument can be passed.
    """
    assert hasattr(clustering_algo, 'n_clusters'), f"Input clustering algo {clustering_algo} does not have a 'n_clusters' argument."
    
    # test différents nombres de clusters
    silhouette_scores = []
    clusters_list = []
    k_range = range(min_k, max_k+1)
    
    for k in k_range:
        clustering_algo = clustering_algo(n_clusters=k, 
                        **clustering_params)
        clusters = clustering_algo.fit_predict(arr)
        clusters_list.append(clusters)
        score = silhouette_score(arr, clusters)
        silhouette_scores.append(score)

    plt.figure(figsize=(6, 4))
    plt.plot(k_range, silhouette_scores, 'o-')
    plt.xlabel('nombre de clusters')
    plt.ylabel('score silhouette')
    plt.title('qualité clustering espace latent')
    plt.grid(True, alpha=0.3)
    plt.show()

    best_idx = np.argmax(silhouette_scores)
    best_silhouette = silhouette_scores[best_idx]
    best_k = k_range[best_idx]
    best_clusters = clusters_list[best_idx]
    print(f"nombre optimal de clusters: {best_k}")
    return best_k, best_clusters, best_silhouette

def cluster_latent_samples(z: np.ndarray, 
                           min_k=2, 
                           max_k=10,
                           **kwargs):
    """
    Clustering on samples drawn from latent space distributions.
    """
    return cluster_input(z, 
                         min_k=min_k, 
                         max_k=max_k,
                         **kwargs)

def cluster_latent_params(mu: np.ndarray, 
                          kappa: np.ndarray, 
                          min_k=2, 
                          max_k=11,
                          **kwargs):
    """
    Instead of clustering on samples from the distribution described by
    by the decoder output, we can also cluster directly from the decoder's
    distribution parameters for each sample = clustering in distribution
    space = stable embeddings so more convenient for experiments 
    """
    samples = np.concat([mu, kappa], axis=1)
    return cluster_input(samples, 
                         min_k=min_k, 
                         max_k=max_k,
                         **kwargs)