# preprocessing pour tahoe
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

def preprocess(data, n_components: int = 50):
    # standardisation
    # scaler = StandardScaler()
    # data_scaled = scaler.fit_transform(sample_data)
    # >> RAPH: We usually dont standardise since we already have 
    # log normalized expression values which has the effect of 
    # 'improving' the distribution of values by homogenizing variance

    # réduction dimensionnalité initiale avec pca
    pca = PCA(n_components=n_components)
    data = pca.fit_transform(data)
    print(f"variance expliquée: {pca.explained_variance_ratio_.sum():.3f}")
    return data, pca