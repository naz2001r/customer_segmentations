from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA, LatentDirichletAllocation
from sklearn.cluster import KMeans, SpectralClustering, Birch, AgglomerativeClustering, MiniBatchKMeans
from sklearn.manifold import TSNE
import matplotlib.colors as mcolors


colors = list(mcolors.TABLEAU_COLORS.keys())

scale_method = {
    'None': 'None',
    'MinMaxScale': MinMaxScaler(),
    'StandardScaler': StandardScaler()
}

dimension_reduction_methods = {
    'None': 'None',
    'PCA': PCA,
    't-SNE': TSNE,
    'LDA':LatentDirichletAllocation
}

cluster_models ={
    'KMeans': KMeans,
    'MiniBatchKMeans': MiniBatchKMeans,
    'SpectralClustering': SpectralClustering,
    'Birch':Birch,
    'AgglomerativeClustering':AgglomerativeClustering
}