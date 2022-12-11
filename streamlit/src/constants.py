
### Backend method
BACKEND = "http://fastapi-backend:8000/"

GET_SCALE_METHODS = BACKEND + 'scale_methods'
GET_DIMENSION_REDUCTION_METHODS = BACKEND + 'dimension_reduction_methods'
GET_CLUSTER_MODELS = BACKEND + 'cluster_models'

POST_PREPROCESS_METHOD = BACKEND + 'preprocess'
POST_DIMENSION_REDUCTION_METHOD = BACKEND + 'dimension_reduction'
POST_CLUSTERING_METHOD = BACKEND + 'clustering'

POST_KELBOW_VIZ_METHOD = BACKEND + 'kelbow_viz'
POST_SILHOUETTE_VIZ_METHOD = BACKEND + 'silhouette_viz'
POST_COMPONENT_VIZ_METHOD = BACKEND + 'component_viz'
POST_CORRELATION_VIZ_METHOD = BACKEND + 'correlation_viz'
POST_CLUSTER_FEATURES_DISTIBUTE_VIZ_METHOD = BACKEND + 'clusters_features_distribute_viz'


### Dimention Reduction
MIN_NUM_COMPONENTS = 2
MAX_NUM_COMPONENTS = 10

### Clustering
MIN_NUM_CLUSTERS = 2
MAX_NUM_CLUSTERS = 10
