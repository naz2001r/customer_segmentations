# **Customer segmentations**

## **Description**
Clustering demo tool for different data. 

For this app is using `FastAPI` for the backend service and `streamlit` for the frontend service. `docker-compose` orchestrates the two services and allows communication between them.

## **Setup**
To run the example in a machine running Docker and docker-compose, run:

    docker-compose build
    docker-compose up

To visit the FastAPI documentation of the resulting service, visit http://localhost:8000 with a web browser.  
To visit the streamlit UI, visit http://localhost:8501.

Logs can be inspected via:

    docker-compose logs


## **Available methods**:

### Preprocessing method:
- `MinMaxScale`
- `StandardScaler`

### Dimension reduction method:
- `PCA`
- `t-SNE`
- `LDA`

### Clustering algorithms:
- `KMeans`
- `MiniBatchKMeans`
- `SpectralClustering`
- `Birch`
- `AgglomerativeClustering`


## **Streamlit app view**
![Streamlit app view](img/general.png)

## **Vizualizations**
### Elbow Cluster Recommendation
![Elbow Cluster Recommendation](img/KElbow.png)

### Silhoutte Vizualization
![Silhoutte Vizualization](img/Silhoutte.png)

### Cluster Features Correlation Vizualization
![Cluster Features Correlation Vizualization](img/correlation.png)

### Features Importances Vizualization
![Features Importances Vizualization](img/feature_importances.png)

### Component Vizualization
![Component Vizualization](img/component_viz.png)

### Clusters Features Distribute Vizualization
![Clusters Features Distribute Vizualization](img/cluster_feature_distribute.png)
