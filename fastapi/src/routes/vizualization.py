import io
from src.models import *
from fastapi import APIRouter
from starlette.responses import Response

import matplotlib.pyplot as plt


router_viz = APIRouter()


@router_viz.get("/all_colors", tags = ['Vizualization'])
def get_all_colors():
    """Get list of colors"""
    from src.constants import colors
    return {'colors': colors}


@router_viz.post("/kelbow_viz", tags = ['Vizualization'])
def viz_features_importanses(input_data:KElbowObject):
    from yellowbrick.cluster import KElbowVisualizer
    from src.constants import cluster_models

    assert len(input_data.n_clusters) > 0
    assert input_data.method in cluster_models.keys()

    data  = input_data.to_array()

    # init model
    try:
        model = cluster_models[input_data.method](n_clusters = input_data.n_clusters,
                                                random_state = input_data.random_state)
    except:
        model = cluster_models[input_data.method](n_clusters = input_data.n_clusters)

    # Kelbow vizualization
    frequency_visualizer = KElbowVisualizer(model, k=input_data.n_clusters,  size=(720, 720))

    frequency_visualizer.fit(data)    # Fit the data to the visualizer

    # save img
    io_object = io.BytesIO()
    frequency_visualizer.show(outpath=io_object)   
    plt.close()

    return Response(io_object.getvalue(), media_type="image/png")


@router_viz.post("/silhouette_viz", tags = ['Vizualization'])
def viz_silhouette(input_data:ClusterObject):

    from yellowbrick.cluster import SilhouetteVisualizer
    from src.constants import cluster_models

    assert input_data.n_clusters >= 2
    assert input_data.method in cluster_models.keys()

    data  = input_data.to_array()

    # init model
    try:
        model = cluster_models[input_data.method](n_clusters = input_data.n_clusters,
                                                random_state = input_data.random_state)
    except:
        model = cluster_models[input_data.method](n_clusters = input_data.n_clusters)

    # viz silhouette
    visualizer = SilhouetteVisualizer(model)
    visualizer.fit(data) 
    
    # save img
    io_object = io.BytesIO()
    visualizer.show(outpath=io_object) 
    plt.close()

    return Response(io_object.getvalue(), media_type="image/png")


@router_viz.post("/component_viz", tags = ['Vizualization'])
def viz_dim_red_components(input_data:ComponentVizObject):
    import random
    import seaborn as sns
    from src.constants import colors

    df = input_data.to_component_viz_df()

    x = df[input_data.component_2]
    y = df[input_data.component_1]

    fig = plt.figure(figsize=(10, 8))
    sns.scatterplot(x, y, hue=df['Cluster'], palette = random.choices(colors, k=len(set(df['Cluster']))))

    plt.title('Clusters by PCA Components', fontsize=20)
    plt.xlabel("Component 2", fontsize=18)
    plt.ylabel("Component 1", fontsize=18)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)

    # save img
    io_object = io.BytesIO()
    plt.savefig(io_object)
    plt.close()

    return Response(io_object.getvalue(), media_type="image/png")


@router_viz.post("/correlation_viz", tags = ['Vizualization'])
def viz_correlation(input_data:BaseDataFrameObject):

    df = input_data.to_df()

    # create corr viz
    plt.figure(figsize=(12,6),dpi=200)
    df.corr()["cluster"].iloc[:-1].sort_values().plot(kind="bar")
    plt.title("Correlation between Clusters and Features")

    # save img
    io_object = io.BytesIO()
    plt.savefig(io_object, format = 'png')
    plt.close()

    return Response(io_object.getvalue(), media_type="image/png")


@router_viz.post("/clusters_features_distribute_viz", tags = ['Vizualization'])
def viz_clusters_features_distribute(input_data:ClusterFeaturesViz):

    import random
    import numpy as np
    from src.constants import colors

    clusters = input_data.clusters
    features = input_data.features
    df = input_data.to_df()

    colors = random.choices(colors, k=len(clusters))
    dim_clusters = len(clusters)
    dim_features = len(features)

    fig, axes = plt.subplots(dim_clusters, dim_features, figsize=(24, 12))
    i = 0
    test_cluster = df.loc[df['cluster'] == clusters[0]]
    for ax in (axes.flatten()):
        if i % dim_clusters == 0 and i != 0:
            test_cluster = df.loc[df['cluster'] == clusters[i // dim_features]]
        col = features[i % dim_features]
        y = test_cluster[col]
        x = [i for i in range(len(y))]
        ax.bar(x, y, color=colors[i//dim_features])
        ax.set_ylabel(col, fontsize=14)
        ax.set_title("Cluster " + str(clusters[i // dim_features]), fontsize=16)
        ax.hlines(np.mean(df[col]), 0, len(y))
        plt.subplots_adjust(wspace=.5, hspace=.5)
        i += 1
    
    io_object = io.BytesIO()
    plt.savefig(io_object, format = 'png')
    plt.close()

    return Response(io_object.getvalue(), media_type="image/png")
