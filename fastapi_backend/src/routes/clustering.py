from fastapi import APIRouter
try:
    from src.models import *
except:
    from fastapi_backend.src.models import *


router_clustering = APIRouter()


@router_clustering.get("/scale_methods", tags = ['Clustering'])
def get_scale_methods():
    """Get list of scale methods"""
    try:
        from src.constants import scale_method
    except:
        from fastapi_backend.src.constants import scale_method
    return {'scale_method': list(scale_method.keys())}


@router_clustering.get("/dimension_reduction_methods", tags = ['Clustering'])
def get_scale_methods():
    """Get list of dimension reduction methods"""
    try:
        from src.constants import dimension_reduction_methods
    except:
        from fastapi_backend.src.constants import dimension_reduction_methods
    return {'dimension_reduction_methods': list(dimension_reduction_methods.keys())}


@router_clustering.get("/cluster_models", tags = ['Clustering'])
def get_scale_methods():
    """Get list of cluster models"""
    try:
        from src.constants import cluster_models
    except:
        from fastapi_backend.src.constants import cluster_models
    return {'cluster_models': list(cluster_models.keys())}


@router_clustering.post("/preprocess", tags = ['Clustering'])
def preprocess_df(input_data:PreprocessObject):
    import pandas as pd
    from src.constants import scale_method

    assert input_data.method in scale_method.keys()

    scaler = scale_method[input_data.method]

    df = input_data.to_df()

    #drop duplicates
    df = df.drop_duplicates()

    # get dummies binary features
    binary_col = [name for name in df.columns if len(set(df[name]))==2]
    df = pd.get_dummies(df, drop_first=True, columns = binary_col)

    #get dummies
    df = pd.get_dummies(df, drop_first=False)

    if scaler != 'None':
        df_scale = scaler.fit_transform(df)
        return {
            'data': df_scale.tolist(),
            'columns': df.columns.tolist()
        }
    return {
        'data': df.values.tolist(),
        'columns': df.columns.tolist()
    }


@router_clustering.post("/dimension_reduction", tags = ['Clustering'])
def dimension_reduction_df(input_data:DimRedObject):
    from src.constants import dimension_reduction_methods

    assert len(input_data.data[0]) > input_data.n_components
    assert input_data.method in dimension_reduction_methods.keys()

    if dimension_reduction_methods[input_data.method] == 'None':
        return {'data': input_data.data}

    data  = input_data.to_array()

    dim_red_method = dimension_reduction_methods[input_data.method]
    if input_data.method == 't-SNE':
        dim_red_method = dim_red_method( n_components = input_data.n_components, random_state = input_data.random_state,
                                         init = input_data.init_method)
    else: 
        dim_red_method = dim_red_method( n_components = input_data.n_components, random_state = input_data.random_state)

    data = dim_red_method.fit_transform(data)
    return {'data': data.tolist()}


@router_clustering.post("/clustering", tags = ['Clustering'])
def clustering(input_data:ClusterObject):
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

    model.fit(data)

    return {
        'data': data.tolist(), 
        'labels': model.labels_.tolist()
        }