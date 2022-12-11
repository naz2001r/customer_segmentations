from pydantic import BaseModel


class BaseDataFrameObject(BaseModel):
    data: dict

    def to_df(self):
        """Convert pydantic object to pandas dataframe"""
        import pandas as pd
        return pd.DataFrame(self.data)

class BaseNumpyObject(BaseModel):
    data: list

    def to_array(self):
        """Convert pydantic object to numpy array """
        import numpy as np
        return np.array(self.data)


class PreprocessObject(BaseDataFrameObject):
    method: str = 'None'

class DimRedObject(BaseNumpyObject):
    method: str = 'None'
    n_components: int = 2
    random_state: int = 42
    init_method: str = 'random'

class ClusterObject(BaseNumpyObject):
    n_clusters: int
    method: str
    random_state: int = 42

class KElbowObject(ClusterObject):
    n_clusters: list
    timings: bool = True

class ComponentVizObject(BaseModel):
    data: list
    labels: list
    component_1: str
    component_2: str

    def to_component_viz_df(self):
        """Convert pydantic object to pandas dataframe"""
        import pandas as pd

        df_seg_pca_kmeans = pd.DataFrame(self.data, 
                                         columns= ["Component " + str(i+1) for i in range(len(self.data[0]))])
        df_seg_pca_kmeans['Cluster'] = self.labels
        return df_seg_pca_kmeans

class ClusterFeaturesViz(BaseDataFrameObject):
    clusters: list
    features: list