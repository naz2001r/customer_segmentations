import io
import requests
import numpy as np
import pandas as pd
from PIL import Image
import streamlit as st
from st_aggrid import AgGrid
import plotly.graph_objects as go

from src.constants import *


if 'edit' not in st.session_state: st.session_state['edit']=False
if 'df_columns' not in st.session_state: st.session_state['df_columns']=[]

if 'preprocess_data' not in st.session_state: st.session_state['preprocess_data']=[]
if 'dim_red_data' not in st.session_state: st.session_state['dim_red_data']=[]
if 'cluster_labels' not in st.session_state: st.session_state['cluster_labels']=[]

if 'show_input_df' not in st.session_state: st.session_state['show_input_df']=False
if 'show_kelbow' not in st.session_state: st.session_state['show_kelbow']=False
if 'show_silhoutte' not in st.session_state: st.session_state['show_silhoutte']=False
if 'show_component' not in st.session_state: st.session_state['show_component']=False
if 'show_correlation' not in st.session_state: st.session_state['show_correlation']=False
if 'show_feature_imp' not in st.session_state: st.session_state['show_feature_imp']=False

if 'scale_method' not in st.session_state: st.session_state['scale_method']= None
if 'scale_method_prev' not in st.session_state: st.session_state['scale_method_prev']= None
if 'scale_method_update' not in st.session_state: st.session_state['scale_method_update']= False

if 'dim_red_method' not in st.session_state: st.session_state['dim_red_method']= None
if 'dim_red_method_prev' not in st.session_state: st.session_state['dim_red_method_prev']= None
if 'dim_red_method_update' not in st.session_state: st.session_state['dim_red_method_update']= False
if 'n_components' not in st.session_state: st.session_state['n_components']= 2
if 'n_components_prev' not in st.session_state: st.session_state['n_components_prev']= 2

if 'cluster_method' not in st.session_state: st.session_state['cluster_method']= None
if 'cluster_method_prev' not in st.session_state: st.session_state['cluster_method_prev']= None
if 'n_clusters' not in st.session_state: st.session_state['n_clusters']= 2
if 'n_clusters_prev' not in st.session_state: st.session_state['n_clusters_prev']= 2


def get_request(endpoint: str):
    r = requests.get(endpoint)
    return r.json()


def post_request(endpoint: str, 
                 json: dict):
    r = requests.post(endpoint, json = json)
    return r


@st.cache 
def preprocess(df):
    request =  post_request(POST_PREPROCESS_METHOD, {
                'data': df.to_dict(),
                'method': st.session_state.scale_method
            }).json()
    return request['data'], request['columns']


@st.cache 
def dimension_reduction(prep_data):
    return post_request(POST_DIMENSION_REDUCTION_METHOD, {
                'data': prep_data,
                'method': st.session_state.dim_red_method,
                'n_components':st.session_state.n_components 
            }).json()['data']


@st.cache 
def clustering(prep_data):
    return post_request(POST_CLUSTERING_METHOD, {
                'data': prep_data,
                'method': st.session_state.cluster_method,
                'n_clusters':st.session_state.n_clusters 
            }).json()['labels']


@st.cache 
def features_importances(df):
    fig = go.Figure()

    angles = list(df.columns)

    layoutdict = dict(
                radialaxis=dict(
                visible=True,
                range=[0, 1]
                ))

    for i in range(st.session_state.n_clusters ):
        subset = df[df['cluster'] == i]
        data = [np.mean(subset[col]) for col in subset.columns[4:]]
        data.append(data[0])
        fig.add_trace(go.Scatterpolar(
            r=data,
            theta=angles,
            fill='toself',
            name="Cluster " + str(i)))
        
    fig.update_layout(
            polar=layoutdict,
            showlegend=True, 
            width = 900,
            height =900
            )
    return fig


@st.cache()
def convert_df(df):
    #drop duplicates
    df_output = df.drop_duplicates()
    df_output['cluster'] = st.session_state.cluster_labels

    # convert df
    return df_output.to_csv().encode('utf-8')


def sidebar():

    data = st.sidebar.file_uploader('Upload data',type=["xlsx","xls","csv"])

    st.session_state.scale_method = st.sidebar.selectbox("Preprocessing Method", 
                                        get_request(GET_SCALE_METHODS)['scale_method'])

    st.session_state.dim_red_method = st.sidebar.selectbox("Dimension Reduction Method", 
                                        get_request(GET_DIMENSION_REDUCTION_METHODS)['dimension_reduction_methods'])

    if st.session_state.dim_red_method != 'None':
        st.session_state.n_components = st.sidebar.slider("Numbers of Components",
                                            MIN_NUM_COMPONENTS,MAX_NUM_COMPONENTS,MIN_NUM_COMPONENTS)
    
    st.session_state.cluster_method = st.sidebar.selectbox("Cluster Algorithm", 
                                        get_request(GET_CLUSTER_MODELS)['cluster_models'])

    st.session_state.n_clusters = st.sidebar.slider("Numbers of Clusters",
                                    MIN_NUM_CLUSTERS,MAX_NUM_CLUSTERS,MIN_NUM_CLUSTERS)
    
    st.sidebar.markdown('### Options:')
    st.session_state.show_input_df = st.sidebar.checkbox('Show Input Data')
    st.session_state.show_kelbow = st.sidebar.checkbox('Elbow Cluster Recommendation')
    st.session_state.show_silhoutte = st.sidebar.checkbox('Silhoutte Vizualization')
    st.session_state.show_correlation = st.sidebar.checkbox('Cluster Features Correlation')
    st.session_state.show_feature_imp = st.sidebar.checkbox('Features Importances Vizualization')

    if st.session_state.dim_red_method != 'None':
        st.session_state.show_component = st.sidebar.checkbox('Component Vizualization')

    st.session_state.show_cluster_feature_dist = st.sidebar.checkbox('Clusters Features Distribute Vizualization')    

    return data


def app():
    st.subheader('Clustering App')
    st.markdown(
    """
         This streamlit example uses a FastAPI service as backend.
         Visit this URL at `:8000/docs` for FastAPI documentation.
    """
    ) 

    data = sidebar()
    if data != None:
        try:
            df = pd.read_csv(data)
        except:
            df = pd.read_excel(data)

        # preprocessing
        if st.session_state.scale_method_prev != st.session_state.scale_method:
            st.session_state.preprocess_data, st.session_state.preprocess_columns = preprocess(df)
            st.session_state.scale_method_prev = st.session_state.scale_method
            st.session_state.scale_method_update = True

        # dim reduction
        if (st.session_state.dim_red_method_prev != st.session_state.dim_red_method or 
            st.session_state.n_components_prev != st.session_state.n_components or 
            st.session_state.scale_method_update)  and st.session_state.dim_red_method != 'None':

            st.session_state.dim_red_data = dimension_reduction(st.session_state.preprocess_data)

            st.session_state.scale_method_update = False
            st.session_state.dim_red_method_update = True
            st.session_state.dim_red_method_prev = st.session_state.dim_red_method
            st.session_state.n_components_prev = st.session_state.n_components
            
        elif st.session_state.dim_red_method_prev != st.session_state.dim_red_method:
            st.session_state.dim_red_data = st.session_state.preprocess_data

        # clustering
        if (st.session_state.cluster_method_prev != st.session_state.cluster_method or 
            st.session_state.n_clusters_prev != st.session_state.n_clusters or
            st.session_state.dim_red_method_update):

            st.session_state.dim_red_method_update = False
            st.session_state.cluster_labels = clustering(st.session_state.dim_red_data)
            st.session_state.cluster_method_prev = st.session_state.cluster_method
            st.session_state.n_clusters_prev = st.session_state.n_clusters

        df_prep = pd.DataFrame(st.session_state.preprocess_data, columns=st.session_state.preprocess_columns)
        df_prep['cluster'] = st.session_state.cluster_labels

        # viz input tables
        if st.session_state.show_input_df:
            st.markdown('##### Input Data')
            AgGrid(df, editable=st.session_state.edit, height=200, theme = 'streamlit')

        # viz KElbow recommendation
        if st.session_state.show_kelbow:
            st.markdown('##### Elbow Cluster Recommendation')
            kelbow = post_request(POST_KELBOW_VIZ_METHOD, {
                'data': st.session_state.dim_red_data,
                'method':st.session_state.cluster_method,
                'n_clusters': [i for i in range(MIN_NUM_CLUSTERS,MAX_NUM_CLUSTERS+1,2)]
            })
            st.image(Image.open(io.BytesIO(kelbow.content)).convert("RGB"))

        # viz Silhoutte 
        if st.session_state.show_silhoutte:
            st.markdown('##### Silhoutte Vizualization')
            silhoutte = post_request(POST_SILHOUETTE_VIZ_METHOD, {
                'data': st.session_state.dim_red_data,
                'method':st.session_state.cluster_method,
                'n_clusters': st.session_state.n_clusters 
            })
            st.image(Image.open(io.BytesIO(silhoutte.content)).convert("RGB"))

        # viz Correlation 
        if st.session_state.show_correlation:
            st.markdown('##### Cluster Features Correlation Vizualization')
            correlation = post_request(POST_CORRELATION_VIZ_METHOD, {
                'data': df_prep.to_dict(),
            })
            st.image(Image.open(io.BytesIO(correlation.content)).convert("RGB"))

        # viz Features Importanses
        if st.session_state.show_feature_imp:
            st.markdown('##### Features Importances Vizualization')
            fig = features_importances(df_prep)
            st.plotly_chart(fig)

        # viz Component Vizualization
        if st.session_state.show_component:
            st.markdown('##### Component Vizualization')

            components = ["Component " + str(i+1) for i in range(len(st.session_state.dim_red_data[0]))]

            component_1 = st.selectbox("First Component", components)
            component_2 = st.selectbox("Second Component", components)
            button_components = st.button("Process Component Vizualization",disabled=False)

            if button_components:
                component_viz = post_request(POST_COMPONENT_VIZ_METHOD, {
                    'data': st.session_state.dim_red_data,
                    'labels':st.session_state.cluster_labels,
                    'component_1': component_1,
                    'component_2': component_2
                })
                st.image(Image.open(io.BytesIO(component_viz.content)).convert("RGB"))

        # viz cluster features Dicstribute
        if st.session_state.show_cluster_feature_dist:
            st.markdown('##### Clusters Features Distribute Vizualization')
            features = st.multiselect("Select features", st.session_state.preprocess_columns)
            clusters = st.multiselect("Select clusters label", list(set(st.session_state.cluster_labels)))
            button_feature_dist = st.button("Process",disabled=False)

            if len(features)>0 and len(clusters)>0 and button_feature_dist:
                component_viz = post_request(POST_CLUSTER_FEATURES_DISTIBUTE_VIZ_METHOD, {
                    'data': df_prep.to_dict(),
                    'clusters': clusters,
                    'features': features
                })
                st.image(Image.open(io.BytesIO(component_viz.content)).convert("RGB"))

        # download data
        st.sidebar.markdown('### Download:')
        st.sidebar.download_button(
            "Press to Download",
            convert_df(df),
            "results.csv",
            "text/csv",
            key='download-csv'
        )


if __name__ == '__main__':
    app()