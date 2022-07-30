from wsgiref.simple_server import demo_app
from fastapi import FastAPI
from src.routes.clustering import router_clustering
from src.routes.vizualization import router_viz
from src.models import *

app = FastAPI(
    title="Clustering App API",
    description="""FastAPI documentation for Clustering App""",
    contact ={
        'name': 'Demo App',
        'url': 'http://localhost:8501'
    },
    version="0.1.0",
)
app.include_router(router_clustering)
app.include_router(router_viz)

tags_metadata = [
    {
        'name': 'Clustering',
        'descriptions': 'Preprocessing, Dimension Reduction, Clustering'
    },
    {
        'name': 'Vizualization',
        'descriptions': 'Return vizualization images'
    }
]
