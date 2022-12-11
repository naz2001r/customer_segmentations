from fastapi.testclient import TestClient
from fastapi_backend.app import app

client = TestClient(app)


def test_scale_methods():
    response = client.get("/scale_methods")
    assert response.status_code == 200

def test_dimension_reduction_methods():
    response = client.get("/dimension_reduction_methods")
    assert response.status_code == 200

def test_cluster_models():
    response = client.get("/cluster_models")
    assert response.status_code == 200   