from starlette.testclient import TestClient

from .app import app

client = TestClient(app)


def test_read_main():
    response = client.get("/scale_methods")
    assert response.status_code == 200

def test_read_main():
    response = client.get("/dimension_reduction_methods")
    assert response.status_code == 200

def test_read_main():
    response = client.get("/cluster_models")
    assert response.status_code == 200   