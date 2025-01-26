from fastapi.testclient import TestClient
from llm.main import app

client = TestClient(app)

def test_upload_file():
    response = client.post("/upload/", files={"file": ("test.json", b'{"content": "Test tweet"}', "application/json")})
    assert response.status_code == 200

def test_plot_sentiment():
    response = client.post("/plot_sentiment/", files={"file": ("test.json", b'{"content": "Test tweet"}', "application/json")})
    assert response.status_code == 200
