from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert "status" in body
    assert "num_chunks" in body


def test_query_invalid_input_returns_422():
    # empty question should fail Pydantic min_length validation -> 422
    response = client.post("/query", json={"question": ""})
    assert response.status_code == 422


def test_query_missing_field_returns_422():
    response = client.post("/query", json={})
    assert response.status_code == 422


def test_query_happy_path():
    response = client.post("/query", json={"question": "What is the speed limit near schools?"})
    assert response.status_code == 200
    body = response.json()
    assert "answer" in body
    assert "sources" in body
    assert isinstance(body["sources"], list)
