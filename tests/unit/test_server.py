"""
Unit tests for the interactive web dashboard and REST API server.
"""

from starlette.testclient import TestClient
from trustworthy_rag.server.app import app


def test_server_health():
    client = TestClient(app)
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "online"
    assert "ports" in data
    assert data["ports"]["frontend"] == 2217


def test_server_index_html():
    client = TestClient(app)
    response = client.get("/")
    assert response.status_code == 200
    assert "Trustworthy RAG" in response.text
    assert "text/html" in response.headers["content-type"]


def test_server_query_clean():
    client = TestClient(app)
    payload = {
        "query": "When was the corporation founded?",
        "condition": "clean"
    }
    response = client.post("/api/query", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "baseline" in data
    assert "proposed" in data
    assert "scores" in data["proposed"]
    assert "passages" in data


def test_server_clarify_endpoint():
    client = TestClient(app)
    # 1. Trigger clarify or establish a session
    from trustworthy_rag.server.app import get_pipeline
    p = get_pipeline()
    mgr = p["dialogue_mgr"]
    mgr.initiate_clarification(
        session_id="test_sess_99",
        query="What year was it founded?",
        missing_entities=["corporation"],
        candidate_passages=[],
    )

    # 2. Call /api/clarify
    res = client.post("/api/clarify", json={
        "session_id": "test_sess_99",
        "clarification": "ACME Corporation in 1998"
    })
    assert res.status_code == 200
    data = res.json()
    assert data["action"] == "ANSWER"
    assert "ACME Corporation in 1998" in data["clarified_query"]

