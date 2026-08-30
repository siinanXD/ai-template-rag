from fastapi.testclient import TestClient


def test_health(client: TestClient) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_ready(client: TestClient) -> None:
    response = client.get("/ready")
    assert response.status_code == 200
    assert response.json() == {"status": "ready"}


def test_ready_when_database_fails(client: TestClient) -> None:
    from app.core.db import get_db
    from app.main import app

    class BrokenSession:
        def execute(self, *_args, **_kwargs):
            raise RuntimeError("db down")

    def _broken_db():
        yield BrokenSession()

    app.dependency_overrides[get_db] = _broken_db
    response = client.get("/ready")
    assert response.status_code == 503
    assert response.json() == {"status": "not_ready"}
