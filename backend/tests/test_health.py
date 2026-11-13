"""Tests for application startup and the health endpoint."""


class TestAppStartup:
    """The backend must start regardless of Groq API key presence."""

    def test_starts_without_api_key(self, client_no_key):
        """Application starts and serves requests without GROQ_API_KEY."""
        response = client_no_key.get("/api/health")
        assert response.status_code == 200

    def test_starts_with_api_key(self, client_with_key):
        """Application starts and serves requests with GROQ_API_KEY."""
        response = client_with_key.get("/api/health")
        assert response.status_code == 200


class TestHealthEndpoint:
    """Health endpoint must reflect backend availability and Groq state."""

    def test_healthy_without_api_key(self, client_no_key):
        """Without an API key the backend is healthy but Groq is not configured."""
        data = client_no_key.get("/api/health").json()
        assert data["status"] == "healthy"
        assert data["groq_client"] == "not_configured"

    def test_healthy_with_api_key(self, client_with_key):
        """With an API key the backend is healthy and Groq is initialized."""
        data = client_with_key.get("/api/health").json()
        assert data["status"] == "healthy"
        assert data["groq_client"] == "initialized"

    def test_response_keys(self, client_no_key):
        """Health response contains exactly the expected keys."""
        data = client_no_key.get("/api/health").json()
        assert set(data.keys()) == {"status", "groq_client"}
