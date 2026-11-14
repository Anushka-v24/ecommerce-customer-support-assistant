"""Phase 1 smoke tests for Flask and LangGraph."""

import unittest

from app import create_app
from app.graph.workflow import build_test_graph


class FlaskHealthTests(unittest.TestCase):
    def setUp(self) -> None:
        self.client = create_app().test_client()

    def test_health_endpoint(self) -> None:
        response = self.client.get("/api/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json(), {"status": "healthy", "service": "ecom-langgraph"})


class TestGraphTests(unittest.TestCase):
    def test_graph_accepts_query_and_returns_response(self) -> None:
        result = build_test_graph().invoke({"query": "Where is my order?"})
        self.assertEqual(result["query"], "Where is my order?")
        self.assertEqual(result["response"], "The test graph received: Where is my order?")

    def test_graph_handles_blank_query(self) -> None:
        result = build_test_graph().invoke({"query": "   "})
        self.assertEqual(result["response"], "Please provide a query.")


if __name__ == "__main__":
    unittest.main()
