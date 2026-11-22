"""Offline FakeStore client, tool, and graph integration coverage."""

from __future__ import annotations

import json
import unittest
from uuid import uuid4
from io import BytesIO
from unittest.mock import patch

from app.config import Config
from app import create_app
from app.graph import nodes
from app.graph.workflow import build_graph
from app.integrations import fakestore_client
from app.integrations.fakestore_client import FakeStoreClient, FakeStoreError
from langgraph.checkpoint.memory import InMemorySaver
from app.tools import fakestore_tools

# Replies stay in English offline; translation has its own mocked tests.
patch.object(Config, "REPLY_TRANSLATION_ENABLED", False).start()
patch.object(Config, "ESCALATION_ENABLED", False).start()
patch.object(Config, "MULTI_INTENT_ENABLED", False).start()
# Tests never share the developer's PostgreSQL checkpoints; .env may select them.
patch.object(Config, "CHECKPOINT_BACKEND", "memory").start()


PRODUCT = {
    "id": 1, "title": "Classic Shirt", "price": 19.99,
    "description": "A cotton shirt", "category": "clothing", "image": "https://example.test/shirt.png",
    "rating": {"rate": 4.2, "count": 10},
}


class FakeResponse:
    status = 200

    def __init__(self, value):
        self.payload = json.dumps(value).encode()

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def read(self):
        return self.payload


class FakeStoreClientTests(unittest.TestCase):
    def test_list_and_details_use_supported_product_routes_and_validate(self):
        calls = []

        def opener(request, timeout):
            calls.append((request.full_url, timeout))
            return FakeResponse([PRODUCT] if request.full_url.endswith("/products?limit=5") else PRODUCT)

        client = FakeStoreClient("https://catalog.test", timeout=2, opener=opener)
        self.assertEqual(client.list_products(5)[0]["id"], 1)
        self.assertEqual(client.get_product(1)["title"], "Classic Shirt")
        self.assertEqual(calls, [("https://catalog.test/products?limit=5", 2), ("https://catalog.test/products/1", 2)])
        with self.assertRaises(ValueError):
            client.get_product(True)

    def test_malformed_product_response_fails_closed(self):
        client = FakeStoreClient("https://catalog.test", opener=lambda *_a, **_k: FakeResponse({"id": 1}))
        with self.assertRaises(FakeStoreError):
            client.get_product(1)

    def test_timeout_is_wrapped_as_safe_provider_error(self):
        def timeout(*_args, **_kwargs):
            raise TimeoutError("private transport detail")

        client = FakeStoreClient("https://catalog.test", opener=timeout)
        with self.assertRaisesRegex(FakeStoreError, "unavailable or returned invalid JSON"):
            client.get_product(1)

    def test_product_tools_search_and_do_not_claim_stock(self):
        class Client:
            def list_products(self, _limit):
                return [PRODUCT]

            def get_product(self, _product_id):
                return PRODUCT

        with patch.object(Config, "FAKESTORE_ENABLED", True), patch.object(fakestore_tools, "get_fakestore_client", return_value=Client()):
            search = fakestore_tools.search_catalog_products("Classic Shirt")
            stock = fakestore_tools.check_catalog_stock("Classic Shirt")
        self.assertEqual(search["matches"][0]["id"], 1)
        self.assertEqual(stock["outcome"], "inventory_unavailable")
        self.assertIsNone(stock["stock_quantity"])

    def test_explicit_catalog_product_404_is_not_found(self):
        from app.integrations.fakestore_client import FakeStoreNotFound

        class MissingClient:
            def get_product(self, _product_id):
                raise FakeStoreNotFound("not found")

        with patch.object(Config, "FAKESTORE_ENABLED", True), patch.object(
            fakestore_tools, "get_fakestore_client", return_value=MissingClient()
        ):
            result = fakestore_tools.get_catalog_product(999)
        self.assertEqual(result["outcome"], "not_found")

    def test_graph_product_route_selects_catalog_for_search_and_stock(self):
        with patch.object(Config, "FAKESTORE_ENABLED", True), patch.object(
            nodes, "search_catalog_products", return_value={"outcome": "found", "matches": [{"id": 1, "title": "Classic Shirt", "price": 19.99, "category": "clothing"}]}
        ):
            result = nodes.product_inquiry_node({"query": "Search for a classic shirt"})
        self.assertIn("Classic Shirt", result["response"])
        self.assertEqual(result["tool_result"]["source"], "fakestore_demo")

        with patch.object(Config, "FAKESTORE_ENABLED", True), patch.object(
            nodes, "check_catalog_stock", return_value={"outcome": "inventory_unavailable", "matches": [{"id": 1, "title": "Classic Shirt", "price": 19.99, "category": "clothing"}], "stock_quantity": None}
        ):
            result = nodes.product_inquiry_node({"query": "Is a classic shirt in stock?"})
        self.assertIn("cannot confirm current stock", result["response"])
        self.assertIsNone(result["tool_result"]["stock_quantity"])

    def test_disabled_fakestore_uses_explicit_local_product_source(self):
        local = {"outcome": "found", "name": "Laptop Pro", "product_id": "PROD-001", "stock": 8}
        with patch.object(Config, "FAKESTORE_ENABLED", False), patch.object(
            nodes, "check_product_stock", return_value=local
        ) as local_lookup, patch.object(nodes, "search_catalog_products") as catalog_lookup:
            result = nodes.product_inquiry_node({"query": "Laptop Pro"})
        local_lookup.assert_called_once_with("Laptop Pro")
        catalog_lookup.assert_not_called()
        self.assertIn("sample stock", result["response"])
        self.assertIn("not live inventory", result["response"])

    def test_enabled_fakestore_success_never_uses_local_stock(self):
        match = {"id": 1, "title": "Classic Shirt", "price": 19.99, "category": "clothing"}
        with patch.object(Config, "FAKESTORE_ENABLED", True), patch.object(
            nodes, "search_catalog_products", return_value={"outcome": "found", "matches": [match]}
        ), patch.object(nodes, "check_product_stock") as local_lookup:
            result = nodes.product_inquiry_node({"query": "Search for Classic Shirt"})
        local_lookup.assert_not_called()
        self.assertEqual(result["tool_result"]["source"], "fakestore_demo")
        self.assertIn("Classic Shirt", result["response"])

    def test_enabled_fakestore_no_match_is_terminal(self):
        with patch.object(Config, "FAKESTORE_ENABLED", True), patch.object(
            nodes, "search_catalog_products", return_value={"outcome": "not_found", "matches": []}
        ), patch.object(nodes, "check_product_stock") as local_lookup:
            result = nodes.product_inquiry_node({"query": "Search for Missing Gadget"})
        local_lookup.assert_not_called()
        self.assertIn("not find", result["response"])
        self.assertIn("FakeStore", result["response"])

    def test_enabled_fakestore_failure_is_terminal_and_does_not_fallback(self):
        with patch.object(Config, "FAKESTORE_ENABLED", True), patch.object(
            nodes, "search_catalog_products", return_value={"outcome": "unavailable", "matches": []}
        ), patch.object(nodes, "check_product_stock") as local_lookup:
            result = nodes.product_inquiry_node({"query": "Search for Classic Shirt"})
        local_lookup.assert_not_called()
        self.assertIn("unavailable", result["response"])
        self.assertEqual(result["tool_result"]["outcome"], "unavailable")

    def test_enabled_fakestore_stock_quantity_remains_unknown(self):
        match = {"id": 1, "title": "Classic Shirt", "price": 19.99, "category": "clothing"}
        with patch.object(Config, "FAKESTORE_ENABLED", True), patch.object(
            nodes, "check_catalog_stock", return_value={
                "outcome": "inventory_unavailable", "matches": [match], "stock_quantity": None,
            }
        ), patch.object(nodes, "check_product_stock") as local_lookup:
            result = nodes.product_inquiry_node({"query": "Is Classic Shirt in stock?"})
        local_lookup.assert_not_called()
        self.assertIsNone(result["tool_result"]["stock_quantity"])
        self.assertIn("cannot confirm current stock", result["response"])

    def test_enabled_fakestore_generic_availability_asks_for_product(self):
        with patch.object(Config, "FAKESTORE_ENABLED", True), patch.object(
            nodes, "check_catalog_stock", return_value={"outcome": "missing_query", "matches": []}
        ), patch.object(nodes, "check_product_stock") as local_lookup:
            result = nodes.product_inquiry_node({"query": "Check product availability"})
        local_lookup.assert_not_called()
        self.assertEqual(result["workflow_status"], "awaiting_input")
        self.assertEqual(result["pending_workflow"]["awaiting_slot"], "product_identifier")
        self.assertIn("product name", result["response"])

    def test_pending_product_name_uses_fakestore_not_local_product_repository(self):
        state = {
            "query": "Classic Shirt",
            "pending_workflow": {
                "workflow": "product_inquiry", "status": "awaiting_input",
                "awaiting_slot": "product_identifier", "collected": {},
                "required_slots": ["product_identifier"],
                "prompt": "Please provide a product name.",
                "product_task": "availability",
            },
        }
        with patch.object(Config, "FAKESTORE_ENABLED", True), patch.object(
            nodes, "search_catalog_products", return_value={
                "outcome": "found", "matches": [PRODUCT | {"id": 1}],
            }
        ), patch.object(nodes, "search_products") as local_search:
            result = nodes.continuity_node(state)
        local_search.assert_not_called()
        self.assertEqual(result["continuity_action"], "resume_pending")
        self.assertEqual(result["product_name"], "Classic Shirt")

        stock_result = {
            "outcome": "inventory_unavailable",
            "matches": [PRODUCT | {"id": 1}],
            "stock_quantity": None,
        }
        with patch.object(Config, "FAKESTORE_ENABLED", True), patch.object(
            nodes, "check_catalog_stock", return_value=stock_result
        ) as catalog_stock, patch.object(nodes, "check_product_stock") as local_stock:
            reply = nodes.product_inquiry_node({
                **state, **result, "query": "Classic Shirt",
            })
        catalog_stock.assert_called_once_with("Classic Shirt")
        local_stock.assert_not_called()
        self.assertIn("cannot confirm current stock", reply["response"])

    def test_flask_api_runs_graph_with_fake_http_catalog_response(self):
        fakestore_client.get_fakestore_client.cache_clear()
        config = type("TestConfig", (Config,), {"TESTING": True, "FAKESTORE_ENABLED": True, "TURN_LOG_ENABLED": False})
        app = create_app(config)
        fake_opener = lambda *_args, **_kwargs: FakeResponse([PRODUCT])
        classification = {
            "predicted_intent": "product_inquiry", "confidence_score": 99.0,
            "final_intent": "product_inquiry", "verifier_status": "ACCEPTED_BY_CLASSIFIER",
            "routing": "DIRECT_CLASSIFIER_OUTPUT", "latency_ms": 1.0, "error": None,
        }
        with patch.object(Config, "FAKESTORE_ENABLED", True), patch.object(nodes, "classify_and_verify", return_value=classification), patch.object(
            fakestore_client, "urlopen", side_effect=fake_opener
        ), patch("app.graph.workflow.get_graph", return_value=build_graph(InMemorySaver())):
            response = app.test_client().post("/api/chat", json={
                "session_id": f"api-fakestore-{uuid4()}", "message": "Search for Classic Shirt",
            })
        self.assertEqual(response.status_code, 200)
        self.assertIn("Classic Shirt", response.get_json()["answer"])
        self.assertEqual(response.get_json()["intent"], "product_inquiry")
        fakestore_client.get_fakestore_client.cache_clear()


if __name__ == "__main__":
    unittest.main()
