"""Narrow policy-question refinement without classifier or provider changes."""

from __future__ import annotations

import unittest
from uuid import uuid4
from unittest.mock import patch

from app.config import Config

from app.graph import nodes
from app.graph.workflow import build_graph, invoke_graph
from app.rag.service import RAGConfigurationError
from langgraph.checkpoint.memory import InMemorySaver
# Tests never share the developer's PostgreSQL checkpoints; .env may select them.
patch.object(Config, "CHECKPOINT_BACKEND", "memory").start()


def issue_classification():
    return {
        "predicted_intent": "issue_support",
        "confidence_score": 98.85,
        "final_intent": "issue_support",
        "verifier_status": "ACCEPTED_BY_CLASSIFIER",
        "routing": "DIRECT_CLASSIFIER_OUTPUT",
        "latency_ms": 1.0,
        "error": None,
    }


class PolicyQuestionRoutingTests(unittest.TestCase):
    def run_query(self, query: str):
        graph = build_graph(InMemorySaver())
        with patch.object(nodes, "classify_and_verify", return_value=issue_classification()), patch.object(
            nodes, "get_rag_service", side_effect=RAGConfigurationError("providers are not configured")
        ):
            return invoke_graph(query, f"policy-test-{uuid4()}", graph)

    def test_return_policy_questions_use_existing_rag_route(self):
        for query in (
            "What is your return policy?",
            "How many days do I have to return an item?",
            "How do I return an item?",
            "What is your refund policy?",
        ):
            with self.subTest(query=query):
                result = self.run_query(query)
                self.assertEqual(result["predicted_intent"], "issue_support")
                self.assertEqual(result["final_intent"], "issue_support")
                self.assertEqual(result["selected_route"], "store_information")
                self.assertEqual(result["workflow_step"], "rag_unavailable")
                self.assertIn("unavailable", result["response"].lower())

    def test_transactional_return_and_refund_requests_stay_on_support_route(self):
        for query in (
            "I want to return my damaged product.",
            "I want to return my product; what is your return policy?",
            "Please initiate a return.",
            "Where is my refund?",
        ):
            with self.subTest(query=query):
                result = self.run_query(query)
                self.assertEqual(result["final_intent"], "issue_support")
                self.assertNotEqual(result["selected_route"], "store_information")
                self.assertTrue(result.get("workflow_status"))


if __name__ == "__main__":
    unittest.main()
