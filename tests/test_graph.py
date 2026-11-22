"""LangGraph routing tests with the classifier mocked by default."""

import os
import unittest
from uuid import uuid4
from unittest.mock import patch

from app.config import Config
from app.graph import nodes
from app.graph.routing import INTENT_LABELS
from app.graph.workflow import build_graph, invoke_graph
from app.rag.service import RAGConfigurationError


patch.object(Config, "BUSINESS_DATA_BACKEND", "json").start()
# Unit tests stay offline; the Groq extraction fallback has its own mocked tests.
patch.object(Config, "LLM_EXTRACTION_ENABLED", False).start()
patch.object(Config, "REPLY_TRANSLATION_ENABLED", False).start()
patch.object(Config, "ESCALATION_ENABLED", False).start()
patch.object(Config, "MULTI_INTENT_ENABLED", False).start()
patch.object(nodes, "get_rag_service", side_effect=RAGConfigurationError("OPENAI_API_KEY is not configured")).start()
# Tests never share the developer's PostgreSQL checkpoints; .env may select them.
patch.object(Config, "CHECKPOINT_BACKEND", "memory").start()


def classifier_result(
    predicted_intent=None,
    final_intent=None,
    verifier_status="ACCEPTED_BY_CLASSIFIER",
    error=None,
):
    return {
        "predicted_intent": predicted_intent,
        "confidence_score": 98.0,
        "final_intent": final_intent,
        "verifier_status": verifier_status,
        "routing": "DIRECT_CLASSIFIER_OUTPUT",
        "latency_ms": 1.0,
        "error": error,
    }


class LangGraphRoutingTests(unittest.TestCase):
    def run_result(self, result, query="test query", session_id=None):
        graph = build_graph()
        with patch.object(nodes, "classify_and_verify", return_value=result):
            return invoke_graph(query, session_id or str(uuid4()), graph)

    def test_order_tracking_route(self):
        result = self.run_result(
            classifier_result("order_tracking", "order_tracking"),
            query="Where is order ORD-1001?",
        )
        self.assertEqual(result["selected_route"], "order_tracking")
        self.assertEqual(result["tool_result"]["outcome"], "found")
        self.assertTrue(result["tool_result"]["lookup_performed"])
        self.assertIn("demonstration data", result["response"].lower())

    def test_product_inquiry_route(self):
        result = self.run_result(
            classifier_result("product_inquiry", "product_inquiry"),
            query="Do you have Laptop Pro in stock?",
        )
        self.assertEqual(result["selected_route"], "product_inquiry")
        self.assertEqual(result["tool_result"]["outcome"], "found")
        self.assertEqual(result["tool_result"]["name"], "Laptop Pro")
        self.assertTrue(result["tool_result"]["lookup_performed"])

    def test_billing_and_payment_route_answers_methods_and_invoices(self):
        result = self.run_result(classifier_result("billing_and_payment", "billing_and_payment"), query="Can I pay with UPI?")
        self.assertEqual(result["selected_route"], "payment")
        self.assertEqual(result["tool_result"]["outcome"], "answered")
        self.assertIn("UPI", result["response"])
        invoice = self.run_result(classifier_result("billing_and_payment", "billing_and_payment"), query="mujhe invoice chahiye")
        self.assertIn("invoice is emailed", invoice["response"])

    def test_issue_support_asks_for_problem_type_when_ambiguous(self):
        result = self.run_result(
            classifier_result("issue_support", "issue_support"), query="I was charged twice"
        )
        self.assertEqual(result["selected_route"], "issue_support:clarification")
        self.assertEqual(result["tool_result"]["outcome"], "clarification_required")
        self.assertIn("Which issue", result["response"])
    def test_delivery_options_route_answers_by_city(self):
        result = self.run_result(classifier_result("delivery_options", "delivery_options"), query="क्या आप बैंगलोर में डिलीवरी करते हैं?")
        self.assertEqual(result["selected_route"], "shipping")
        self.assertIn("Yes, we deliver to Bengaluru", result["response"])
        self.assertIn("Same-day delivery", result["response"])
        abroad = self.run_result(classifier_result("delivery_options", "delivery_options"), query="can you deliver internationally?")
        self.assertIn("only within India", abroad["response"])

    def test_shipping_cost_route_includes_fees(self):
        result = self.run_result(classifier_result("shipping_cost", "shipping_cost"), query="shipping fee for Chennai?")
        self.assertEqual(result["selected_route"], "shipping")
        self.assertIn("₹49", result["response"])
        self.assertNotIn("Same-day", result["response"])

    def test_store_information_route(self):
        result = self.run_result(
            classifier_result("store_information", "store_information"),
            query="Are there any sales?",
        )
        self.assertEqual(result["selected_route"], "store_information")
        self.assertEqual(result["tool_result"]["outcome"], "found")
        self.assertIn("not live or redeemable", result["response"])
    def test_out_of_scope_route_does_not_run_business_workflow(self):
        result = self.run_result(
            classifier_result(None, None, "OUT_OF_SCOPE"),
            query="Who won the game?",
        )
        self.assertEqual(result["selected_route"], "out_of_scope")
        self.assertEqual(result["tool_result"]["status"], "not_executed")

    def test_all_five_unsupported_intents_are_explicitly_not_executed(self):
        expected = {
            "account_management",
            "add_to_cart",
            "order_management",
            "place_order",
            "remove_from_cart",
        }
        for intent in sorted(expected):
            with self.subTest(intent=intent):
                result = self.run_result(classifier_result(intent, intent))
                self.assertEqual(result["selected_route"], f"unsupported:{intent}")
                self.assertEqual(result["tool_result"]["status"], "unsupported")
                self.assertFalse(result["tool_result"]["executed"])

    def test_unsupported_responses_hide_labels_and_offer_alternatives(self):
        for intent in sorted(nodes.UNSUPPORTED_RESPONSES):
            with self.subTest(intent=intent):
                response = self.run_result(classifier_result(intent, intent))["response"]
                self.assertNotIn(intent, response)
                self.assertNotIn("No business operation", response)
                self.assertIn(nodes.UNSUPPORTED_RESPONSES[intent], response)
                self.assertIn(nodes.SUPPORTED_ALTERNATIVES, response)

    def test_corrected_intent_routes_by_final_intent(self):
        result = self.run_result(
            classifier_result("order_tracking", "issue_support", "CORRECTED_INTENT"),
            query="Where is my refund for order ORD-1001?",
        )
        self.assertEqual(result["predicted_intent"], "order_tracking")
        self.assertEqual(result["final_intent"], "issue_support")
        self.assertEqual(result["selected_route"], "issue_support:refund_status")
        self.assertEqual(result["tool_result"]["outcome"], "found")
    def test_correct_intent_routes_by_verified_final_intent(self):
        result = self.run_result(
            classifier_result("shipping_cost", "shipping_cost", "CORRECT_INTENT")
        )
        self.assertEqual(result["verifier_status"], "CORRECT_INTENT")
        self.assertEqual(result["selected_route"], "shipping")

    def test_unknown_ecommerce_intent_does_not_run_business_workflow(self):
        result = self.run_result(
            classifier_result(None, None, "UNKNOWN_ECOMMERCE_INTENT"),
            query="Can you change my delivery address?",
        )
        self.assertEqual(result["selected_route"], "unknown_intent")
        self.assertEqual(result["tool_result"]["status"], "not_executed")

    def test_groq_verification_error_fails_closed(self):
        result = self.run_result(
            classifier_result(
                "order_tracking", None, "UNVERIFIED",
                "Groq verification failed: GROQ_API_KEY is not configured",
            )
        )
        self.assertEqual(result["selected_route"], "verification_error")
        self.assertEqual(result["final_intent"], None)
        self.assertEqual(result["tool_result"]["status"], "not_executed")
        self.assertIn("GROQ_API_KEY", result["error"])

    def test_missing_or_unknown_final_intent_uses_safe_fallback(self):
        for final_intent in (None, "not_a_trained_intent"):
            with self.subTest(final_intent=final_intent):
                result = self.run_result(
                    classifier_result("order_tracking", final_intent)
                )
                self.assertEqual(result["selected_route"], "unknown_intent")
                self.assertEqual(result["tool_result"]["status"], "not_executed")

    def test_separate_thread_ids_keep_conversations_isolated(self):
        graph = build_graph()

        def classify_by_query(query, history=None):
            intent = "order_tracking" if "order" in query.lower() else "product_inquiry"
            return classifier_result(intent, intent)

        with patch.object(nodes, "classify_and_verify", side_effect=classify_by_query):
            first_a = invoke_graph("Track my order", "customer-a", graph)
            first_b = invoke_graph("Check this product", "customer-b", graph)
            second_a = invoke_graph("Another order question", "customer-a", graph)

        self.assertEqual(first_a["selected_route"], "order_tracking")
        self.assertEqual(first_b["selected_route"], "product_inquiry")
        self.assertEqual(second_a["selected_route"], "order_tracking")
        self.assertEqual(len(second_a["messages"]), 4)
        self.assertEqual(len(first_b["messages"]), 2)
        self.assertNotIn("Track my order", [m["content"] for m in first_b["messages"]])

    def test_english_query_flows_through_classifier_node(self):
        query = "Where is my order?"
        result = self.run_result(
            classifier_result("order_tracking", "order_tracking"), query=query
        )
        self.assertEqual(result["messages"][0], {"role": "user", "content": query})
        self.assertEqual(result["final_intent"], "order_tracking")

    def test_hindi_query_flows_through_classifier_node(self):
        query = "à¤®à¥‡à¤°à¤¾ à¤‘à¤°à¥à¤¡à¤° à¤•à¤¹à¤¾à¤ à¤ªà¤¹à¥à¤à¤šà¤¾ à¤¹à¥ˆ?"
        result = self.run_result(
            classifier_result("order_tracking", "order_tracking"), query=query
        )
        self.assertEqual(result["messages"][0], {"role": "user", "content": query})
        self.assertEqual(result["final_intent"], "order_tracking")

    def test_route_table_covers_original_and_new_labels(self):
        self.assertEqual(len(INTENT_LABELS), 20)
        self.assertEqual(INTENT_LABELS[15:], [
            "exchange_request", "goodbye_thanks", "greeting", "product_recommendation", "warranty_inquiry",
        ])


@unittest.skipUnless(
    os.getenv("RUN_REAL_CLASSIFIER_TESTS") == "1",
    "Set RUN_REAL_CLASSIFIER_TESTS=1 to load local weights and possibly call Groq.",
)
class OptionalRealClassifierSmokeTests(unittest.TestCase):
    def test_local_classifier_connects_to_graph(self):
        result = invoke_graph(
            "Where is my order?", f"real-classifier-{uuid4()}"
        )
        self.assertIn(result.get("predicted_intent"), INTENT_LABELS)
        self.assertIsNotNone(result.get("selected_route"))





