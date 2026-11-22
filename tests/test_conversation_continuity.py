"""Multi-turn continuity tests using the existing LangGraph checkpointer."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from app.config import Config
from app.graph import nodes
from app.graph.workflow import build_graph, invoke_graph
from app.rag.service import RAGConfigurationError
from app.services import case_repository
from app.services.case_repository import list_demo_records


patch.object(Config, "BUSINESS_DATA_BACKEND", "json").start()
# These modules test the workflows reached after a confirmed request; test_intent_confirmation covers the confirmation step.
patch.object(Config, "INTENT_CONFIRMATION_ENABLED", False).start()
# Unit tests stay offline; the Groq extraction fallback has its own mocked tests.
patch.object(Config, "LLM_EXTRACTION_ENABLED", False).start()
patch.object(Config, "REPLY_TRANSLATION_ENABLED", False).start()
patch.object(Config, "ESCALATION_ENABLED", False).start()
patch.object(Config, "MULTI_INTENT_ENABLED", False).start()
patch.object(nodes, "get_rag_service", side_effect=RAGConfigurationError("OPENAI_API_KEY is not configured")).start()
# Tests never share the developer's PostgreSQL checkpoints; .env may select them.
patch.object(Config, "CHECKPOINT_BACKEND", "memory").start()


def classified(intent, status="ACCEPTED_BY_CLASSIFIER"):
    return {
        "predicted_intent": intent,
        "confidence_score": 99.0,
        "final_intent": intent if status == "ACCEPTED_BY_CLASSIFIER" else None,
        "verifier_status": status,
        "routing": "DIRECT_CLASSIFIER_OUTPUT",
        "latency_ms": 1.0,
        "error": None,
    }


class ConversationContinuityTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_patcher = patch.object(
            case_repository, "DEFAULT_DB_PATH", Path(self.temp_dir.name) / "support.sqlite3"
        )
        self.db_patcher.start()
        self.graph = build_graph()

    def tearDown(self):
        self.db_patcher.stop()
        self.temp_dir.cleanup()

    def turn(self, session, query, intent=None, status="ACCEPTED_BY_CLASSIFIER"):
        if intent is None:
            return invoke_graph(query, session, self.graph)
        with patch.object(nodes, "classify_and_verify", return_value=classified(intent, status)):
            return invoke_graph(query, session, self.graph)

    def test_order_id_answer_resumes_order_tracking_and_keeps_history(self):
        first = self.turn("order-thread", "Where is my order?", "order_tracking")
        self.assertEqual(first["pending_workflow"]["awaiting_slot"], "order_id")
        with patch.object(nodes, "classify_and_verify") as classifier:
            second = self.turn("order-thread", "ORD-1001")
        classifier.assert_not_called()
        self.assertEqual(second["selected_route"], "order_tracking")
        self.assertEqual(second["tool_result"]["order_id"], "ORD-1001")
        self.assertIsNone(second["pending_workflow"])
        self.assertEqual([message["role"] for message in second["messages"]], ["user", "assistant", "user", "assistant"])

    def test_return_workflow_collects_order_then_reason(self):
        first = self.turn("return-thread", "I want to return my order.", "issue_support")
        self.assertEqual(first["pending_workflow"]["awaiting_slot"], "order_id")
        second = self.turn("return-thread", "ORD-1001")
        self.assertEqual(second["pending_workflow"]["awaiting_slot"], "return_reason")
        third = self.turn("return-thread", "It does not fit me.", "issue_support")
        self.assertEqual(third["tool_result"]["outcome"], "demo_request_prepared")
        self.assertEqual(third["tool_result"]["request"]["reason"], "It does not fit me.")
        self.assertIsNone(third["pending_workflow"])

    def test_damaged_workflow_collects_both_order_and_description(self):
        first = self.turn("damage-thread", "My order arrived damaged.", "issue_support")
        self.assertEqual(first["pending_workflow"]["awaiting_slot"], "order_id")
        second = self.turn("damage-thread", "ORD-1001")
        self.assertEqual(second["pending_workflow"]["awaiting_slot"], "damage_description")
        with patch.object(nodes, "classify_and_verify") as classifier:
            third = self.turn("damage-thread", "The box was crushed and the screen is cracked.")
        classifier.assert_not_called()
        self.assertEqual(third["tool_result"]["outcome"], "demo_case_recorded")
        self.assertIn("screen is cracked", third["tool_result"]["case"]["reason"])
        self.assertIn("demonstration case", third["response"].lower())
        self.assertIn("no retailer or courier claim", third["response"].lower())

    def test_hindi_damage_description_is_a_valid_pending_slot(self):
        self.turn("hindi-damage", "My order arrived damaged.", "issue_support")
        self.turn("hindi-damage", "ORD-1001")
        with patch.object(nodes, "classify_and_verify") as classifier:
            result = self.turn("hindi-damage", "\u0921\u093f\u092c\u094d\u092c\u093e \u0915\u0941\u091a\u0932 \u0917\u092f\u093e \u0914\u0930 \u0938\u094d\u0915\u094d\u0930\u0940\u0928 \u091f\u0942\u091f \u0917\u0908")
        classifier.assert_not_called()
        self.assertEqual(result["tool_result"]["outcome"], "demo_case_recorded")

    def test_new_request_calls_classifier_once_and_valid_slot_answer_skips_it(self):
        with patch.object(nodes, "classify_and_verify", return_value=classified("order_tracking")) as classifier:
            first = self.turn("count-thread", "Where is my order?")
        classifier.assert_called_once()
        self.assertEqual(classifier.call_args.args[0], "Where is my order?")
        self.assertEqual(first["pending_workflow"]["awaiting_slot"], "order_id")
        with patch.object(nodes, "classify_and_verify") as classifier:
            self.turn("count-thread", "ORD-1001")
        classifier.assert_not_called()

    def test_invalid_slot_calls_classifier_and_unverified_reply_keeps_pending(self):
        self.turn("invalid-slot", "Where is my order?", "order_tracking")
        with patch.object(nodes, "classify_and_verify", return_value=classified(None, "UNVERIFIED")) as classifier:
            result = self.turn("invalid-slot", "ORD-ABC")
        classifier.assert_called_once()
        self.assertEqual(classifier.call_args.args[0], "ORD-ABC")
        self.assertEqual(result["selected_route"], "pending:order_tracking")
        self.assertEqual(result["pending_workflow"]["status"], "awaiting_input")
        self.assertIn("format is not recognized", result["response"])

    def test_verified_interruption_runs_then_restores_original_pending_workflow(self):
        self.turn("interrupt-stack", "Where is my order?", "order_tracking")
        with patch.object(nodes, "classify_and_verify", return_value=classified("issue_support")) as classifier:
            refund_start = self.turn("interrupt-stack", "Where is my refund?")
        classifier.assert_called_once()
        self.assertEqual(classifier.call_args.args[0], "Where is my refund?")
        self.assertEqual(refund_start["pending_workflow"]["workflow"], "issue_support")
        self.assertEqual(len(refund_start["suspended_workflows"]), 1)

        refund_done = self.turn("interrupt-stack", "ORD-1004")
        self.assertEqual(refund_done["tool_result"]["outcome"], "no_refund_record")
        self.assertEqual(refund_done["pending_workflow"]["workflow"], "order_tracking")
        self.assertEqual(refund_done["pending_workflow"]["awaiting_slot"], "order_id")
        self.assertIn("earlier request", refund_done["response"].lower())

        order_done = self.turn("interrupt-stack", "ORD-1001")
        self.assertEqual(order_done["tool_result"]["order_id"], "ORD-1001")
        self.assertIsNone(order_done.get("pending_workflow"))

    def test_missing_delivery_collects_details_and_never_claims_lost(self):
        first = self.turn("missing-thread", "My order has not arrived.", "issue_support")
        second = self.turn("missing-thread", "ORD-1001")
        self.assertEqual(second["pending_workflow"]["awaiting_slot"], "missing_delivery_details")
        third = self.turn("missing-thread", "Tracking has not moved for four days.", "issue_support")
        self.assertEqual(third["tool_result"]["outcome"], "demo_case_recorded")
        self.assertFalse(third["tool_result"]["parcel_declared_lost"])
        self.assertIn("four days", third["tool_result"]["case"]["details"]["customer_reported_details"])

    def test_product_name_answer_resumes_product_inquiry(self):
        first = self.turn("product-thread", "Do you have this in stock?", "product_inquiry")
        self.assertEqual(first["pending_workflow"]["awaiting_slot"], "product_identifier")
        with patch.object(nodes, "classify_and_verify") as classifier:
            second = self.turn("product-thread", "Laptop Pro")
        classifier.assert_not_called()
        self.assertEqual(second["selected_route"], "product_inquiry")
        self.assertEqual(second["tool_result"]["product_id"], "PROD-001")
        self.assertIsNone(second["pending_workflow"])

    def test_ambiguous_product_answer_asks_again_without_stock_lookup(self):
        self.turn("ambiguous-product", "Do you have an item in stock?", "product_inquiry")
        with patch.object(nodes, "classify_and_verify", return_value=classified(None, "UNVERIFIED")) as classifier, patch.object(nodes, "check_product_stock") as stock_lookup:
            result = self.turn("ambiguous-product", "Laptop Pro or Galaxy S21", None)
        classifier.assert_called_once()
        self.assertEqual(classifier.call_args.args[0], "Laptop Pro or Galaxy S21")
        self.assertEqual(result["selected_route"], "pending:product_inquiry")
        self.assertIn("multiple sample products", result["response"])
        stock_lookup.assert_not_called()

    def test_issue_clarification_accepts_explicit_subtype_then_collects_order(self):
        first = self.turn("issue-choice", "I need help with an issue.", "issue_support")
        self.assertEqual(first["pending_workflow"]["awaiting_slot"], "issue_type")
        second = self.turn("issue-choice", "return")
        self.assertEqual(second["issue_type"], "return_order")
        self.assertEqual(second["pending_workflow"]["awaiting_slot"], "order_id")

    def test_invalid_order_identifier_asks_again_without_running_tool(self):
        self.turn("invalid-id", "Where is my order?", "order_tracking")
        with patch.object(nodes, "classify_and_verify", return_value=classified(None, "UNVERIFIED")) as classifier, patch.object(nodes, "lookup_order_status") as lookup:
            result = self.turn("invalid-id", "ORD-ABC")
        classifier.assert_called_once()
        self.assertEqual(classifier.call_args.args[0], "ORD-ABC")
        self.assertEqual(result["selected_route"], "pending:order_tracking")
        self.assertIn("format is not recognized", result["response"])
        lookup.assert_not_called()

    def test_interrupted_workflow_is_suspended_then_resumed(self):
        first = self.turn("interrupt-thread", "Where is my order?", "order_tracking")
        self.assertEqual(first["workflow_status"], "awaiting_input")
        middle = self.turn("interrupt-thread", "What promotions are available?", "store_information")
        self.assertEqual(middle["selected_route"], "store_information")
        self.assertEqual(middle["pending_workflow"]["workflow"], "order_tracking")
        self.assertEqual(middle["pending_workflow"]["status"], "awaiting_input")
        self.assertIn("earlier request", middle["response"].lower())
        last = self.turn("interrupt-thread", "ORD-1001")
        self.assertEqual(last["selected_route"], "order_tracking")
        self.assertEqual(last["tool_result"]["order_id"], "ORD-1001")

    def test_session_isolation_prevents_pending_order_leakage(self):
        self.turn("session-a", "Where is my order?", "order_tracking")
        with patch.object(nodes, "classify_and_verify", return_value=classified(None, "OUT_OF_SCOPE")):
            other = self.turn("session-b", "ORD-1001", None)
        self.assertEqual(other["selected_route"], "out_of_scope")
        self.assertIsNone(other.get("pending_workflow"))
        resumed = self.turn("session-a", "ORD-1001")
        self.assertEqual(resumed["selected_route"], "order_tracking")

    def test_explicit_confirmation_is_required_by_confirmation_gate(self):
        config = {"configurable": {"thread_id": "confirmation-thread"}}
        with patch.object(nodes, "classify_and_verify", return_value=classified(None, "UNVERIFIED")) as classifier:
            initial = self.graph.invoke(
                {
                    "query": "maybe",
                    "session_id": "confirmation-thread",
                    "pending_workflow": {
                        "workflow": "future_irreversible_action",
                        "status": "awaiting_confirmation",
                        "awaiting_slot": "confirmation",
                        "prompt": "Confirm or cancel.",
                    },
                    "confirmation_required": True,
                    "confirmation_status": "pending",
                },
                config,
            )
        classifier.assert_called_once()
        self.assertEqual(classifier.call_args.args[0], "maybe")
        self.assertIn("explicit confirmation", initial["response"])
        self.assertEqual(initial["confirmation_status"], "pending")
        with patch.object(nodes, "classify_and_verify") as classifier:
            confirmed = self.graph.invoke(
                {"query": "हाँ", "session_id": "confirmation-thread"}, config
            )
        classifier.assert_not_called()
        self.assertEqual(confirmed["confirmation_status"], "confirmed")
        self.assertEqual(confirmed["tool_result"]["status"], "not_executed")

    def test_handoff_summary_uses_same_session_conversation_context(self):
        self.turn("handoff-thread", "My order has not arrived.", "issue_support")
        result = self.turn("handoff-thread", "I want to speak to an agent.", "customer_care_support")
        record = list_demo_records("customer_handoff", "handoff-thread")[0]
        self.assertIn("My order has not arrived", record["summary"])
        self.assertFalse(result["tool_result"]["live_agent_contacted"])


if __name__ == "__main__":
    unittest.main()
