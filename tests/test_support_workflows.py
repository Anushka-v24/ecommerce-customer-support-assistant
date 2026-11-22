"""Service and graph coverage for Phase 4C support workflows."""

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
from app.services.customer_care_service import request_customer_handoff
from app.services.issue_resolver import resolve_issue_type
from app.services.refund_service import get_refund_status
from app.services.return_service import check_return_eligibility, prepare_demo_return_request
from app.services.store_information_service import get_store_information
from app.services.support_case_service import investigate_missing_order, report_damaged_order


# Unit tests must keep demo mutations out of a developer's configured database.
patch.object(Config, "BUSINESS_DATA_BACKEND", "json").start()
# These modules test the workflows reached after a confirmed request; test_intent_confirmation covers the confirmation step.
patch.object(Config, "INTENT_CONFIRMATION_ENABLED", False).start()
# Unit tests stay offline; the Groq extraction fallback has its own mocked tests.
patch.object(Config, "LLM_EXTRACTION_ENABLED", False).start()
patch.object(nodes, "get_rag_service", side_effect=RAGConfigurationError("OPENAI_API_KEY is not configured")).start()
# Tests never share the developer's PostgreSQL checkpoints; .env may select them.
patch.object(Config, "CHECKPOINT_BACKEND", "memory").start()


def classifier_result(intent, status="ACCEPTED_BY_CLASSIFIER"):
    return {
        "predicted_intent": intent,
        "confidence_score": 99.0,
        "final_intent": intent if status == "ACCEPTED_BY_CLASSIFIER" else None,
        "verifier_status": status,
        "routing": "DIRECT_CLASSIFIER_OUTPUT",
        "latency_ms": 1.0,
        "error": None,
    }


class DemoSupportServiceTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = Path(self.temp_dir.name) / "support.sqlite3"
        self.db_patcher = patch.object(case_repository, "DEFAULT_DB_PATH", self.db_path)
        self.db_patcher.start()

    def tearDown(self):
        self.db_patcher.stop()
        self.temp_dir.cleanup()

    def test_refund_status_never_invents_a_status(self):
        no_record = get_refund_status("ORD-1004")
        self.assertEqual(no_record["outcome"], "no_refund_record")
        self.assertIsNone(no_record["refund"])
        self.assertIn("cannot", no_record["error"])
        self.assertEqual(get_refund_status("ORD-9999")["outcome"], "unknown_order")
        self.assertEqual(get_refund_status(None)["outcome"], "missing_order_id")

    def test_refund_status_returns_only_a_stored_demo_value(self):
        with patch(
            "app.services.refund_service.load_refunds",
            return_value={"ORD-1001": {"order_id": "ORD-1001", "status": "demo_pending", "refund_id": "R-DEMO-1"}},
        ):
            result = get_refund_status("ORD-1001")
        self.assertEqual(result["outcome"], "found")
        self.assertEqual(result["refund"]["status"], "demo_pending")

    def test_fictional_refund_scenarios_are_read_from_the_fixture(self):
        expected = {
            "ORD-1001": "not_initiated",
            "ORD-1002": "processing",
            "ORD-1003": "completed",
        }
        for order_id, status in expected.items():
            with self.subTest(order_id=order_id):
                result = get_refund_status(order_id)
                self.assertEqual(result["outcome"], "found")
                self.assertEqual(result["refund"]["status"], status)
        self.assertEqual(get_refund_status("ORD-1004")["outcome"], "no_refund_record")

    def test_return_policy_and_eligibility_are_not_overclaimed(self):
        result = check_return_eligibility("ORD-1001")
        self.assertEqual(result["outcome"], "manual_review_required")
        self.assertEqual(result["eligibility"], "not_determined")
        self.assertEqual(result["policy"]["days"], 30)
        self.assertEqual(result["policy"]["condition"], "sealed box")
        self.assertIn("purchase date", result["missing_evidence"])

    def test_return_request_is_only_a_local_demo_draft(self):
        missing_reason = prepare_demo_return_request("ORD-1001", None, "session-a")
        self.assertEqual(missing_reason["outcome"], "missing_reason")
        result = prepare_demo_return_request("ORD-1001", "The size does not fit", "session-a")
        self.assertEqual(result["outcome"], "demo_request_prepared")
        self.assertFalse(result["real_return_submitted"])
        self.assertEqual(result["request"]["status"], "demo_prepared_pending_manual_review")
        self.assertEqual(len(list_demo_records("return_request", "session-a")), 1)

    def test_damaged_order_report_records_product_and_details_as_demo_case(self):
        result = report_damaged_order(
            "ORD-1001", "The screen arrived cracked", "session-damage"
        )
        self.assertEqual(result["outcome"], "demo_case_recorded")
        self.assertFalse(result["retailer_or_courier_claim_submitted"])
        self.assertEqual(result["case"]["details"]["product_name"], "Laptop Pro")
        self.assertIn("cracked", result["case"]["reason"])
        self.assertEqual(report_damaged_order("ORD-1001", None, "session-damage")["outcome"], "missing_damage_details")

    def test_missing_order_investigation_reports_delay_but_not_lost(self):
        result = investigate_missing_order("ORD-1001", "session-missing", "Tracking has not updated in several days.")
        self.assertEqual(result["outcome"], "demo_case_recorded")
        self.assertEqual(result["order"]["delivery_progress"]["stage"], "delayed")
        self.assertFalse(result["parcel_declared_lost"])
        self.assertFalse(result["carrier_contacted"])
        self.assertEqual(investigate_missing_order("ORD-9999", "session-missing", "No delivery yet.")["outcome"], "not_found")

    def test_handoff_records_session_reason_summary_and_simulated_status(self):
        result = request_customer_handoff(
            "session-human", "Customer asks for a person", "User asked about a missing delivery."
        )
        self.assertEqual(result["outcome"], "simulated_handoff_ready")
        self.assertFalse(result["live_agent_contacted"])
        record = list_demo_records("customer_handoff", "session-human")[0]
        self.assertEqual(record["session_id"], "session-human")
        self.assertEqual(record["reason"], "Customer asks for a person")
        self.assertIn("missing delivery", record["summary"])
        self.assertEqual(record["status"], "simulated_handoff_ready")

    def test_store_sales_are_labeled_as_demonstration_only(self):
        result = get_store_information("Are there any sales?")
        self.assertEqual(result["outcome"], "found")
        self.assertIn("sales", result["sections"])
        self.assertIn("DEMO10", result["sections"]["sales"][0]["code"])
        self.assertIn("fictional", result["data_notice"])

    def test_issue_resolver_handles_clear_phrases_and_ambiguity(self):
        examples = {
            "Where is my refund?": "refund_status",
            "I want to return my order.": "return_order",
            "My product arrived damaged.": "damaged_order",
            "My order has not arrived.": "missing_order",
        }
        for query, issue in examples.items():
            with self.subTest(query=query):
                self.assertEqual(resolve_issue_type(query)["issue_type"], issue)
        ambiguous = resolve_issue_type("I want to return my order, and where is my refund?")
        self.assertEqual(ambiguous["outcome"], "clarification_required")
        self.assertEqual(len(ambiguous["candidates"]), 2)

    def invoke_graph(self, query, intent, session_id):
        graph = build_graph()
        with patch.object(nodes, "classify_and_verify", return_value=classifier_result(intent)):
            return invoke_graph(query, session_id, graph)

    def test_issue_graph_routes_refund_return_damage_and_missing(self):
        refund = self.invoke_graph("Where is my refund for ORD-1004?", "issue_support", "g-refund")
        self.assertEqual(refund["selected_route"], "issue_support:refund_status")
        self.assertEqual(refund["tool_result"]["outcome"], "no_refund_record")

        returned = self.invoke_graph(
            "I want to return order ORD-1001 because the size does not fit.",
            "issue_support", "g-return",
        )
        self.assertEqual(returned["tool_result"]["outcome"], "demo_request_prepared")
        self.assertIn("manual review", returned["response"])
        self.assertIsNotNone(returned["support_case_reference"])

        damaged = self.invoke_graph(
            "My product arrived damaged, screen cracked, order ORD-1001.",
            "issue_support", "g-damaged",
        )
        self.assertEqual(damaged["tool_result"]["outcome"], "demo_case_recorded")
        self.assertIn("no retailer or courier claim", damaged["response"])

        missing = self.invoke_graph(
            "My order has not arrived, order ORD-1001.",
            "issue_support", "g-missing",
        )
        self.assertEqual(missing["tool_result"]["outcome"], "demo_case_recorded")
        self.assertFalse(missing["tool_result"]["parcel_declared_lost"])

    def test_refund_intent_reports_only_each_fixture_status(self):
        for order_id, status in {
            "ORD-1001": "not initiated",
            "ORD-1002": "processing",
            "ORD-1003": "completed",
        }.items():
            with self.subTest(order_id=order_id):
                result = self.invoke_graph(
                    f"Where is my refund for {order_id}?", "issue_support", f"refund-status-{order_id}"
                )
                self.assertEqual(result["tool_result"]["refund"]["status"], status.replace(" ", "_"))
                self.assertIn(f"'{status}'", result["response"])
        missing = self.invoke_graph(
            "Where is my refund for ORD-1004?", "issue_support", "refund-status-no-record"
        )
        self.assertEqual(missing["tool_result"]["outcome"], "no_refund_record")
        self.assertIn("cannot confirm", missing["response"])

    def test_missing_issue_fields_ask_clarification_without_recording(self):
        no_id = self.invoke_graph("I want to return my order because it does not fit.", "issue_support", "g-no-id")
        self.assertEqual(no_id["tool_result"]["outcome"], "missing_order_id")
        self.assertEqual(no_id["workflow_step"], "awaiting_order_id")
        self.assertIn("provide your order ID", no_id["response"])
        no_reason = self.invoke_graph("I want to return my order ORD-1001.", "issue_support", "g-no-reason")
        self.assertEqual(no_reason["tool_result"]["outcome"], "missing_reason")
        self.assertEqual(no_reason["workflow_step"], "awaiting_return_reason")

    def test_ambiguous_issue_prompts_and_does_not_execute_issue_tools(self):
        with patch.object(nodes, "refund_status_tool") as refund_tool, patch.object(
            nodes, "prepare_return_tool"
        ) as return_tool:
            result = self.invoke_graph(
                "I want to return my order, and where is my refund?",
                "issue_support", "g-ambiguous",
            )
        self.assertEqual(result["selected_route"], "issue_support:clarification")
        refund_tool.assert_not_called()
        return_tool.assert_not_called()

    def test_customer_care_graph_creates_simulated_handoff_and_isolates_sessions(self):
        first = self.invoke_graph("Please connect me to a human about my order.", "customer_care_support", "g-human-a")
        second = self.invoke_graph("I need a representative.", "customer_care_support", "g-human-b")
        self.assertEqual(first["handoff_status"], "simulated_handoff_ready")
        self.assertFalse(first["tool_result"]["live_agent_contacted"])
        self.assertIn("No live agent has been contacted", first["response"])
        self.assertEqual(first["messages"][-1]["role"], "assistant")
        self.assertEqual(len(list_demo_records("customer_handoff", "g-human-a")), 1)
        self.assertEqual(len(list_demo_records("customer_handoff", "g-human-b")), 1)
        self.assertNotEqual(first["support_case_reference"], second["support_case_reference"])

    def test_store_graph_and_out_of_scope_no_tool_execution(self):
        store = self.invoke_graph("What promotions are available?", "store_information", "g-store")
        self.assertEqual(store["selected_route"], "store_information")
        self.assertIn("not live or redeemable", store["response"])

        with patch.object(nodes, "refund_status_tool") as refund_tool, patch.object(
            nodes, "customer_handoff_tool"
        ) as handoff_tool, patch.object(nodes, "store_information_tool") as store_tool, patch.object(
            nodes, "prepare_return_tool"
        ) as return_tool, patch.object(nodes, "damaged_order_tool") as damage_tool, patch.object(
            nodes, "missing_order_tool"
        ) as missing_tool, patch.object(nodes, "rag_node") as rag_node:
            graph = build_graph()
            with patch.object(nodes, "classify_and_verify", return_value=classifier_result(None, "OUT_OF_SCOPE")):
                result = invoke_graph("Tell me about a movie", "g-outscope", graph)
        self.assertEqual(result["selected_route"], "out_of_scope")
        self.assertIn("order status", result["response"])
        refund_tool.assert_not_called()
        handoff_tool.assert_not_called()
        store_tool.assert_not_called()
        return_tool.assert_not_called()
        damage_tool.assert_not_called()
        missing_tool.assert_not_called()
        rag_node.assert_not_called()


if __name__ == "__main__":
    unittest.main()


