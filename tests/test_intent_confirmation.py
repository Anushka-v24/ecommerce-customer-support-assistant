"""Intent confirmation, the correction menu, and tickets raised at the end of a conversation."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from uuid import uuid4

from langgraph.checkpoint.memory import InMemorySaver

from app import create_app
from app.config import Config
from app.graph import nodes
from app.graph.workflow import build_graph, invoke_graph
from app.services import case_repository
from app.services.ticket_service import list_tickets


patch.object(Config, "BUSINESS_DATA_BACKEND", "json").start()
patch.object(Config, "CHECKPOINT_BACKEND", "memory").start()
patch.object(Config, "LLM_EXTRACTION_ENABLED", False).start()
patch.object(Config, "REPLY_TRANSLATION_ENABLED", False).start()
patch.object(Config, "ESCALATION_ENABLED", False).start()
patch.object(Config, "MULTI_INTENT_ENABLED", False).start()


def classified(intent):
    return {
        "predicted_intent": intent, "confidence_score": 98.0, "final_intent": intent,
        "verifier_status": "ACCEPTED_BY_CLASSIFIER", "routing": "DIRECT_CLASSIFIER_OUTPUT",
        "latency_ms": 1.0, "error": None,
    }


def fake_classifier(query, history=None):
    text = query.casefold()
    if "where is my order" in text:
        return classified("order_tracking")
    if "person" in text or "human" in text:
        return classified("customer_care_support")
    if "exchange" in text:
        return classified("exchange_request")
    return classified("issue_support")


class IntentConfirmationTests(unittest.TestCase):
    def setUp(self):
        # Other modules switch confirmation off globally; this module tests it on.
        for name, value in (("INTENT_CONFIRMATION_ENABLED", True), ("TURN_LOG_ENABLED", False)):
            patcher = patch.object(Config, name, value)
            patcher.start()
            self.addCleanup(patcher.stop)
        temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(temp_dir.cleanup)
        db_patch = patch.object(case_repository, "DEFAULT_DB_PATH", Path(temp_dir.name) / "support.sqlite3")
        db_patch.start()
        self.addCleanup(db_patch.stop)
        classifier = patch.object(nodes, "classify_and_verify", side_effect=fake_classifier)
        self.classifier = classifier.start()
        self.addCleanup(classifier.stop)
        self.graph = build_graph(InMemorySaver())
        self.session = str(uuid4())

    def say(self, message, customer_id=None):
        return invoke_graph(message, self.session, self.graph, customer_id=customer_id)

    def test_damage_report_is_confirmed_then_raises_a_ticket(self):
        asked = self.say("My headphones arrived broken, order ORD-1001", customer_id="CUST-001")
        self.assertEqual(asked["selected_route"], "confirm:issue_support")
        self.assertEqual(asked["workflow_status"], "awaiting_confirmation")
        self.assertIn("report a damaged item for order ORD-1001. Is that right?", asked["response"])
        self.assertEqual([item["value"] for item in asked["quick_replies"]], ["yes", "no"])
        self.assertIsNone(asked["ticket"])
        self.assertEqual(list_tickets(customer_id="CUST-001"), [])

        done = self.say("yes")
        self.assertEqual(done["tool_result"]["outcome"], "demo_case_recorded")
        ticket = done["ticket"]
        self.assertRegex(ticket["ticket_id"], r"^TCK-[0-9A-F]{8}$")
        self.assertEqual((ticket["category"], ticket["order_id"], ticket["status"], ticket["priority"]),
                         ("damaged_order", "ORD-1001", "open", "high"))
        self.assertIn(f"I've raised ticket {ticket['ticket_id']}", done["response"])
        # A raised ticket closes the request: Thanks and End conversation are offered.
        self.assertEqual([item["label"] for item in done["quick_replies"]], ["Thanks", "End conversation"])
        self.assertFalse(done["confirmation_required"])
        self.assertIsNone(done["pending_workflow"])
        stored = list_tickets(customer_id="CUST-001")
        self.assertEqual([item["ticket_id"] for item in stored], [ticket["ticket_id"]])
        self.assertEqual(stored[0]["session_id"], self.session)
        self.assertEqual(self.classifier.call_count, 1)

    def test_follow_up_questions_run_after_yes(self):
        self.say("my item arrived damaged")
        asked = self.say("yes")
        self.assertEqual(asked["tool_result"]["outcome"], "missing_order_id")
        self.say("ORD-1001")
        done = self.say("the screen is cracked across the middle")
        self.assertEqual(done["tool_result"]["outcome"], "demo_case_recorded")
        self.assertEqual(done["ticket"]["order_id"], "ORD-1001")
        self.assertIn("screen is cracked", list_tickets(session_id=self.session)[0]["description"])

    def test_no_offers_choices_and_the_chosen_workflow_asks_its_questions(self):
        self.say("my package came damaged, order ORD-1003")
        menu = self.say("no")
        self.assertEqual(menu["selected_route"], "pending:intent_correction")
        self.assertIn("Which of these is closest", menu["response"])
        self.assertIn("Talk to a person", [item["label"] for item in menu["quick_replies"]])
        self.assertFalse(menu["confirmation_required"])

        reason = self.say("return an item")
        self.assertEqual(reason["tool_result"]["outcome"], "missing_reason")
        self.assertEqual(reason["issue_details"]["order_id"], "ORD-1003")
        done = self.say("it does not fit me")
        self.assertEqual(done["tool_result"]["outcome"], "demo_request_prepared")
        self.assertEqual((done["ticket"]["category"], done["ticket"]["priority"]), ("return_order", "normal"))
        self.assertEqual(self.classifier.call_count, 1)

    def test_unrecognised_menu_answer_repeats_the_choices(self):
        self.say("my package came damaged")
        self.say("no")
        self.classifier.side_effect = lambda query, history=None: classified("issue_support")
        again = self.say("hmm")
        self.assertIn("Sorry, I didn't quite catch that", again["response"])
        self.assertEqual(len(again["quick_replies"]), 7)

    def test_something_else_asks_for_a_description_and_clears_the_menu(self):
        self.say("my package came damaged")
        self.say("no")
        result = self.say("something else")
        self.assertIn("in your own words", result["response"])
        self.assertIsNone(result["pending_workflow"])
        self.assertEqual(list_tickets(session_id=self.session), [])

    def test_customer_care_request_is_confirmed_and_ticketed(self):
        asked = self.say("I want to talk to a person", customer_id="CUST-002")
        self.assertEqual(asked["selected_route"], "confirm:customer_care")
        self.assertIn("Shall I raise a ticket", asked["response"])
        done = self.say("yes")
        self.assertEqual(done["handoff_status"], "simulated_handoff_ready")
        self.assertEqual(done["ticket"]["category"], "customer_care")
        self.assertEqual(len(list_tickets(customer_id="CUST-002")), 1)

    def test_exchange_is_confirmed_and_ticketed(self):
        asked = self.say("I want to exchange ORD-1003 for a bigger size", customer_id="CUST-002")
        self.assertEqual(asked["selected_route"], "confirm:exchange")
        self.assertIn("exchange an item for order ORD-1003. Is that right?", asked["response"])
        done = self.say("yes")
        self.assertEqual(done["tool_result"]["outcome"], "exchange_prepared")
        self.assertEqual((done["ticket"]["category"], done["ticket"]["case_reference"]),
                         ("exchange_request", done["tool_result"]["reference"]))
        self.assertIn("bigger size", done["ticket"]["description"])

    def test_menu_exchange_choice_runs_the_exchange_workflow(self):
        self.say("my package came damaged, order ORD-1003")
        self.say("no")
        done = self.say("exchange an item")
        self.assertEqual(done["tool_result"]["outcome"], "exchange_prepared")
        self.assertEqual(done["ticket"]["description"], "Exchange requested.")

    def test_refund_status_lookup_skips_confirmation(self):
        result = self.say("where is my refund for ORD-1002")
        self.assertEqual(result["selected_route"], "issue_support:refund_status")
        self.assertIsNone(result.get("ticket"))

    def test_ambiguous_issue_asks_which_issue_with_buttons(self):
        result = self.say("I have a problem with my order")
        self.assertEqual(result["selected_route"], "issue_support:clarification")
        self.assertEqual([item["label"] for item in result["quick_replies"]],
                         ["Refund status", "Return an item", "Damaged item", "Missing delivery"])

    def test_hindi_request_is_confirmed_in_hindi(self):
        with patch("app.graph.nodes.resolve_issue_type",
                   return_value={"outcome": "resolved", "issue_type": "missing_order", "order_id": "ORD-1001"}):
            asked = self.say("मेरा पार्सल अभी तक नहीं आया")
        self.assertIn("गुम डिलीवरी", asked["response"])
        self.assertIn("ORD-1001", asked["response"])
        self.assertEqual(asked["workflow_status"], "awaiting_confirmation")

    def test_new_request_during_confirmation_drops_the_confirmation(self):
        self.say("my item arrived damaged")
        tracked = self.say("where is my order ORD-1001")
        self.assertEqual(tracked["selected_route"], "order_tracking")
        self.assertNotIn("To continue your earlier request", tracked["response"])
        self.assertFalse(tracked["confirmation_required"])
        self.assertFalse(tracked.get("suspended_workflows"))

    def test_disabled_confirmation_routes_directly(self):
        with patch.object(Config, "INTENT_CONFIRMATION_ENABLED", False):
            result = self.say("my item arrived damaged, order ORD-1001")
        self.assertEqual(result["selected_route"], "issue_support:damaged_order")


class TicketApiTests(unittest.TestCase):
    class ApiConfig(Config):
        TESTING = True
        APP_ENV = "test"
        CORS_ALLOWED_ORIGINS = "https://shop.example"
        TURN_LOG_ENABLED = False

    def setUp(self):
        self.client = create_app(self.ApiConfig).test_client()
        temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(temp_dir.cleanup)
        db_patch = patch.object(case_repository, "DEFAULT_DB_PATH", Path(temp_dir.name) / "support.sqlite3")
        db_patch.start()
        self.addCleanup(db_patch.stop)

    def test_chat_returns_buttons_and_ticket_and_passes_customer(self):
        graph_result = {
            "response": "Done.", "final_intent": "issue_support", "workflow_status": "completed",
            "quick_replies": [{"label": "Yes", "value": "yes"}, {"bad": "dropped"}],
            "ticket": {"ticket_id": "TCK-1234ABCD", "category": "damaged_order", "session_id": "private",
                       "customer_id": "CUST-001", "status": "open"},
        }
        with patch("app.api.routes.invoke_graph", return_value=graph_result) as invoke:
            response = self.client.post("/api/chat", json={
                "session_id": "s-1", "message": "yes", "customer_id": "CUST-001",
            })
        body = response.get_json()
        self.assertEqual(response.status_code, 200)
        self.assertEqual(body["quick_replies"], [{"label": "Yes", "value": "yes"}])
        self.assertEqual(body["ticket"]["ticket_id"], "TCK-1234ABCD")
        self.assertNotIn("session_id", body["ticket"])
        self.assertNotIn("customer_id", body["ticket"])
        invoke.assert_called_once_with("yes", "s-1", customer_id="CUST-001")

    def test_unknown_customer_is_rejected(self):
        response = self.client.post("/api/chat", json={"session_id": "s-1", "message": "hi", "customer_id": "CUST-999"})
        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.get_json()["error"], "invalid_customer_id")

    def test_customers_and_ticket_listing(self):
        customers = self.client.get("/api/customers").get_json()["customers"]
        self.assertEqual([item["customer_id"] for item in customers[:2]], ["CUST-001", "CUST-002"])
        self.assertGreaterEqual(len(customers), 100)

        from app.services.ticket_service import create_ticket
        create_ticket(session_id="s-1", category="missing_order", description="Not delivered",
                      customer_id="CUST-001", order_id="ORD-1001")
        create_ticket(session_id="s-2", category="return_order", description="Wrong size", customer_id="CUST-002")

        listed = self.client.get("/api/tickets?customer_id=CUST-001").get_json()["tickets"]
        self.assertEqual([item["category"] for item in listed], ["missing_order"])
        self.assertNotIn("customer_id", listed[0])
        self.assertEqual(self.client.get("/api/tickets").status_code, 400)
        self.assertEqual(self.client.get("/api/tickets?customer_id=CUST-999").status_code, 400)


if __name__ == "__main__":
    unittest.main()
