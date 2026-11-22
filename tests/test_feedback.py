import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from uuid import uuid4

from app.config import Config
from app.graph import nodes
from app.graph.workflow import build_graph, invoke_graph
from app.services.feedback_service import has_feedback_content
from tests.test_graph import classifier_result

# Replies stay in English offline; translation has its own mocked tests.
patch.object(Config, "REPLY_TRANSLATION_ENABLED", False).start()
patch.object(Config, "ESCALATION_ENABLED", False).start()


class FeedbackContentTests(unittest.TestCase):
    def test_announcements_have_no_content(self):
        for text in ["I want to give feedback", "feedback", "feedback dena hai", "मुझे फ़ीडबैक देना है", "how do I leave a review?"]:
            with self.subTest(text=text):
                self.assertFalse(has_feedback_content(text))

    def test_opinions_have_content(self):
        for text in ["the product is good", "delivery was super fast, loved it", "app bahut slow hai, improve karo", "पैकिंग बहुत अच्छी थी"]:
            with self.subTest(text=text):
                self.assertTrue(has_feedback_content(text))


class FeedbackWorkflowTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        for name, value in {"TURN_LOG_ENABLED": True, "TURN_LOG_DIR": self.tmp.name, "INTENT_CONFIRMATION_ENABLED": False}.items():
            patcher = patch.object(Config, name, value)
            patcher.start()
            self.addCleanup(patcher.stop)
        self.graph = build_graph()

    def turn(self, query, session_id, intent="submit_feedback"):
        with patch.object(nodes, "classify_and_verify", return_value=classifier_result(intent, intent)):
            return invoke_graph(query, session_id, self.graph)

    def stored(self):
        path = Path(self.tmp.name) / "customer_feedback.jsonl"
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()] if path.exists() else []

    def test_opinion_is_recorded_with_pii_masked(self):
        result = self.turn("great packaging, call me on 9876543210 if you need more", str(uuid4()))
        self.assertEqual(result["selected_route"], "feedback")
        self.assertEqual(result["tool_result"]["status"], "recorded")
        self.assertIn(result["tool_result"]["reference"], result["response"])
        records = self.stored()
        self.assertEqual(len(records), 1)
        self.assertIn("[PHONE]", records[0]["text"])
        self.assertNotIn("9876543210", records[0]["text"])

    def test_announcement_asks_then_records_next_message(self):
        session = str(uuid4())
        first = self.turn("I want to give feedback", session)
        self.assertEqual(first["pending_workflow"]["awaiting_slot"], "feedback_text")
        self.assertEqual(self.stored(), [])
        second = self.turn("the delivery boy was very polite", session, intent="greeting")
        self.assertEqual(second["selected_route"], "feedback")
        self.assertEqual(self.stored()[0]["text"], "the delivery boy was very polite")
        self.assertIsNone(second.get("pending_workflow"))

    def test_hindi_feedback_gets_hindi_reply(self):
        result = self.turn("पैकिंग बहुत अच्छी थी", str(uuid4()))
        self.assertIn("धन्यवाद", result["response"])


if __name__ == "__main__":
    unittest.main()
