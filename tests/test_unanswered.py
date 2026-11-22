"""Questions the chat can't answer get a polite, specific reply that never mentions documents."""

import unittest
from unittest.mock import patch
from uuid import uuid4

from langgraph.checkpoint.memory import InMemorySaver

from app.config import Config
from app.graph import nodes
from app.graph.workflow import build_graph, invoke_graph
from app.rag.service import UNAVAILABLE_ANSWER
from app.services.unanswered import (
    FOLLOW_UP_ANSWER, FOLLOW_UP_PROMPT, FOLLOW_UP_REPLIES, LOOKUP_TROUBLE_REPLY, NO_ANSWER_REPLY, OFFERS_REPLY,
    follow_up_choice, is_offer_question, unanswered_reply,
)

patch.object(Config, "BUSINESS_DATA_BACKEND", "json").start()
patch.object(Config, "CHECKPOINT_BACKEND", "memory").start()
patch.object(Config, "LLM_EXTRACTION_ENABLED", False).start()
patch.object(Config, "REPLY_TRANSLATION_ENABLED", False).start()
patch.object(Config, "ESCALATION_ENABLED", False).start()
patch.object(Config, "MULTI_INTENT_ENABLED", False).start()


def classified(intent, status="ACCEPTED_BY_CLASSIFIER"):
    return {
        "predicted_intent": intent, "confidence_score": 98.0, "final_intent": intent,
        "verifier_status": status, "routing": "DIRECT_CLASSIFIER_OUTPUT", "latency_ms": 1.0, "error": None,
    }


class OfferDetectionTests(unittest.TestCase):
    def test_offer_questions(self):
        for text in ("when is the offer day?", "any discount today", "do you have a coupon", "big sale kab hai", "Black Friday deals?"):
            self.assertTrue(is_offer_question(text), text)

    def test_support_requests_that_mention_offers_are_not_offer_questions(self):
        for text in ("refund my coupon amount for ORD-1001", "return the item I bought on sale", "where is my order"):
            self.assertFalse(is_offer_question(text), text)

    def test_reply_kinds(self):
        self.assertEqual(unanswered_reply("when is the offer day?"), f"{OFFERS_REPLY}\n\n{FOLLOW_UP_PROMPT}")
        self.assertEqual(unanswered_reply("what colour is your logo"), f"{NO_ANSWER_REPLY}\n\n{FOLLOW_UP_PROMPT}")
        self.assertEqual(unanswered_reply("what colour is your logo", lookup_failed=True), f"{LOOKUP_TROUBLE_REPLY}\n\n{FOLLOW_UP_PROMPT}")
        for reply in (OFFERS_REPLY, NO_ANSWER_REPLY, LOOKUP_TROUBLE_REPLY, FOLLOW_UP_PROMPT):
            self.assertNotIn("document", reply.casefold())
        self.assertIn("cart page", OFFERS_REPLY)


class EndConversationDropsPendingRequestsTests(unittest.TestCase):
    def test_ending_the_conversation_does_not_bring_back_an_earlier_question(self):
        patcher = patch.object(Config, "TURN_LOG_ENABLED", False)
        patcher.start()
        self.addCleanup(patcher.stop)
        intents = patch.object(nodes, "classify_and_verify", side_effect=lambda q, history=None: classified("issue_support"))
        intents.start()
        self.addCleanup(intents.stop)
        graph = build_graph(InMemorySaver())
        session = f"end-{uuid4()}"
        asked = invoke_graph("my order has not arrived", session, graph=graph)
        self.assertIn("order", asked["response"].casefold())
        done = invoke_graph("end conversation", session, graph=graph)
        self.assertIn("Thank you for chatting with AgentiCart", done["response"])
        self.assertNotIn("To continue your earlier request", done["response"])


class FollowUpChoiceTests(unittest.TestCase):
    def test_buttons_and_typed_equivalents(self):
        for text in ("end conversation", "End the chat", "that's all", "no thanks", "I'm done"):
            self.assertEqual(follow_up_choice(text), "end", text)
        for text in ("thanks", "Thank you!", "thank you so much", "thx"):
            self.assertEqual(follow_up_choice(text), "thanks", text)
        for text in ("I have another question", "another doubt", "yes, I have one more question", "ask something else"):
            self.assertEqual(follow_up_choice(text), "another", text)

    def test_ordinary_messages_are_not_follow_up_choices(self):
        for text in ("where is my order", "I have a question about my refund", "thanks for the quick reply, but where is my order"):
            self.assertIsNone(follow_up_choice(text), text)


class UnansweredFlowTests(unittest.TestCase):
    def setUp(self):
        patcher = patch.object(Config, "TURN_LOG_ENABLED", False)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.graph = build_graph(InMemorySaver())
        self.session = f"unanswered-{uuid4()}"

    def classify_as(self, intent, status="ACCEPTED_BY_CLASSIFIER"):
        patcher = patch.object(nodes, "classify_and_verify", side_effect=lambda q, history=None: classified(intent, status))
        patcher.start()
        self.addCleanup(patcher.stop)

    def say(self, text):
        return invoke_graph(text, self.session, graph=self.graph)

    def test_offer_question_points_to_the_cart_page_without_searching(self):
        self.classify_as("store_information")
        with patch.object(nodes, "get_rag_service"), patch.object(nodes, "run_agentic_rag", side_effect=AssertionError("must not search")):
            result = self.say("when is the offer day?")
        self.assertIn(OFFERS_REPLY, result["response"])
        self.assertIn(FOLLOW_UP_PROMPT, result["response"])
        self.assertEqual(result["quick_replies"], FOLLOW_UP_REPLIES)
        self.assertEqual(result["workflow_step"], "offers_redirect")

    def test_offer_question_misclassified_as_out_of_scope_still_gets_the_cart_page_reply(self):
        self.classify_as("out_of_scope", "OUT_OF_SCOPE")
        self.assertIn(OFFERS_REPLY, self.say("any big sale coming up?")["response"])

    def test_a_plain_knowledge_miss_is_rephrased_politely(self):
        self.classify_as("store_information")
        miss = {"rag_status": "no_evidence", "rag_answer": UNAVAILABLE_ANSWER, "sources": [], "rag_documents": []}
        with patch.object(nodes, "get_rag_service"), patch.object(nodes, "run_agentic_rag", return_value=miss):
            result = self.say("what colour is your logo")
        self.assertIn(NO_ANSWER_REPLY, result["response"])
        self.assertIn(FOLLOW_UP_PROMPT, result["response"])
        self.assertNotIn("knowledge documents", result["response"])
        # Two buttons only, in place of the usual help menu.
        self.assertEqual([item["label"] for item in result["quick_replies"]], ["Ask another question", "End conversation"])

    def test_ask_another_question_invites_it_without_a_menu_or_classifier_call(self):
        with patch.object(nodes, "classify_and_verify", side_effect=AssertionError("must not classify")):
            result = self.say("I have another question")
        self.assertEqual(result["response"], FOLLOW_UP_ANSWER)
        self.assertFalse(result.get("quick_replies"))

    def test_thanks_gets_a_welcome_without_a_classifier_call(self):
        with patch.object(nodes, "classify_and_verify", side_effect=AssertionError("must not classify")):
            result = self.say("thanks")
        self.assertEqual(result["selected_route"], "goodbye")
        self.assertIn("You're welcome", result["response"])
        self.assertFalse(result.get("quick_replies"))

    def test_end_conversation_says_goodbye_without_a_classifier_call(self):
        with patch.object(nodes, "classify_and_verify", side_effect=AssertionError("must not classify")):
            result = self.say("end conversation")
        self.assertEqual(result["selected_route"], "goodbye")
        self.assertIn("Thank you for chatting with AgentiCart", result["response"])
        self.assertFalse(result.get("quick_replies"))


if __name__ == "__main__":
    unittest.main()
