import unittest
from unittest.mock import patch

from app.config import Config
from app.graph import nodes
from app.graph.routing import route_classifier_result
from app.rag import local_search
from app.rag.service import RAGConfigurationError


class LocalSearchTests(unittest.TestCase):
    def setUp(self):
        patcher = patch.object(Config, "APP_ENV", "development")
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_answers_are_quoted_from_the_cited_section(self):
        result = local_search.answer("How do I reset my password?")
        self.assertEqual(result["sources"][0]["document_id"], "demo-help-account-en-v1")
        section = next(h for h in local_search.search("How do I reset my password?") if h["heading"] == result["sources"][0]["section"])
        self.assertIn(section["body"][:80], result["answer"])
        self.assertIn("demo help documents", result["answer"])

    def test_unrelated_questions_are_declined(self):
        for question in ("Who won the cricket world cup?", "tell me a joke", "What is the weather in Mumbai?"):
            with self.subTest(question=question):
                self.assertIsNone(local_search.answer(question))

    def test_production_never_reads_demo_documents(self):
        with patch.object(Config, "APP_ENV", "production"), patch.object(Config, "RAG_USE_DEMO_KNOWLEDGE", False):
            self.assertEqual(local_search.search("How do I reset my password?"), [])


class HowToRoutingTests(unittest.TestCase):
    def state(self, intent, query):
        return {"verifier_status": "ACCEPTED_BY_CLASSIFIER", "final_intent": intent, "query": query}

    def test_how_to_questions_for_actions_go_to_documents(self):
        self.assertEqual(route_classifier_result(self.state("place_order", "how do I place an order?")), "store_information")
        self.assertEqual(route_classifier_result(self.state("account_management", "password kaise badlu")), "store_information")
        self.assertEqual(route_classifier_result(self.state("submit_feedback", "how do I leave a review")), "store_information")

    def test_requests_that_are_not_questions_keep_their_route(self):
        self.assertEqual(route_classifier_result(self.state("place_order", "order a Laptop Pro for me")), "unsupported")
        self.assertEqual(route_classifier_result(self.state("submit_feedback", "delivery was great")), "feedback")

    def test_offline_answer_used_when_rag_keys_missing(self):
        with patch.object(nodes, "get_rag_service", side_effect=RAGConfigurationError("OPENAI_API_KEY is not configured")), \
                patch.object(Config, "APP_ENV", "development"):
            result = nodes.store_information_node({"query": "how do I place an order", "final_intent": "place_order", "messages": []})
        self.assertEqual(result["workflow_step"], "rag_offline_answered")
        self.assertEqual(result["sources"][0]["document_id"], "demo-help-placing-order-en-v1")


if __name__ == "__main__":
    unittest.main()
