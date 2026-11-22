import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from app.config import Config
from app.rag.local_providers import LocalVectorIndex
from app.rag.service import RAGConfigurationError, RAGService

NS = "ecom-support-kb-demo-v1"


def vec(*values):
    return list(values)


class LocalVectorIndexTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.index = LocalVectorIndex(Path(tmp.name) / "vectors.json")

    def fill(self):
        self.index.upsert(namespace=NS, vectors=[
            {"id": "eskb-a", "values": vec(1, 0, 0), "metadata": {"text": "a"}},
            {"id": "eskb-b", "values": vec(0, 1, 0), "metadata": {"text": "b"}},
            {"id": "eskb-c", "values": vec(0.9, 0.1, 0), "metadata": {"text": "c"}},
        ])

    def test_query_ranks_by_cosine_similarity(self):
        self.fill()
        matches = self.index.query(vector=vec(1, 0, 0), top_k=2, namespace=NS)["matches"]
        self.assertEqual([m["id"] for m in matches], ["eskb-a", "eskb-c"])
        self.assertAlmostEqual(matches[0]["score"], 1.0, places=5)
        self.assertEqual(matches[0]["metadata"], {"text": "a"})

    def test_list_filters_by_prefix_and_delete_removes(self):
        self.fill()
        listed = {v.id for page in self.index.list(prefix="eskb-", namespace=NS) for v in page.vectors}
        self.assertEqual(listed, {"eskb-a", "eskb-b", "eskb-c"})
        self.index.delete(ids=["eskb-b"], namespace=NS)
        listed = {v.id for page in self.index.list(prefix="eskb-", namespace=NS) for v in page.vectors}
        self.assertEqual(listed, {"eskb-a", "eskb-c"})

    def test_namespaces_are_separate(self):
        self.fill()
        self.assertEqual(list(self.index.list(prefix="", namespace="other"))[0].vectors, [])

    def test_empty_index_points_to_the_ingest_command(self):
        with self.assertRaises(RAGConfigurationError) as raised:
            self.index.query(vector=vec(1, 0, 0), top_k=1, namespace=NS)
        self.assertIn("app.rag.ingest --demo", str(raised.exception))


class FreeStackServiceTests(unittest.TestCase):
    def setUp(self):
        for name, value in (("RAG_EMBEDDING_PROVIDER", "local"), ("RAG_VECTOR_STORE", "local"),
                            ("RAG_GENERATION_PROVIDER", "groq"), ("RAG_MIN_SCORE", 0.45), ("APP_ENV", "development")):
            patcher = patch.object(Config, name, value)
            patcher.start()
            self.addCleanup(patcher.stop)
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.index = LocalVectorIndex(Path(tmp.name) / "vectors.json")
        self.index.upsert(namespace=NS, vectors=[
            {"id": "eskb-1", "values": vec(1, 0), "metadata": {"text": "Reset it from the sign-in page.", "source": "help/account.md", "document_id": "d1"}},
            {"id": "eskb-2", "values": vec(0.5, 0.87), "metadata": {"text": "Unrelated.", "source": "help/other.md", "document_id": "d2"}},
        ])
        embedder = MagicMock()
        embedder.embeddings.create.return_value = SimpleNamespace(data=[SimpleNamespace(embedding=vec(1, 0))])
        self.service = RAGService(embedding_client=embedder, index=self.index, include_demo=True)

    def test_no_paid_settings_are_required(self):
        with patch.dict("os.environ", {"GROQ_API_KEY": "k"}):
            self.assertEqual(RAGService()._missing_settings(generation=True), [])

    def test_missing_groq_key_is_the_only_requirement(self):
        with patch.dict("os.environ", {"GROQ_API_KEY": ""}):
            self.assertEqual(RAGService()._missing_settings(generation=True), ["GROQ_API_KEY"])
            self.assertEqual(RAGService()._missing_settings(generation=False), [])

    def test_retrieve_applies_the_configured_min_score(self):
        # Against the query vector (1, 0): d1 scores 1.0 and d2 about 0.50.
        ids = lambda: [d["document_id"] for d in self.service.retrieve("how do I reset my password")]  # noqa: E731
        self.assertEqual(ids(), ["d1", "d2"])  # 0.45 keeps both
        with patch.object(Config, "RAG_MIN_SCORE", 0.6):
            self.assertEqual(ids(), ["d1"])
        with patch.object(Config, "RAG_MIN_SCORE", 1.1):
            self.assertEqual(ids(), [])

    def test_groq_client_is_used_with_retries_and_the_groq_model(self):
        groq = MagicMock()
        with patch.dict("os.environ", {"GROQ_API_KEY": "k"}), patch.dict("sys.modules", {"groq": SimpleNamespace(Groq=groq)}), \
                patch.object(Config, "RAG_GENERATION_MODEL", "openai/gpt-oss-120b"):
            service = RAGService()
            service._ensure_generation_client()
            self.assertEqual(groq.call_args.kwargs["max_retries"], 4)
            self.assertEqual(service._generation_model(), "openai/gpt-oss-120b")


if __name__ == "__main__":
    unittest.main()
