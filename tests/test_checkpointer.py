"""Checkpoint selection and managed-pool tests without PostgreSQL."""

import unittest
from unittest.mock import MagicMock, patch

from langgraph.checkpoint.memory import InMemorySaver

from app.config import Config
from app.graph.checkpointer import (
    CheckpointerConfigurationError,
    CheckpointerInitializationError,
    close_postgres_checkpointer,
    get_configured_checkpointer,
)
from app.graph.workflow import build_graph, close_graph_resources

# Replies stay in English offline; translation has its own mocked tests.
patch.object(Config, "REPLY_TRANSLATION_ENABLED", False).start()


class CheckpointerConfigurationTests(unittest.TestCase):
    def tearDown(self):
        close_graph_resources()

    def test_memory_backend_remains_available_without_database_url(self):
        with patch.object(Config, "CHECKPOINT_BACKEND", "memory"), patch.object(Config, "DATABASE_URL", None):
            self.assertIsInstance(get_configured_checkpointer(), InMemorySaver)

    def test_invalid_backend_is_rejected(self):
        with patch.object(Config, "CHECKPOINT_BACKEND", "redis"):
            with self.assertRaisesRegex(CheckpointerConfigurationError, "memory.*postgres"):
                get_configured_checkpointer()

    def test_postgres_requires_database_url(self):
        with patch.object(Config, "CHECKPOINT_BACKEND", "postgres"), patch.object(Config, "DATABASE_URL", None):
            with self.assertRaisesRegex(CheckpointerConfigurationError, "requires DATABASE_URL"):
                get_configured_checkpointer()

    def test_postgres_rejects_invalid_or_non_postgres_urls(self):
        for url in ("not-a-url", "sqlite:///bad.db"):
            with self.subTest(url=url), patch.object(Config, "CHECKPOINT_BACKEND", "postgres"), \
                    patch.object(Config, "DATABASE_URL", url):
                with self.assertRaisesRegex(CheckpointerConfigurationError, "valid PostgreSQL DATABASE_URL"):
                    get_configured_checkpointer()

    def test_postgres_saver_uses_one_persistent_pool_and_initializes_once(self):
        pool = MagicMock()
        saver = MagicMock()
        with patch.object(Config, "CHECKPOINT_BACKEND", "postgres"), \
             patch.object(Config, "DATABASE_URL", "postgresql+psycopg://user:pass@localhost/ecom"), \
             patch("app.graph.checkpointer.ConnectionPool", return_value=pool) as pool_factory, \
             patch("app.graph.checkpointer.PostgresSaver", return_value=saver) as saver_factory:
            first = get_configured_checkpointer()
            second = get_configured_checkpointer()
            self.assertIs(first, saver)
            self.assertIs(second, saver)
            pool_factory.assert_called_once()
            saver_factory.assert_called_once()
            pool.open.assert_called_once_with(wait=True, timeout=Config.DB_CONNECT_TIMEOUT_SECONDS)
            saver.setup.assert_called_once_with()
            kwargs = pool_factory.call_args.kwargs["kwargs"]
            self.assertTrue(kwargs["autocommit"])
            self.assertEqual(kwargs["row_factory"].__name__, "dict_row")
        close_postgres_checkpointer()
        pool.close.assert_called_once()

    def test_initialization_failure_closes_pool_and_does_not_fall_back(self):
        pool = MagicMock()
        saver = MagicMock()
        saver.setup.side_effect = OSError("connection unavailable")
        with patch.object(Config, "CHECKPOINT_BACKEND", "postgres"), \
             patch.object(Config, "DATABASE_URL", "postgresql://user:pass@localhost/ecom"), \
             patch("app.graph.checkpointer.ConnectionPool", return_value=pool), \
             patch("app.graph.checkpointer.PostgresSaver", return_value=saver):
            with self.assertRaises(CheckpointerInitializationError):
                get_configured_checkpointer()
        pool.close.assert_called_once()

    def test_injected_checkpointer_is_preserved(self):
        injected = InMemorySaver()
        with patch.object(Config, "CHECKPOINT_BACKEND", "invalid"):
            graph = build_graph(checkpointer=injected)
        self.assertIs(graph.checkpointer, injected)


if __name__ == "__main__":
    unittest.main()
