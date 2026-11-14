"""Database foundation tests that do not require a running PostgreSQL server."""

import os
import subprocess
import sys
import unittest
from unittest.mock import patch

from app.config import Config
from app.db.session import (
    DatabaseNotConfiguredError,
    dispose_engines,
    get_engine,
    get_session_factory,
    normalize_database_url,
    session_scope,
)


class FakeSession:
    def __init__(self):
        self.rolled_back = False
        self.closed = False

    def rollback(self):
        self.rolled_back = True

    def close(self):
        self.closed = True


class DatabaseFoundationTests(unittest.TestCase):
    def tearDown(self):
        dispose_engines()

    def test_database_url_is_optional_until_database_access(self):
        with patch.object(Config, "DATABASE_URL", None):
            with self.assertRaises(DatabaseNotConfiguredError):
                get_engine()

    def test_database_url_normalizes_to_psycopg_and_rejects_sqlite(self):
        normalized = normalize_database_url("postgresql://user:pass@localhost/example")
        self.assertTrue(normalized.startswith("postgresql+psycopg://"))
        with self.assertRaisesRegex(ValueError, "SQLite is not supported"):
            normalize_database_url("sqlite:///local.db")

    def test_config_reads_database_url_from_environment(self):
        child_env = os.environ.copy()
        child_env["DATABASE_URL"] = "postgresql+psycopg://dbuser:dbpassword@localhost/ecom_test"
        command = (
            "import dotenv; dotenv.load_dotenv=lambda *a, **k: False; "
            "from app.config import Config; "
            "assert Config.DATABASE_URL.startswith('postgresql+psycopg://')"
        )
        result = subprocess.run(
            [sys.executable, "-c", command],
            cwd=os.getcwd(), env=child_env, capture_output=True, text=True, check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_engine_and_session_factory_are_created_without_connecting(self):
        engine = get_engine("postgresql+psycopg://user:pass@127.0.0.1:1/example")
        self.assertEqual(engine.dialect.name, "postgresql")
        self.assertTrue(engine.pool._pre_ping)
        self.assertIs(get_engine("postgresql+psycopg://user:pass@127.0.0.1:1/example"), engine)
        factory = get_session_factory(engine)
        self.assertIs(get_session_factory(engine), factory)
        with session_scope(engine) as session:
            self.assertIsNotNone(session)

    def test_session_scope_rolls_back_on_error_and_always_closes(self):
        fake = FakeSession()
        with patch("app.db.session.get_session_factory", return_value=lambda: fake):
            with self.assertRaisesRegex(RuntimeError, "test failure"):
                with session_scope():
                    raise RuntimeError("test failure")
        self.assertTrue(fake.rolled_back)
        self.assertTrue(fake.closed)

    def test_session_scope_closes_on_success_without_implicit_commit(self):
        fake = FakeSession()
        with patch("app.db.session.get_session_factory", return_value=lambda: fake):
            with session_scope() as session:
                self.assertIs(session, fake)
        self.assertFalse(fake.rolled_back)
        self.assertTrue(fake.closed)

    def test_application_import_and_health_factory_do_not_require_database_url(self):
        child_env = os.environ.copy()
        child_env.pop("DATABASE_URL", None)
        command = (
            "import os; os.environ.pop('DATABASE_URL', None); "
            "import dotenv; dotenv.load_dotenv=lambda *a, **k: False; "
            "import app; from app.config import Config; "
            "assert Config.DATABASE_URL is None; app.create_app()"
        )
        result = subprocess.run(
            [sys.executable, "-c", command],
            cwd=os.getcwd(), env=child_env, capture_output=True, text=True, check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
