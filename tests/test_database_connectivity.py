"""Opt-in connectivity check for an explicitly configured PostgreSQL server."""

import unittest

from sqlalchemy import text

from app.config import Config
from app.db.session import dispose_engines, get_engine


DATABASE_URL = (Config.DATABASE_URL or "").strip()


@unittest.skipUnless(DATABASE_URL, "Set DATABASE_URL to run the PostgreSQL connectivity check.")
class PostgreSQLConnectivityTests(unittest.TestCase):
    def test_postgresql_accepts_a_simple_connection(self):
        engine = get_engine(DATABASE_URL)
        try:
            with engine.connect() as connection:
                self.assertEqual(connection.execute(text("SELECT 1")).scalar_one(), 1)
        finally:
            dispose_engines(engine)


if __name__ == "__main__":
    unittest.main()
