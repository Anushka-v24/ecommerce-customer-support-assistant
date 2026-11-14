"""Real process-restart checkpoint test; opt in with PostgreSQL and an explicit flag."""

from __future__ import annotations

import os
import re
import subprocess
import sys
import unittest
from pathlib import Path
from uuid import uuid4

from app.config import Config


DATABASE_URL = (Config.DATABASE_URL or "").strip()
RUN_INTEGRATION = os.getenv("RUN_POSTGRES_CHECKPOINT_TESTS") == "1"


@unittest.skipUnless(
    DATABASE_URL and RUN_INTEGRATION,
    "Set DATABASE_URL and RUN_POSTGRES_CHECKPOINT_TESTS=1 to verify PostgreSQL checkpoint persistence across processes.",
)
class PostgreSQLCheckpointRestartTests(unittest.TestCase):
    def run_child(self, source: str, env: dict[str, str]) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [sys.executable, "-c", source],
            cwd=str(Path(__file__).resolve().parents[1]),
            env=env,
            capture_output=True,
            text=True,
            check=False,
            timeout=90,
        )

    @staticmethod
    def safe_failure(result: subprocess.CompletedProcess[str], label: str) -> str:
        # Driver errors can include a URI; redact credential-bearing authorities.
        details = re.sub(r"(?i)(postgres(?:ql)?(?:\+psycopg)?://)[^@\s]+@", r"\1<redacted>@", result.stderr)
        return f"{label} (exit {result.returncode}): {details[-2500:]}"

    def test_pending_workflow_resumes_after_new_process_and_threads_are_isolated(self):
        thread_id = f"checkpoint-restart-{uuid4().hex}"
        other_thread = f"checkpoint-isolation-{uuid4().hex}"
        env = os.environ.copy()
        env["DATABASE_URL"] = DATABASE_URL
        env["CHECKPOINT_BACKEND"] = "postgres"
        env["BUSINESS_DATA_BACKEND"] = "json"
        env["TEST_THREAD_ID"] = thread_id
        env["OTHER_THREAD_ID"] = other_thread

        first_process = r'''
import os
from app.config import Config
Config.CHECKPOINT_BACKEND = "postgres"
Config.BUSINESS_DATA_BACKEND = "json"
from app.graph import nodes
nodes.classify_and_verify = lambda query: {
    "predicted_intent": "order_tracking", "confidence_score": 99.0,
    "final_intent": "order_tracking", "verifier_status": "ACCEPTED_BY_CLASSIFIER",
    "routing": "DIRECT_CLASSIFIER_OUTPUT", "latency_ms": 1.0, "error": None,
}
from app.graph.workflow import close_graph_resources, invoke_graph
try:
    result = invoke_graph("Where is my order?", os.environ["TEST_THREAD_ID"])
    assert result.get("pending_workflow", {}).get("awaiting_slot") == "order_id", result
finally:
    close_graph_resources()
'''
        restarted_process = r'''
import os
from app.config import Config
Config.CHECKPOINT_BACKEND = "postgres"
Config.BUSINESS_DATA_BACKEND = "json"
from app.graph import nodes
nodes.classify_and_verify = lambda query: {
    "predicted_intent": "out_of_scope", "confidence_score": 99.0,
    "final_intent": "out_of_scope", "verifier_status": "ACCEPTED_BY_CLASSIFIER",
    "routing": "DIRECT_CLASSIFIER_OUTPUT", "latency_ms": 1.0, "error": None,
}
from app.graph.checkpointer import get_configured_checkpointer
from app.graph.workflow import close_graph_resources, invoke_graph
try:
    resumed = invoke_graph("ORD-1001", os.environ["TEST_THREAD_ID"])
    assert resumed.get("tool_result", {}).get("outcome") == "found", resumed.get("tool_result")
    assert resumed.get("order_id") == "ORD-1001", resumed.get("order_id")
    isolated = invoke_graph("ORD-1001", os.environ["OTHER_THREAD_ID"])
    assert isolated.get("selected_route") == "out_of_scope", isolated.get("selected_route")
finally:
    saver = get_configured_checkpointer()
    saver.delete_thread(os.environ["TEST_THREAD_ID"])
    saver.delete_thread(os.environ["OTHER_THREAD_ID"])
    close_graph_resources()
'''
        cleanup = r'''
import os
from app.config import Config
Config.CHECKPOINT_BACKEND = "postgres"
from app.graph.checkpointer import get_configured_checkpointer
from app.graph.workflow import close_graph_resources
try:
    saver = get_configured_checkpointer()
    saver.delete_thread(os.environ["TEST_THREAD_ID"])
    saver.delete_thread(os.environ["OTHER_THREAD_ID"])
finally:
    close_graph_resources()
'''
        try:
            first = self.run_child(first_process, env)
            self.assertEqual(first.returncode, 0, self.safe_failure(first, "Initial PostgreSQL checkpoint process failed"))
            restarted = self.run_child(restarted_process, env)
            self.assertEqual(restarted.returncode, 0, self.safe_failure(restarted, "Restart/resumption or session-isolation process failed"))
        finally:
            # Only unique thread IDs created by this test are removed.
            self.run_child(cleanup, env)


if __name__ == "__main__":
    unittest.main()
