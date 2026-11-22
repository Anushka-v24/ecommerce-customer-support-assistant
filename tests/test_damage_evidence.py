import io
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from app import create_app
from app.config import Config
from app.services import damage_evidence
from app.services.damage_evidence import EvidenceError, analyze_damage_photo
from tests.test_chat_api import ApiTestConfig

PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 64
GOOD = {"is_product_photo": True, "damage_visible": True, "damage_type": "cracked", "severity": "moderate",
        "description": "The phone screen has a diagonal crack across the top half.", "confidence": 0.86}


def groq_returning(content):
    client = MagicMock()
    client.chat.completions.create.return_value = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=content))])
    return SimpleNamespace(Groq=MagicMock(return_value=client)), client


class DamageEvidenceTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.dir = Path(tmp.name)
        for patcher in (patch.object(Config, "TURN_LOG_DIR", tmp.name), patch.dict("os.environ", {"GROQ_API_KEY": "test-key"})):
            patcher.start()
            self.addCleanup(patcher.stop)

    def analyze(self, content, data=PNG, mime="image/png"):
        module, client = groq_returning(content)
        with patch.dict("sys.modules", {"groq": module}):
            return analyze_damage_photo(data, mime, "s1", "ORD-1003"), client

    def test_damage_is_described_and_photo_kept(self):
        result, client = self.analyze(json.dumps(GOOD))
        self.assertTrue(result["evidence_id"].startswith("EV-"))
        self.assertIn("diagonal crack", result["summary"])
        self.assertIn("moderate", result["summary"])
        self.assertTrue((self.dir / "evidence" / f"{result['evidence_id']}.png").exists())
        sent = client.chat.completions.create.call_args.kwargs
        self.assertEqual(sent["model"], Config.VISION_MODEL)
        self.assertTrue(sent["messages"][1]["content"][1]["image_url"]["url"].startswith("data:image/png;base64,"))

    def test_schema_violation_falls_back_to_human_review(self):
        for bad in ['{"damage": true}', json.dumps({**GOOD, "severity": "catastrophic"}), json.dumps({**GOOD, "confidence": 3}), "not json"]:
            with self.subTest(bad=bad):
                result, _ = self.analyze(bad)
                self.assertIsNone(result["analysis"])
                self.assertIn("support agent will review", result["summary"])

    def test_non_product_photo_is_flagged(self):
        result, _ = self.analyze(json.dumps({**GOOD, "is_product_photo": False, "damage_visible": False,
                                             "damage_type": "none", "severity": "none"}))
        self.assertIn("doesn't appear to show a product", result["summary"])

    def test_bad_uploads_are_rejected_before_any_call(self):
        cases = [(PNG, "application/pdf"), (b"", "image/png"), (b"GIF89a....", "image/png"),
                 (b"\x89PNG\r\n\x1a\n" + b"\x00" * (Config.EVIDENCE_MAX_BYTES + 1), "image/png")]
        for data, mime in cases:
            with self.subTest(mime=mime, size=len(data)):
                with self.assertRaises(EvidenceError):
                    analyze_damage_photo(data, mime, "s1")

    def test_never_triggers_a_refund(self):
        result, _ = self.analyze(json.dumps(GOOD))
        self.assertNotIn("refund", json.dumps(result).lower())


class EvidenceEndpointTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        patcher = patch.object(Config, "TURN_LOG_DIR", tmp.name)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.client = create_app(ApiTestConfig).test_client()

    def test_upload_larger_than_chat_limit_is_accepted(self):
        photo = b"\x89PNG\r\n\x1a\n" + b"\x00" * (200 * 1024)
        with patch.object(damage_evidence, "_describe", return_value=damage_evidence._validate(GOOD)), \
                patch.dict("os.environ", {"GROQ_API_KEY": "k"}):
            response = self.client.post("/api/evidence", data={"session_id": "s1", "photo": (io.BytesIO(photo), "d.png", "image/png")},
                                        content_type="multipart/form-data")
        self.assertEqual(response.status_code, 201)
        self.assertIn("crack", response.get_json()["summary"])

    def test_wrong_type_is_a_400(self):
        response = self.client.post("/api/evidence", data={"session_id": "s1", "photo": (io.BytesIO(b"%PDF"), "x.pdf", "application/pdf")},
                                    content_type="multipart/form-data")
        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.get_json()["error"], "invalid_photo")


if __name__ == "__main__":
    unittest.main()
