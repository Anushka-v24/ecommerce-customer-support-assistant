"""Tests for masking personal and payment identifiers in free text."""

import unittest

from app.services.pii import mask_obj, mask_pii


class PhoneMaskingTests(unittest.TestCase):
    def test_indian_phone_formats(self):
        for text in ("+91 98765 43210", "9876543210", "098765-43210", "+91-9876543210", "+919876543210", "98765 43210"):
            with self.subTest(text=text):
                self.assertEqual(mask_pii(f"call me on {text} please"), "call me on [PHONE] please")

    def test_international_phone_formats(self):
        for text in ("+1 415 555 2671", "+44 20 7946 0958", "+971-50-123-4567"):
            with self.subTest(text=text):
                self.assertEqual(mask_pii(f"reach me at {text}."), "reach me at [PHONE].")


class EmailAndUpiTests(unittest.TestCase):
    def test_emails(self):
        self.assertEqual(mask_pii("mail anushka.v+shop@example.co.in now"), "mail [EMAIL] now")
        self.assertEqual(mask_pii("Contact: a_b@mail.com."), "Contact: [EMAIL].")

    def test_upi_ids(self):
        for text in ("anushka@okaxis", "98765@ybl", "rahul.k@paytm", "shop-01@oksbi"):
            with self.subTest(text=text):
                self.assertEqual(mask_pii(f"refund to {text} please"), "refund to [UPI] please")

    def test_upi_at_sentence_end(self):
        self.assertEqual(mask_pii("Send it to priya@okicici."), "Send it to [UPI].")

    def test_email_is_not_tagged_as_upi(self):
        self.assertEqual(mask_pii("priya@gmail.com"), "[EMAIL]")


class CardMaskingTests(unittest.TestCase):
    def test_luhn_valid_cards(self):
        for text in ("4111 1111 1111 1111", "4111-1111-1111-1111", "4111111111111111", "378282246310005", "5555 5555 5555 4444"):
            with self.subTest(text=text):
                self.assertEqual(mask_pii(f"my card {text} was charged"), "my card [CARD] was charged")

    def test_luhn_invalid_number_untouched(self):
        for text in ("1234 5678 9012 3456", "1234567890123456", "4111 1111 1111 1112"):
            with self.subTest(text=text):
                self.assertEqual(mask_pii(f"ref {text}"), f"ref {text}")


class IdentityDocumentTests(unittest.TestCase):
    def test_aadhaar(self):
        for text in ("2345 6789 0123", "2345-6789-0123", "234567890123"):
            with self.subTest(text=text):
                self.assertEqual(mask_pii(f"aadhaar {text} attached"), "aadhaar [AADHAAR] attached")

    def test_aadhaar_cannot_start_with_zero_or_one(self):
        self.assertEqual(mask_pii("id 1234 5678 9012"), "id 1234 5678 9012")

    def test_pan(self):
        self.assertEqual(mask_pii("PAN ABCDE1234F for invoice"), "PAN [PAN] for invoice")
        self.assertEqual(mask_pii("pan is abcde1234f"), "pan is [PAN]")


class NonPiiUntouchedTests(unittest.TestCase):
    def test_business_values_untouched(self):
        for text in (
            "Where is ORD-1001?",
            "I paid ₹15,000 for it",
            "refund 20000 rupees",
            "price ₹1,50,000.00",
            "delivered on 2025-11-08 or 08/11/2025",
            "deliver to pincode 560001",
            "Samsung Galaxy S24 Ultra 256GB",
            "iPhone 15 Pro Max and Sony WH-1000XM5",
            "order PROD-001 qty 2",
        ):
            with self.subTest(text=text):
                self.assertEqual(mask_pii(text), text)

    def test_empty_text(self):
        self.assertEqual(mask_pii(""), "")

    def test_mixed_message(self):
        text = "ORD-1001: refund ₹15,000 to 98765@ybl or card 4111 1111 1111 1111, call +91 98765 43210, mail a@b.com"
        self.assertEqual(
            mask_pii(text),
            "ORD-1001: refund ₹15,000 to [UPI] or card [CARD], call [PHONE], mail [EMAIL]",
        )


class MaskObjTests(unittest.TestCase):
    def test_nested_structures(self):
        payload = {
            "query": "call 9876543210",
            "order_id": "ORD-1001",
            "amount": 15000,
            "history": [{"text": "mail me at x@y.com"}, ("PAN ABCDE1234F", None)],
        }
        self.assertEqual(mask_obj(payload), {
            "query": "call [PHONE]",
            "order_id": "ORD-1001",
            "amount": 15000,
            "history": [{"text": "mail me at [EMAIL]"}, ("PAN [PAN]", None)],
        })

    def test_does_not_mutate_input(self):
        payload = {"query": "call 9876543210"}
        mask_obj(payload)
        self.assertEqual(payload, {"query": "call 9876543210"})


if __name__ == "__main__":
    unittest.main()
