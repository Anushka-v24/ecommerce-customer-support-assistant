"""Tests for choosing the reply language from a customer message."""

import unittest

from app.services.language import HINGLISH_MARKERS, detect_reply_language, language_name


class HindiDetectionTests(unittest.TestCase):
    def test_devanagari_message(self):
        self.assertEqual(detect_reply_language("मेरा ऑर्डर कहाँ है?"), "hi")

    def test_devanagari_with_order_id(self):
        self.assertEqual(detect_reply_language("ORD-1001 का रिफंड कब मिलेगा?"), "hi")

    def test_mostly_english_with_one_hindi_word(self):
        self.assertEqual(detect_reply_language("Please check my order status today, धन्यवाद"), "en")


class HinglishDetectionTests(unittest.TestCase):
    def test_hinglish_sentences(self):
        for text in (
            "Mera order kab aayega?",
            "bhai refund abhi tak nahi aaya",
            "Mujhe return karna hai",
            "Order cancel kaise karu yaar, kuch samajh nahi aa raha",
            "KYA MERA PAYMENT HO GAYA?",
        ):
            with self.subTest(text=text):
                self.assertEqual(detect_reply_language(text), "hinglish")

    def test_short_text_needs_one_marker(self):
        self.assertEqual(detect_reply_language("order kab aayega"), "hinglish")
        self.assertEqual(detect_reply_language("refund kab?"), "hinglish")

    def test_long_text_needs_two_markers(self):
        self.assertEqual(detect_reply_language("I want to know when my parcel from Delhi will arrive, kal"), "en")
        self.assertEqual(detect_reply_language("I want to know when my parcel will arrive, kal tak milega?"), "hinglish")

    def test_markers_match_whole_words_only(self):
        self.assertEqual(detect_reply_language("The kaiser and the khaki chair have arrived safely"), "en")

    def test_marker_set_is_large_enough(self):
        self.assertGreaterEqual(len(HINGLISH_MARKERS), 40)
        self.assertTrue({"hai", "nahi", "kya", "mera", "mujhe", "kab", "kaise", "chahiye", "yaar"} <= HINGLISH_MARKERS)


class EnglishDetectionTests(unittest.TestCase):
    def test_english_sentences(self):
        for text in (
            "Hi, where is my order?",
            "hi",
            "Hi there, I want to return my headphones because they do not fit.",
            "Can you tell me the main reason my refund is delayed?",
            "Where is ORD-1001?",
            "Hello! Do you ship to Mumbai?",
        ):
            with self.subTest(text=text):
                self.assertEqual(detect_reply_language(text), "en")

    def test_empty_and_non_letter_text(self):
        for text in ("", "   ", "12345 ?!", "ORD-1001"):
            with self.subTest(text=text):
                self.assertEqual(detect_reply_language(text), "en")


class LanguageNameTests(unittest.TestCase):
    def test_known_codes(self):
        self.assertEqual(language_name("hi"), "Hindi")
        self.assertEqual(language_name("hinglish"), "Hinglish")
        self.assertEqual(language_name("en"), "English")

    def test_unknown_code_falls_back_to_english(self):
        self.assertEqual(language_name("fr"), "English")


if __name__ == "__main__":
    unittest.main()
