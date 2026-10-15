"""
Unit tests for multilingual normalizer and cross-lingual negation detection.
"""

from trustworthy_rag.data.multilingual import MultilingualNormalizer


def test_multilingual_normalization():
    # Spanish with accents
    text_es = "La investigación científica confirmó la hipótesis."
    stripped = MultilingualNormalizer.strip_accents_and_canonicalize(text_es)
    assert "investigacion" in stripped
    assert "cientifica" in stripped
    assert "hipotesis" in stripped

    tokens = MultilingualNormalizer.tokenize_multilingual(text_es)
    assert "la" not in tokens  # Stopword removed
    assert "investigacion" in tokens


def test_cross_lingual_negation():
    assert MultilingualNormalizer.detect_cross_lingual_negation("The statement was completely refutado in 2024.")
    assert MultilingualNormalizer.detect_cross_lingual_negation("Das ist absolut nicht wahr.")
    assert MultilingualNormalizer.detect_cross_lingual_negation("Ce n'est pas correct.")
    assert not MultilingualNormalizer.detect_cross_lingual_negation("The hypothesis was confirmed by the board.")
