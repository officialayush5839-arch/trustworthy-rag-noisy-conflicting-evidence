"""
Multi-lingual text normalization and cross-lingual contradiction filtering.
Provides Unicode NFKC decomposition, diacritic stripping, multi-lingual stopword removal
(English, Spanish, French, German), and cross-lingual negation token mapping.
"""

from __future__ import annotations

import re
import unicodedata
from typing import List, Set


MULTILINGUAL_STOPWORDS: Set[str] = {
    # English
    "the", "a", "an", "is", "are", "was", "were", "in", "on", "at", "by", "for", "with",
    # Spanish
    "el", "la", "los", "las", "un", "una", "es", "son", "fue", "en", "de", "por", "para", "con",
    # French
    "le", "la", "les", "un", "une", "est", "sont", "ete", "dans", "sur", "par", "pour", "avec",
    # German
    "der", "die", "das", "ein", "eine", "ist", "sind", "war", "in", "auf", "mit", "von", "fur",
}

CROSS_LINGUAL_NEGATION_MARKERS: Set[str] = {
    "not", "never", "no", "neither", "nor",
    "no", "nunca", "jamas", "falso", "refutado",
    "pas", "jamais", "aucun", "non", "faux", "refute",
    "nicht", "nie", "niemals", "kein", "keine", "falsch", "widerlegt",
}


class MultilingualNormalizer:
    """Canonicalizes cross-lingual texts to facilitate cross-lingual evidence matching."""

    @staticmethod
    def strip_accents_and_canonicalize(text: str) -> str:
        """Decompose Unicode characters, strip diacritics, and normalize to NFKC."""
        nfkc_text = unicodedata.normalize("NFKC", text)
        decomposed = unicodedata.normalize("NFD", nfkc_text)
        stripped = "".join(ch for ch in decomposed if unicodedata.category(ch) != "Mn")
        return stripped.lower()

    @classmethod
    def tokenize_multilingual(cls, text: str) -> List[str]:
        """Extract alphanumeric tokens excluding multi-lingual stop words."""
        norm = cls.strip_accents_and_canonicalize(text)
        raw_tokens = re.findall(r"\b[a-z0-9_\-]{2,}\b", norm)
        return [t for t in raw_tokens if t not in MULTILINGUAL_STOPWORDS]

    @classmethod
    def detect_cross_lingual_negation(cls, text: str) -> bool:
        """Check if passage contains cross-lingual negation or refutation markers."""
        tokens = set(re.findall(r"\b[a-z]+\b", cls.strip_accents_and_canonicalize(text)))
        return bool(tokens.intersection(CROSS_LINGUAL_NEGATION_MARKERS))
