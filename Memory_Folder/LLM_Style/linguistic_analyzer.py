# -*- coding: utf-8 -*-
"""
linguistic_analyzer.py — File 1 of 4 (Advanced)

Purpose
-------
Robust feature extractor for cloning a user's communication pattern.
Returns rich, structured features covering lexical habits, structure,
punctuation, casing, emojis, elongations, fillers/hedges/slang, repetition,
keysmash, onomatopoeia, and "negative" signals (rare/absent features).

Design Goals
------------
- **User-only analysis**: Analyze a single speaker's text per call.
- **Structured output**: Normalized ratios + raw counts + examples.
- **Placement awareness**: Emoji & punctuation placement (start/middle/end, sentence boundaries).
- **Extensible**: Custom dictionaries (fillers, slang, etc.) via constructor.
- **Back-compat wrappers**: `extract_patterns_from_text` & `extract_phrase_counts_from_text`.

Public API
----------
- class LinguisticAnalyzer:
    - analyze(text: str) -> FeatureReport (dict-like dataclass)
    - extract_phrase_counts(text: str) -> Dict[str, int]
    - merge_reports(reports: Iterable[FeatureReport]) -> FeatureReport

- module-level compatibility helpers:
    - extract_patterns_from_text(text: str) -> Dict[str, Dict[str, List[str]]]
    - extract_phrase_counts_from_text(analyzer: "LinguisticAnalyzer", text: str) -> Dict[str, int]

Notes
-----
No heavy external dependencies. Emoji detection uses a broad Unicode regex.
Sentence splitting and tokenization are heuristic (regex-based) but resilient.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Dict, List, Tuple, Iterable, Any, Optional
import math
import re
import unicodedata
import emoji

__all__ = [
    "LinguisticAnalyzer",
    "extract_patterns_from_text",
    "extract_phrase_counts_from_text",
]
# -------------------------------------------------------------
# Canonicalization helpers
# -------------------------------------------------------------
def remove_diacritics(text: str) -> str:
    if not text:
        return text
    norm = unicodedata.normalize("NFD", text)
    return "".join(ch for ch in norm if unicodedata.category(ch) != 'Mn')


def collapse_elongations_token(token: str) -> str:
    # Collapse runs of the same alpha character to a single instance
    return re.sub(r"([A-Za-z])\1{1,}", r"\1", token)


def normalize_token(token: str) -> str:
    t = token.strip()
    t = remove_diacritics(t)
    t = t.lower()
    t = collapse_elongations_token(t)
    # Strip common punctuation tails/heads
    t = t.strip(".,!?;:\"'()[]{}")
    return t



# -------------------------------------------------------------
# Regexes (compiled once)
# -------------------------------------------------------------
# Basic sentence boundary (keeps ellipsis & unicode ellipsis)
SENT_SEP_RE = re.compile(r"(?<=[.!?…])\s+")
# Token: split on whitespace, keep inline punctuation tokens with simple split
WHITESPACE_RE = re.compile(r"\s+")
# Word characters (letters + marks + digits + underscore)
WORD_RE = re.compile(r"\w+", re.UNICODE)
# Elongation: any alpha char repeated >=3 times consecutively (e.g., sooooo)
ELONGATION_RE = re.compile(r"([A-Za-z])\1{2,}")
# Repeated punctuation sequences like !!!, ???, !!!?, ??!
REPEAT_PUNCT_RE = re.compile(r"([!?.,~\-])\1{1,}")
# Ellipsis forms: ... or …
ELLIPSIS_RE = re.compile(r"(\.{3}|…)")
# Mixed/studly case (LoL, sPoNgEbOb, iPhone-like)
MIXED_CASE_RE = re.compile(r"(?=.*[A-Z])(?=.*[a-z])[A-Za-z]{3,}")
# All-caps token of length >= 2 alpha
ALL_CAPS_RE = re.compile(r"^[A-Z]{2,}$")
# Lowercase pronoun i as a token
LOWER_I_RE = re.compile(r"(?<![A-Za-z])i(?![A-Za-z])")
# Adjacent repeated words (case-insensitive)
REPEAT_WORD_RE = re.compile(r"\b(\w+)\b(?:\s+\1\b)+", re.IGNORECASE)
# Simple URL/handle patterns to optionally ignore
URL_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
HANDLE_RE = re.compile(r"[@#][\w_]+", re.UNICODE)

# Emoji ranges (approximate but broad). Covers emoticons, pictographs, flags, dingbats, etc.
def extract_emojis(text: str) -> list[str]:
    """Return all emojis in a string using the emoji library."""
    return [ch for ch in text if ch in emoji.EMOJI_DATA]

# Skin tone modifiers
SKIN_TONE_RE = re.compile(r"[\U0001F3FB-\U0001F3FF]")
# Variation selectors which often appear with emojis
VARIATION_SELECTOR_RE = re.compile(r"[\uFE0E\uFE0F]")

# Oxford comma patterns: X, Y, and Z  vs  X, Y and Z
OXFORD_COMMA_WITH_RE = re.compile(r"\b\w+\b,\s+\b\w+\b,\s+and\s+\b\w+\b", re.IGNORECASE)
OXFORD_COMMA_WITHOUT_RE = re.compile(r"\b\w+\b,\s+\b\w+\b\s+and\s+\b\w+\b", re.IGNORECASE)

# Keysmash heuristics: long runs of home-row sets or high consonant/noise ratio
KEYSMASH_RE = re.compile(r"(?i)\b(?:[asdfghjkl]{5,}|[qwertyuiop]{5,}|[zxcvbnm]{5,})\b")

# -------------------------------------------------------------
# Default dictionaries (can be overridden in constructor)
# -------------------------------------------------------------
DEFAULT_FILLERS = {
    "uh", "uhh", "uhhh", "um", "umm", "ummm", "er", "erm", "eh", "hmm", "mmm",
    "like", "so", "well", "you know", "i mean", "okay", "ok", "okaaay", "alright",
}
DEFAULT_HEDGES = {
    "maybe", "perhaps", "kinda", "sorta", "somewhat", "probably", "possibly", "i think",
    "i guess", "not sure", "idk", "ngl", "tbh",
}
DEFAULT_SLANG = {
    "lol", "lmao", "rofl", "omfg", "wtf", "idk", "btw", "asap", "tbh", "ikr", "fr",
    "smh", "brb", "afaik", "imho", "irl", "nsfw", "ofc", "af", "no cap", "bet", "mid", "sus",
    "yeet", "pog", "based", "cringe", "gyatt", "rizz",
}
DEFAULT_AFFECTIVE = {
    "haha", "hahaha", "hehe", "eheh", "hehehe", "heheh", "tehee", "aww", "awww", "mwah",
    "sigh", "tsk", "grr", "oof", "woah", "whoa", "omg", "ugh", "hmm",
}
DEFAULT_STOPLIKE = {"the", "a", "an", "and", "or", "but", "to", "of", "in", "on", "at", "for", "with"}

# -------------------------------------------------------------
# Dataclass for report
# -------------------------------------------------------------
@dataclass
class FeatureReport:
    """Structured feature report with both counts and normalized ratios.

    Fields are simple dicts so they can be JSON-serialized easily.
    """

    # Basic corpus stats
    meta: Dict[str, Any] = field(default_factory=dict)  # chars, tokens, sentences

    # Key categories
    structure: Dict[str, Any] = field(default_factory=dict)
    punctuation: Dict[str, Any] = field(default_factory=dict)
    casing: Dict[str, Any] = field(default_factory=dict)
    emoji: Dict[str, Any] = field(default_factory=dict)
    elongation: Dict[str, Any] = field(default_factory=dict)
    repetition: Dict[str, Any] = field(default_factory=dict)
    lexical: Dict[str, Any] = field(default_factory=dict)  # fillers/hedges/slang/affective
    special: Dict[str, Any] = field(default_factory=dict)  # keysmash, urls, handles
    negatives: Dict[str, Any] = field(default_factory=dict)

    examples: Dict[str, List[str]] = field(default_factory=dict)  # example strings for inspection

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

# -------------------------------------------------------------
# Utility functions
# -------------------------------------------------------------

def normalize_space(text: str) -> str:
    # Collapse whitespace, normalize some unicode punctuation spacing
    return WHITESPACE_RE.sub(" ", text.strip())


def split_sentences(text: str) -> List[str]:
    text = text.strip()
    if not text:
        return []
    # First, normalize unusual line breaks
    text = re.sub(r"\s*\n+\s*", " ", text)
    # Split by boundary regex; if none, return whole
    parts = SENT_SEP_RE.split(text)
    if len(parts) == 1:
        return [text]
    return [p.strip() for p in parts if p.strip()]


def tokenize(text: str) -> List[str]:
    # Basic whitespace tokenization; keep punctuation as separate tokens when surrounded by spaces
    # For our stats we mostly need wordish tokens and punctuation counts separately.
    # We'll return simple split for now and rely on specific regexes for counts.
    if not text:
        return []
    # Remove URLs/handles from token stats (still counted in special)
    cleaned = URL_RE.sub("", text)
    cleaned = HANDLE_RE.sub("", cleaned)
    return [t for t in WHITESPACE_RE.split(cleaned.strip()) if t]


def is_emoji(ch: str) -> bool:
    emojis = extract_emojis(ch)
    return bool(emojis)


def iter_emojis(text: str) -> List[str]:
    # Return grapheme-like emoji clusters by including skin tones and variation selectors
    # This is a simplification; full grapheme clustering would need regex with \X.
    return extract_emojis(text)



def ratio(n: int, d: int) -> float:
    return round((n / d) if d else 0.0, 6)


def safe_mean(vals: List[float]) -> float:
    return round(sum(vals) / len(vals), 6) if vals else 0.0


def count_elongations(tokens: List[str]) -> Tuple[int, List[str]]:
    matches = []
    for t in tokens:
        if ELONGATION_RE.search(t):
            matches.append(t)
    return len(matches), matches[:20]


def count_repeated_punct(text: str) -> Dict[str, int]:
    out: Dict[str, int] = {"!!!": 0, "??": 0, "...": 0, "!!??": 0}
    out["..."] = len(ELLIPSIS_RE.findall(text))
    out["!!!"] = len(re.findall(r"!{2,}", text))
    out["??"] = len(re.findall(r"\?{2,}", text))
    out["!!??"] = len(re.findall(r"(?:!\?){1,}|(?:\?!){1,}", text))
    return out


def count_case(tokens: List[str]) -> Tuple[int, int, int, int, List[str]]:
    all_caps = 0
    title = 0
    lower = 0
    mixed = 0
    examples: List[str] = []
    for t in tokens:
        w = re.sub(r"[^A-Za-z]", "", t)
        if len(w) < 2:
            continue
        if ALL_CAPS_RE.match(w):
            all_caps += 1
            if len(examples) < 10:
                examples.append(t)
        elif w.istitle():
            title += 1
        elif w.islower():
            lower += 1
        elif MIXED_CASE_RE.match(w):
            mixed += 1
    return all_caps, title, lower, mixed, examples


def sentence_start_lower(sent: str) -> bool:
    # First character that's a letter
    for ch in sent.strip():
        if ch.isalpha():
            return ch.islower()
    return False


def count_lowercase_i(tokens: List[str]) -> int:
    return len(re.findall(LOWER_I_RE, " ".join(tokens)))


def count_adjacent_repeat_words(text: str) -> Tuple[int, List[str]]:
    matches = [m.group(0) for m in REPEAT_WORD_RE.finditer(text)]
    return len(matches), matches[:10]


def count_emoji_positions(sentences: List[str]) -> Dict[str, Any]:
    # Position within a sentence: start/middle/end (per sentence with any emoji)
    start = mid = end = total = 0
    examples: Dict[str, List[str]] = {"start": [], "mid": [], "end": []}
    for s in sentences:
        em_indices = [i for i, ch in enumerate(s) if is_emoji(ch)]
        if not em_indices:
            continue
        total += 1
        first = em_indices[0]
        last = em_indices[-1]
        L = len(s)
        if first <= max(0, int(0.1 * L)):
            start += 1
            if len(examples["start"]) < 5:
                examples["start"].append(s)
        if last >= max(0, int(0.9 * L) - 1):
            end += 1
            if len(examples["end"]) < 5:
                examples["end"].append(s)
        if start + end < total:  # simplistic mid detection
            mid += 1
            if len(examples["mid"]) < 5:
                examples["mid"].append(s)
    return {
        "sentences_with_emoji": total,
        "start_ratio": ratio(start, total),
        "mid_ratio": ratio(mid, total),
        "end_ratio": ratio(end, total),
        "examples": examples,
    }


def detect_oxford_comma(text: str) -> Dict[str, Any]:
    with_ = len(OXFORD_COMMA_WITH_RE.findall(text))
    without = len(OXFORD_COMMA_WITHOUT_RE.findall(text))
    total = with_ + without
    return {
        "with_oxford": with_,
        "without_oxford": without,
        "oxford_ratio": ratio(with_, total),
    }


# -------------------------------------------------------------
# Analyzer class
# -------------------------------------------------------------
@dataclass
class LinguisticAnalyzer:
    fillers: Optional[Iterable[str]] = None
    hedges: Optional[Iterable[str]] = None
    slang: Optional[Iterable[str]] = None
    affective: Optional[Iterable[str]] = None
    stoplike: Optional[Iterable[str]] = None

    def __post_init__(self) -> None:
        self.fillers = set(self.fillers or DEFAULT_FILLERS)
        self.hedges = set(self.hedges or DEFAULT_HEDGES)
        self.slang = set(self.slang or DEFAULT_SLANG)
        self.affective = set(self.affective or DEFAULT_AFFECTIVE)
        self.stoplike = set(self.stoplike or DEFAULT_STOPLIKE)

    # ---------------------------
    # Core entry point
    # ---------------------------
    def analyze(self, text: str) -> FeatureReport:
        text = normalize_space(text)
        sentences = split_sentences(text)
        tokens = tokenize(text)

        chars = len(text)
        token_count = len(tokens)
        sentence_count = max(1, len(sentences))

        # Punctuation stats
        rep_punct = count_repeated_punct(text)
        comma = text.count(",")
        semicolon = text.count(";")
        colon = text.count(":")
        dash = text.count("—") + text.count("-")
        quotes_ascii = text.count('"') + text.count("'")
        quotes_smart = sum(text.count(q) for q in ["“", "”", "‘", "’"]) 

        # Casing stats
        all_caps, title, lower, mixed, case_examples = count_case(tokens)
        lowercase_start_count = sum(1 for s in sentences if sentence_start_lower(s))
        lowercase_i_count = count_lowercase_i(tokens)

        # Emoji stats
        emojis = iter_emojis(text)
        emoji_count = len(emojis)
        emoji_positions = count_emoji_positions(sentences)

        # Elongations
        elong_count, elong_examples = count_elongations(tokens)

        # Repetition
        repeat_word_count, repeat_word_examples = count_adjacent_repeat_words(text)

        # Lexical categories (predefined + dynamic frequency-based detection)
        canon_tokens = [normalize_token(t) for t in tokens]
        lex_lower = canon_tokens
        
        # Predefined categories
        filler_hits = [w for w in lex_lower if w in self.fillers]
        hedge_hits = [w for w in lex_lower if w in self.hedges]
        slang_hits = [w for w in lex_lower if w in self.slang]
        affect_hits = [w for w in lex_lower if w in self.affective]
        
        # Dynamic pattern detection - capture ALL words for frequency analysis
        all_words = [w for w in lex_lower if w.isalpha() and len(w) > 1]  # Filter out single chars and non-alpha
        
        # Detect potential user-specific patterns
        user_patterns = self._detect_user_patterns(all_words, text)

        # Special
        urls = URL_RE.findall(text)
        handles = HANDLE_RE.findall(text)
        keysmash_hits = KEYSMASH_RE.findall(text)

        # Structure
        words_total = sum(len(WORD_RE.findall(s)) for s in sentences)
        avg_sent_len_words = words_total / sentence_count if sentence_count else 0.0
        questions = sum(1 for s in sentences if s.strip().endswith("?"))
        exclaims = sum(1 for s in sentences if s.strip().endswith("!"))
        periods = sum(1 for s in sentences if s.strip().endswith("."))

        # Negative signals (single-text inference; aggregate over corpus in profile)
        negatives = {
            "never_capital_start": lowercase_start_count == sentence_count,
            "never_semicolon": semicolon == 0,
            "never_colon": colon == 0,
            "avoids_oxford_comma": detect_oxford_comma(text)["oxford_ratio"] < 0.2,
        }

        report = FeatureReport()
        report.meta = {
            "chars": chars,
            "tokens": token_count,
            "sentences": sentence_count,
        }

        report.structure = {
            "avg_sentence_len_words": round(avg_sent_len_words, 6),
            "questions_ratio": ratio(questions, sentence_count),
            "exclaims_ratio": ratio(exclaims, sentence_count),
            "period_ratio": ratio(periods, sentence_count),
        }

        report.punctuation = {
            "comma": comma,
            "semicolon": semicolon,
            "colon": colon,
            "dash": dash,
            "quotes_ascii": quotes_ascii,
            "quotes_smart": quotes_smart,
            **rep_punct,
            **detect_oxford_comma(text),
        }

        report.casing = {
            "all_caps_token_ratio": ratio(all_caps, token_count),
            "title_token_ratio": ratio(title, token_count),
            "lower_token_ratio": ratio(lower, token_count),
            "mixed_case_token_ratio": ratio(mixed, token_count),
            "lowercase_sentence_start_ratio": ratio(lowercase_start_count, sentence_count),
            "lowercase_i_count": lowercase_i_count,
        }

        report.emoji = {
            "emoji_count": emoji_count,
            "emoji_per_token": ratio(emoji_count, token_count),
            **emoji_positions,
        }

        report.elongation = {
            "elongation_token_ratio": ratio(elong_count, token_count),
            "count": elong_count,
        }

        report.repetition = {
            "adjacent_repeat_word_count": repeat_word_count,
        }

        report.lexical = {
            "fillers": _to_count_dict(filler_hits),
            "hedges": _to_count_dict(hedge_hits),
            "slang": _to_count_dict(slang_hits),
            "affective": _to_count_dict(affect_hits),
            "user_frequent": user_patterns.get("frequent_words", {}),
            "user_rare": user_patterns.get("rare_words", {}),
            "user_signature": user_patterns.get("signature_words", {}),
        }

        report.special = {
            "urls": urls[:20],
            "handles": handles[:20],
            "keysmash_count": len(keysmash_hits),
            "keysmash_examples": keysmash_hits[:10],
        }

        report.negatives = negatives

        report.examples = {
            "case_tokens": case_examples,
            "elongations": elong_examples,
            "repeat_words": repeat_word_examples,
        }

        return report

    def _detect_user_patterns(self, words: List[str], text: str) -> Dict[str, Dict[str, int]]:
        """Detect user-specific patterns beyond predefined categories."""
        patterns = {
            "frequent_words": {},
            "rare_words": {},
            "signature_words": {},
        }
        
        if not words:
            return patterns
        
        # Count word frequencies in this text
        word_counts = {}
        for word in words:
            word_counts[word] = word_counts.get(word, 0) + 1
        
        # Detect words repeated within this single text (potential user habits)
        for word, count in word_counts.items():
            if count > 1:  # Word appears multiple times in this text
                    patterns["frequent_words"][word] = count
        
        # Detect potentially signature words (uncommon words that might be user-specific)
        for word in words:
            # Look for words that might be user signatures:
            # Longer words, unusual letter patterns, or technical terms
            if (len(word) > 6 or 
                any(char in word for char in ['x', 'z', 'q']) or 
                word.endswith(('ing', 'tion', 'ness', 'ment'))):
                    patterns["signature_words"][word] = patterns["signature_words"].get(word, 0) + 1
        
        # Detect rare/unusual words based on intrinsic characteristics (not frequency)
        # Focus on words that are linguistically uncommon or distinctive
        for word in set(words):  # Use set to avoid duplicates
            is_rare = False
            
            # Very long words (8+ chars) are typically more distinctive
            if len(word) > 8:
                is_rare = True
            # Uncommon letter combinations
            elif any(combo in word for combo in ['xz', 'qw', 'zx', 'qq', 'xx', 'zz']):
                is_rare = True
            # Multiple unusual letters
            elif word.count('y') > 1 or word.count('z') > 0 or word.count('q') > 0:
                is_rare = True
            # Technical/academic suffixes
            elif word.endswith(('ology', 'ification', 'ness', 'ment', 'tion', 'sion', 'ism', 'itis', 'phobia')):
                is_rare = True
            # Technical/modern prefixes
            elif any(word.startswith(prefix) for prefix in ['pseudo', 'quasi', 'anti', 'super', 'ultra', 'mega', 'micro', 'nano', 'hyper', 'meta', 'crypto', 'cyber', 'bio', 'geo', 'eco']):
                is_rare = True
            # Words with apostrophes or hyphens (often distinctive)
            elif "'" in word or "-" in word:
                is_rare = True
                
            if is_rare:
                patterns["rare_words"][word] = patterns["rare_words"].get(word, 0) + 1
        
        return patterns

    def _detect_dynamic_slang(self, words: List[str], text: str) -> Dict[str, int]:
        """Detect slang words conservatively: prefer accuracy over recall.

        We match against a curated set of internet abbreviations and modern slang
        tokens. Unknown tokens will still be learned under user_* categories.
        """
        slang_words = {}

        abbreviations = {
            'ngl','af','rn','dm','pm','ig','bc','imo',
            'gg','omw','ftw','nvm','atm','jk','lmk','np','thx','pls','plz'
        }
        keywords = {
            'cap','periodt','bestie','bro','sis','fam','stan','ship','vibe',
            'flex','ghost','simp','cope','salty','toxic','woke','fire','slaps','hits',
            'lowkey','highkey','noob','pro','skibidi'
        }

        for word in set(words):
            count = text.lower().split().count(word)
            if count <= 0:
                continue
            if word in abbreviations or word in keywords:
                slang_words[word] = count

        return slang_words

    def _detect_dynamic_affective(self, words: List[str], text: str) -> Dict[str, int]:
        """Detect affective/emotional interjections and laughter only (strict).

        We intentionally avoid broad heuristics (like suffixes) to prevent
        misclassifying normal words. Emotional tone is otherwise captured by
        the emoji/punctuation/casing features and by the emotion model.
        """
        affective_words = {}

        interjections = {
            'eww','oh','ohh','ah','ahh','umm','mhm','yay','wow','gah','pfft',
            'meh','nah','yup','nope','damn','dang','gosh','jeez','geez','ouch','yikes','oops','duh','psst','shh'
        }
        sounds = {
            'sigh','grr','tsk','boo','mwah','smooch','giggle','snort','sob','wail','squeal','gasp','gulp','burp','hiccup'
        }

        for word in set(words):
            count = text.lower().split().count(word)
            if count <= 0:
                continue
            # Laughter variations and obvious patterns
            if any(pat in word for pat in ['haha','hehe','hihi','hoho','huhu','keke','jaja']):
                affective_words[word] = count
                continue
            if word in interjections or word in sounds:
                affective_words[word] = count

        return affective_words

    # ---------------------------
    # Phrase counts (for FAISS dedupe compatibility)
    # ---------------------------
    def extract_phrase_counts(self, text: str) -> Dict[str, int]:
        """Return a flat bag of phrases/markers with counts for indexing.
        Designed for compatibility with an external FAISS dedupe store.
        """
        text = normalize_space(text)
        tokens = tokenize(text)
        canon_tokens = [normalize_token(t) for t in tokens if t]
        counts: Dict[str, int] = {}

        # Fillers / hedges / slang / affective as phrases
        def add_many(label: str, words: List[str]) -> None:
            for w in words:
                key = f"[{label}] {w.lower()}"
                counts[key] = counts.get(key, 0) + 1

        lex_lower = canon_tokens
        add_many("filler", [w for w in lex_lower if w in self.fillers])
        add_many("hedge", [w for w in lex_lower if w in self.hedges])
        add_many("slang", [w for w in lex_lower if w in self.slang])
        add_many("affective", [w for w in lex_lower if w in self.affective])
        
        # Add dynamic user patterns
        all_words = [w for w in lex_lower if w.isalpha() and len(w) > 1]
        user_patterns = self._detect_user_patterns(all_words, text)
        
        # Add user-specific frequent words
        for word, count in user_patterns.get("frequent_words", {}).items():
            key = f"[user_frequent] {word}"
            counts[key] = counts.get(key, 0) + count
            
        # Add user signature words
        for word, count in user_patterns.get("signature_words", {}).items():
            key = f"[user_signature] {word}"
            counts[key] = counts.get(key, 0) + count
            
        # Add rare/unusual words
        for word, count in user_patterns.get("rare_words", {}).items():
            key = f"[user_rare] {word}"
            counts[key] = counts.get(key, 0) + count

        # Dynamic slang detection (beyond predefined list)
        dynamic_slang = self._detect_dynamic_slang(all_words, text)
        for word, count in dynamic_slang.items():
            key = f"[slang] {word}"
            counts[key] = counts.get(key, 0) + count

        # Dynamic affective detection (emotional expressions)
        dynamic_affective = self._detect_dynamic_affective(all_words, text)
        for word, count in dynamic_affective.items():
            key = f"[affective] {word}"
            counts[key] = counts.get(key, 0) + count

        # Elongations as phrase markers (pre-canonical examples for readability)
        el_count, el_examples = count_elongations(tokens)
        if el_count:
            for ex in el_examples:
                key = f"[elongation] {ex}"
                counts[key] = counts.get(key, 0) + 1

        # Repeated punctuation markers (aggregate)
        rep = count_repeated_punct(text)
        for k, v in rep.items():
            if v:
                key = f"[punct] {k}"
                counts[key] = counts.get(key, 0) + v

        # Case cues
        if any(ALL_CAPS_RE.match(re.sub(r"[^A-Za-z]", "", t)) for t in tokens):
            counts["[case] ALL_CAPS"] = counts.get("[case] ALL_CAPS", 0) + 1
        if any(sentence_start_lower(s) for s in split_sentences(text)):
            counts["[case] lowercase_sentence_starts"] = counts.get("[case] lowercase_sentence_starts", 0) + 1
        li = count_lowercase_i(tokens)
        if li:
            counts["[case] lowercase_i"] = counts.get("[case] lowercase_i", 0) + li

        # Emoji markers (aggregate & per-emoji)
        emojis = list(iter_emojis(text))
        ecount = len(emojis)
        if ecount:
            counts["[emoji] any"] = counts.get("[emoji] any", 0) + ecount
            # Track specific emoji types for richer style blueprint
            for ch in emojis:
                key = f"[emoji] {ch}"
                counts[key] = counts.get(key, 0) + 1

        # Special
        if KEYSMASH_RE.search(text):
            counts["[special] keysmash"] = counts.get("[special] keysmash", 0) + 1

        # N-grams (bigrams/trigrams) from canonical tokens
        def add_ngram(label: str, n: int, toks: List[str]) -> None:
            for i in range(len(toks) - n + 1):
                ng = toks[i:i+n]
                if all(w.isalpha() and len(w) > 1 for w in ng):
                    key = f"[{label}] {'_'.join(ng)}"
                    counts[key] = counts.get(key, 0) + 1
        add_ngram("ngram2", 2, canon_tokens)
        add_ngram("ngram3", 3, canon_tokens)

        return counts

    # ---------------------------
    # Report merging (for batch updates)
    # ---------------------------
    @staticmethod
    def merge_reports(reports: Iterable[FeatureReport]) -> FeatureReport:
        out = FeatureReport()
        total_tokens = 0
        total_sent = 0
        total_chars = 0

        def add_dict(dst: Dict[str, Any], src: Dict[str, Any]) -> None:
            for k, v in src.items():
                if isinstance(v, (int, float)):
                    dst[k] = dst.get(k, 0) + v
                elif isinstance(v, dict):
                    if k not in dst:
                        dst[k] = {}
                    add_dict(dst[k], v)
                else:
                    dst[k] = v  # last one wins for non-numeric non-dicts

        for r in reports:
            total_tokens += int(r.meta.get("tokens", 0))
            total_sent += int(r.meta.get("sentences", 0))
            total_chars += int(r.meta.get("chars", 0))
            add_dict(out.structure, r.structure)
            add_dict(out.punctuation, r.punctuation)
            add_dict(out.casing, r.casing)
            add_dict(out.emoji, r.emoji)
            add_dict(out.elongation, r.elongation)
            add_dict(out.repetition, r.repetition)
            # Merge lexical sub-dicts as counts
            for cat in ("fillers", "hedges", "slang", "affective", "user_frequent", "user_rare", "user_signature"):
                dst = out.lexical.setdefault(cat, {})
                src = r.lexical.get(cat, {})
                for k, v in src.items():
                    dst[k] = dst.get(k, 0) + v
            add_dict(out.special, r.special)
            add_dict(out.negatives, r.negatives)
            # examples not aggregated beyond cap to avoid bloat

        out.meta = {"chars": total_chars, "tokens": total_tokens, "sentences": total_sent}

        # Recompute derived ratios with totals where sensible
        if total_sent:
            for k in ("questions_ratio", "exclaims_ratio", "period_ratio"):
                if k in out.structure:
                    out.structure[k] = ratio(int(round(out.structure[k])), total_sent)
            # Fix avg_sentence_len_words: should be total_tokens / total_sentences, not summed averages
            if total_tokens:
                out.structure["avg_sentence_len_words"] = round(total_tokens / total_sent, 6)
        if total_tokens:
            for k in ("all_caps_token_ratio", "title_token_ratio", "lower_token_ratio", "mixed_case_token_ratio"):
                if k in out.casing:
                    out.casing[k] = ratio(int(round(out.casing[k])), total_tokens)
            out.emoji["emoji_per_token"] = ratio(int(round(out.emoji.get("emoji_count", 0))), total_tokens)
            out.elongation["elongation_token_ratio"] = ratio(int(round(out.elongation.get("count", 0))), total_tokens)

        return out


# -------------------------------------------------------------
# Helpers
# -------------------------------------------------------------

def _to_count_dict(items: Iterable[str]) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for x in items:
        out[x] = out.get(x, 0) + 1
    return out


# -------------------------------------------------------------
# Back-compat wrappers (so existing code keeps working)
# -------------------------------------------------------------

def extract_patterns_from_text(text: str) -> Dict[str, Dict[str, List[str]]]:
    """Compatibility layer returning a nested dict of pattern categories -> subcats -> examples.

    This wraps the advanced analyzer and maps core signals into familiar buckets
    (elongations, emojis, casing, punctuation, etc.).
    """
    analyzer = LinguisticAnalyzer()
    rep = analyzer.analyze(text)

    out: Dict[str, Dict[str, List[str]]] = {
        "elongations": {"examples": rep.examples.get("elongations", [])},
        "emoji": {"examples": list({ch for ch in iter_emojis(text)})[:20]},
        "casing": {
            "ALL_CAPS": rep.examples.get("case_tokens", [])[:10],
            "lowercase_sentence_starts": ["<detected>"] if rep.casing.get("lowercase_sentence_start_ratio", 0) > 0 else [],
            "lowercase_i": ["i"] * int(rep.casing.get("lowercase_i_count", 0)),
        },
        "punctuation": {
            k: [k] * int(v) for k, v in rep.punctuation.items() if k in {"!!!", "??", "...", "!!??"} and v
        },
        "repetition": {"adjacent_repeat": rep.examples.get("repeat_words", [])},
        "lexical": {
            "fillers": list(rep.lexical.get("fillers", {}).keys())[:20],
            "hedges": list(rep.lexical.get("hedges", {}).keys())[:20],
            "slang": list(rep.lexical.get("slang", {}).keys())[:20],
            "affective": list(rep.lexical.get("affective", {}).keys())[:20],
            "user_frequent": list(rep.lexical.get("user_frequent", {}).keys())[:10],
            "user_signature": list(rep.lexical.get("user_signature", {}).keys())[:10],
            "user_rare": list(rep.lexical.get("user_rare", {}).keys())[:10],
        },
        "negatives": {k: ["<true>"] for k, v in rep.negatives.items() if v},
    }
    return out


def extract_phrase_counts_from_text(analyzer: LinguisticAnalyzer, text: str) -> Dict[str, int]:
    """Compatibility shim to keep existing FAISS pipeline working."""
    if not isinstance(analyzer, LinguisticAnalyzer):
        analyzer = LinguisticAnalyzer()
    return analyzer.extract_phrase_counts(text)
