# -*- coding: utf-8 -*-
"""
profile_index.py — File 2 of 4 (Advanced)

Responsibilities
----------------
- EmbeddingModel: robust embedding loader (HF, SentenceTransformers, fallback hash).
- FAISSStore: similarity index with FAISS or NumPy fallback; phrase <-> vector mapping.
- EmotionAnalyzer: RoBERTa-only GoEmotions analyzer (HF pipeline; no heuristic fallback).
- StyleEmotionProfile: persistent, aggregating profile of style & emotion.
  * Global aggregates (structure, punctuation, casing, emoji, elongation, repetition, lexical, negatives)
  * Phrase counts for phrase-level style cues (to show Section 1 phrases if needed)
  * Conditional profiles by (topic_id, emotion_bucket)
  * Temporal signature (to be updated by conversation_context.py)
- Formatting helpers for the two exported sections used by your system.
- Dedupe helper: merge near-duplicate phrases via embeddings+index.

Notes
-----
- Pure Python + optional torch/transformers/faiss. Fallbacks keep it runnable.
- This file only defines storage/aggregation; extraction is in `linguistic_analyzer.py`.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Dict, List, Tuple, Optional, Iterable, Any
import os
import json
import hashlib
import unicodedata

# Optional heavy deps
try:
    import numpy as np  # type: ignore
except Exception:  # pragma: no cover
    np = None  # type: ignore

try:
    import faiss  # type: ignore
except Exception:  # pragma: no cover
    faiss = None  # type: ignore

# transformers optional
try:  # pragma: no cover
    import torch  # type: ignore
    from transformers import AutoTokenizer, AutoModel, pipeline  # type: ignore
except Exception:
    torch = None  # type: ignore
    AutoTokenizer = AutoModel = pipeline = None  # type: ignore

# Import pattern deduplicator
try:
    from pattern_deduplicator import PatternDeduplicator
except ImportError:
    PatternDeduplicator = None

# Local import types (runtime import by caller)
# from linguistic_analyzer import FeatureReport

__all__ = [
    "EmbeddingModel",
    "FAISSStore",
    "EmotionAnalyzer",
    "StyleEmotionProfile",
    "format_sections",
    "dedupe_and_add_phrase_counts",
]

# =============================================================
# Embedding model
# =============================================================

class EmbeddingModel:
    """Embeds short phrases/text for clustering/deduplication.

    Robust loader:
      - tries T5EncoderModel for encoder-only T5 checkpoints
      - falls back to AutoModelForSeq2SeqLM for seq2seq checkpoints
      - falls back to AutoModel (encoder-only / other)
      - final fallback: deterministic hash embedding
    """

    def __init__(self, model_path: str, device: Optional[str] = None, max_len: int = 128):
        self.model_path = model_path
        self.max_len = max_len
        self.device = device or ("cuda" if (torch and torch.cuda.is_available()) else "cpu")
        self._tokenizer = None
        self._model = None
        self.dim = 1024  # default fallback

        # local safe imports (some envs may not have transformers fully loaded at module import)
        from transformers import AutoConfig, AutoTokenizer, AutoModel, AutoModelForSeq2SeqLM, T5EncoderModel  # type: ignore

        # Try to inspect config to decide which HF class to load
        config = None
        if AutoConfig is not None:
            config = AutoConfig.from_pretrained(model_path, local_files_only=True)

        # Prefer encoder-only class for seq2seq checkpoints when available
        if config is not None and getattr(config, "is_encoder_decoder", False):
            # If possible, try to load an encoder-only model first (T5EncoderModel).
            if T5EncoderModel is not None:
                self._tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
                self._model = T5EncoderModel.from_pretrained(model_path, local_files_only=True)
                self.dim = getattr(self._model.config, "d_model", getattr(self._model.config, "hidden_size", self.dim))
            else:
                # No T5EncoderModel available -> fallback to seq2seq then AutoModel
                self._tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
                self._model = AutoModelForSeq2SeqLM.from_pretrained(model_path, local_files_only=True)
                self.dim = getattr(self._model.config, "d_model", getattr(self._model.config, "hidden_size", self.dim))
        else:
            # Non-encoder-decoder model: load regular AutoModel
            if AutoTokenizer is not None and AutoModel is not None:
                self._tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
                self._model = AutoModel.from_pretrained(model_path, local_files_only=True)
                self.dim = getattr(self._model.config, "hidden_size", self.dim)

        # CRITICAL: Move model to device to match input tensors
        if self._model is not None:
            self._model = self._model.to(self.device)
            self._model.eval()  # Set to evaluation mode

    def _mean_pool_tensor(self, token_embeddings, attention_mask):
        # token_embeddings: (batch, seq_len, dim), attention_mask: (batch, seq_len)
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = (token_embeddings * input_mask_expanded).sum(1)
        sum_mask = input_mask_expanded.sum(1)
        return sum_embeddings / torch.clamp(sum_mask, min=1e-9)

    def encode(self, texts: Iterable[str]) -> List[List[float]]:
        texts = [t if isinstance(t, str) else str(t) for t in texts]
        # Full HF path
        if self._model is not None and self._tokenizer is not None and torch is not None:
            with torch.no_grad():
                vecs: List[List[float]] = []
                for t in texts:
                    inputs = self._tokenizer(t, truncation=True, max_length=self.max_len, return_tensors="pt")
                    inputs = {k: v.to(self.device) for k, v in inputs.items()}
                    # If the model is a seq2seq object with encoder, prefer encoder outputs.
                    try:
                        if hasattr(self._model, "get_encoder"):
                            encoder = self._model.get_encoder()
                            enc_out = encoder(**inputs)
                            token_embeddings = enc_out.last_hidden_state
                        elif hasattr(self._model, "encoder"):
                            enc = getattr(self._model, "encoder")
                            enc_out = enc(**inputs)
                            token_embeddings = enc_out.last_hidden_state
                        else:
                            out = self._model(**inputs)
                            # standard HF model last_hidden_state
                            token_embeddings = out[0]
                    except Exception:
                        # best-effort fallback: call model and take first output
                        out = self._model(**inputs)
                        token_embeddings = out[0]

                    attention_mask = inputs.get("attention_mask", None)
                    pooled = self._mean_pool_tensor(token_embeddings, attention_mask)
                    v = pooled[0].detach().cpu().numpy().astype("float32")
                    v = v / (np.linalg.norm(v) + 1e-9) if np is not None else v
                    vecs.append(v.tolist())
                return vecs

        # Fallback: deterministic hash embedding
        return [self._hash_embed(t, self.dim) for t in texts]


    @staticmethod
    def _hash_embed(text: str, dim: int) -> List[float]:
        # Deterministic pseudo-embedding based on SHA256
        h = hashlib.sha256(text.encode("utf-8")).digest()
        # Expand digest to required size
        needed = dim * 4
        buf = (h * ((needed // len(h)) + 1))[:needed]
        if np is None:
            # Crude non-numpy variant
            ints = [buf[i] for i in range(0, len(buf))]
            # Map to [-1, 1]
            vec = [(x / 127.5 - 1.0) for x in ints[:dim]]
            return vec
        arr = np.frombuffer(buf, dtype=np.uint8)[: dim * 4]
        arr = arr.view(np.float32)
        # Normalize
        arr = arr / (np.linalg.norm(arr) + 1e-9)
        return arr.tolist()


# =============================================================
# FAISS-backed (or NumPy) vector store
# =============================================================

class FAISSStore:
    def __init__(self, dim: int, metric: str = "ip"):
        self.dim = dim
        self.metric = metric
        self._index = None
        self._keys: List[str] = []
        self._vecs = None  # NumPy fallback storage
        self._use_faiss = faiss is not None and np is not None

        if self._use_faiss:
            if metric == "ip":
                self._index = faiss.IndexFlatIP(dim)
            elif metric == "l2":
                self._index = faiss.IndexFlatL2(dim)
            else:
                raise ValueError("metric must be 'ip' or 'l2'")
        elif np is not None:
            self._vecs = np.zeros((0, dim), dtype="float32")

    def add(self, vectors: List[List[float]], keys: List[str]) -> None:
        if len(vectors) != len(keys):
            raise ValueError("vectors and keys must be same length")
        if np is None:
            # nothing to do; keep only keys to preserve count
            self._keys.extend(keys)
            return
        X = np.array(vectors, dtype="float32")
        if self._use_faiss:
            self._index.add(X)
        else:
            self._vecs = np.vstack([self._vecs, X]) if self._vecs is not None else X
        self._keys.extend(keys)

    def search(self, query: List[float], k: int = 5) -> Tuple[List[int], List[float]]:
        if np is None:
            return [], []
        q = np.array(query, dtype="float32")[None, :]
        if self._use_faiss:
            D, I = self._index.search(q, k)
            return I[0].tolist(), D[0].tolist()
        # NumPy fallback (cosine sim)
        if self._vecs is None or len(self._keys) == 0:
            return [], []
        # normalize
        V = self._vecs
        qn = q / (np.linalg.norm(q) + 1e-9)
        Vn = V / (np.linalg.norm(V, axis=1, keepdims=True) + 1e-9)
        sims = (Vn @ qn.T).ravel()
        idx = np.argsort(-sims)[:k]
        return idx.tolist(), sims[idx].tolist()

    def save(self, directory: str, basename: str) -> None:
        os.makedirs(directory, exist_ok=True)
        meta = {
            "dim": self.dim,
            "metric": self.metric,
            "keys": self._keys,
            "use_faiss": bool(self._use_faiss),
        }
        with open(os.path.join(directory, f"{basename}.meta.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
        if self._use_faiss and self._index is not None:
            faiss.write_index(self._index, os.path.join(directory, f"{basename}.index"))
        elif np is not None and self._vecs is not None:
            np.save(os.path.join(directory, f"{basename}.npy"), self._vecs)

    def load(self, directory: str, basename: str) -> None:
        with open(os.path.join(directory, f"{basename}.meta.json"), "r", encoding="utf-8") as f:
            meta = json.load(f)
        self.dim = meta.get("dim", self.dim)
        self.metric = meta.get("metric", self.metric)
        self._keys = list(meta.get("keys", []))
        self._use_faiss = bool(meta.get("use_faiss", False) and faiss is not None and np is not None)
        if self._use_faiss:
            path = os.path.join(directory, f"{basename}.index")
            if os.path.exists(path):
                self._index = faiss.read_index(path)
            else:
                self._index = faiss.IndexFlatIP(self.dim) if self.metric == "ip" else faiss.IndexFlatL2(self.dim)
        elif np is not None:
            path = os.path.join(directory, f"{basename}.npy")
            if os.path.exists(path):
                self._vecs = np.load(path)
            else:
                self._vecs = np.zeros((0, self.dim), dtype="float32")

    @property
    def keys(self) -> List[str]:
        return self._keys


# =============================================================
# Emotion analyzer (GoEmotions compatible)
# =============================================================

GOEMOTIONS_28 = [
    # 27 emotions + neutral (commonly used compact set)
    'admiration','amusement','anger','annoyance','approval','caring','confusion','curiosity','desire','disappointment',
    'disapproval','disgust','embarrassment','excitement','fear','gratitude','grief','joy','love','nervousness',
    'optimism','pride','realization','relief','remorse','sadness','surprise','neutral'
]

class EmotionAnalyzer:
    def __init__(self, model_path: str = "", device: Optional[str] = None):
        self.labels = list(GOEMOTIONS_28)
        self._pipe = None
        self.device = device
        # Require a model path for proper emotion extraction; no heuristic fallbacks
        if pipeline is not None and model_path:
            try:  # multi-label classifier
                self._pipe = pipeline(
                    task="text-classification",
                    model=model_path,
                    tokenizer=model_path,
                    top_k=None,
                    function_to_apply="sigmoid",
                    device=0 if (torch and torch.cuda.is_available()) else -1,
                )
            except Exception:
                self._pipe = None

    def scores(self, text: str) -> Dict[str, float]:
        text = (text or "").strip()
        if not text:
            return {k: 0.0 for k in self.labels}
        if self._pipe is None:
            # If model not available, return zeros (no heuristic fallback)
            return {k: 0.0 for k in self.labels}
        try:
            preds = self._pipe(text)[0]  # list of {label, score}
            raw = {p['label'].lower(): float(p['score']) for p in preds}
            out = {k: float(raw.get(k, 0.0)) for k in self.labels}
            s = sum(out.values())
            if s <= 0:
                return {k: 0.0 for k in self.labels}
            # Normalize to sum=1.0 probabilities
            return {k: v / s for k, v in out.items()}
        except Exception:
            return {k: 0.0 for k in self.labels}


# =============================================================
# Style & Emotion profile (persistent)
# =============================================================

@dataclass
class StyleEmotionProfile:
    # Aggregated raw totals
    totals: Dict[str, float] = field(default_factory=lambda: {"tokens": 0.0, "sentences": 0.0, "chars": 0.0})

    # Category aggregates (sums)
    structure: Dict[str, float] = field(default_factory=dict)
    punctuation: Dict[str, float] = field(default_factory=dict)
    casing: Dict[str, float] = field(default_factory=dict)
    emoji: Dict[str, float] = field(default_factory=dict)
    elongation: Dict[str, float] = field(default_factory=dict)
    repetition: Dict[str, float] = field(default_factory=dict)
    lexical: Dict[str, Dict[str, float]] = field(default_factory=lambda: {
        "fillers": {}, "hedges": {}, "slang": {}, "affective": {},
        "user_frequent": {}, "user_rare": {}, "user_signature": {}
    })
    negatives: Dict[str, float] = field(default_factory=dict)

    # Phrase-level signals
    phrase_counts: Dict[str, int] = field(default_factory=dict)

    # Session (recency) structures for EMA blending
    session_lexical: Dict[str, Dict[str, float]] = field(default_factory=lambda: {
        "fillers": {}, "hedges": {}, "slang": {}, "affective": {},
        "user_frequent": {}, "user_rare": {}, "user_signature": {}
    })
    session_phrase_counts: Dict[str, int] = field(default_factory=dict)

    # Emotion distribution (running sum of probabilities)
    emotions: Dict[str, float] = field(default_factory=lambda: {k: 0.0 for k in GOEMOTIONS_28})
    emotion_updates: int = 0
    # Per-message emotion percentages history (each entry sums to ~100)
    emotion_history: List[Dict[str, float]] = field(default_factory=list)

    # Conditional profiles: (topic_id, emotion_bucket) -> lightweight aggregates
    conditional: Dict[str, Dict[str, float]] = field(default_factory=dict)

    # Temporal signature container (updated by conversation_context)
    temporal: Dict[str, float] = field(default_factory=dict)
    
    # Pattern deduplicator for intelligent clustering (explicit init; no fallbacks)
    _deduplicator: Optional[Any] = field(default_factory=lambda: None, init=False)
    _dedup_store_dir: Optional[str] = field(default=None, init=False)
    _dedup_store_basename: str = field(default="pattern_store", init=False)
    _dedup_memory_path: Optional[str] = field(default=None, init=False)
    _ema_decay: float = field(default=0.9, init=False)
    
    def __post_init__(self):
        """Do not auto-initialize deduplicator. Call initialize_deduplicator() explicitly."""
        self._deduplicator = None

    def initialize_deduplicator(self, embed_model_path: str, store_dir: str) -> None:
        """Strict initialization of dedup engine and FAISS store. No fallbacks."""
        if not PatternDeduplicator:
            raise RuntimeError("PatternDeduplicator module not available")
        if not embed_model_path or not os.path.exists(embed_model_path):
            raise FileNotFoundError(f"Embedding model path missing: {embed_model_path}")
        if not store_dir:
            raise ValueError("store_dir is required for FAISS persistence")
        os.makedirs(store_dir, exist_ok=True)

        # Build embedding and FAISS store (must succeed)
        embed = EmbeddingModel(embed_model_path)
        if getattr(embed, "_model", None) is None:
            raise RuntimeError("EmbeddingModel failed to load HF model (strict mode)")
        store = FAISSStore(dim=embed.dim, metric="ip")

        # Persist paths
        self._dedup_store_dir = store_dir
        self._dedup_memory_path = os.path.join(store_dir, "pattern_memory.json")

        # Load prior state (strict: require presence)
        meta_path = os.path.join(store_dir, f"{self._dedup_store_basename}.meta.json")
        if os.path.exists(meta_path):
            store.load(store_dir, self._dedup_store_basename)
        else:
            # Start with empty store but still strict init
            store.save(store_dir, self._dedup_store_basename)

        self._deduplicator = PatternDeduplicator(embed_model=embed, faiss_store=store)
        # Load dedup memory (if exists), else create empty file immediately
        if os.path.exists(self._dedup_memory_path):
            self._deduplicator.load_memory(self._dedup_memory_path)
        else:
            self._deduplicator.save_memory(self._dedup_memory_path)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        # Remove non-serializable/transient fields
        d.pop('_deduplicator', None)
        return d

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "StyleEmotionProfile":
        prof = StyleEmotionProfile()
        for k, v in d.items():
            setattr(prof, k, v)
        return prof

    # -----------------------
    # Update pipelines
    # -----------------------
    def update_from_features(self, rep: Any) -> None:
        """Ingest a FeatureReport (from linguistic_analyzer.analyze)."""
        # Apply recency decay to session structures
        self.apply_recency_decay(self._ema_decay)
        meta = rep.meta or {}
        self.totals["tokens"] += float(meta.get("tokens", 0))
        self.totals["sentences"] += float(meta.get("sentences", 0))
        self.totals["chars"] += float(meta.get("chars", 0))

        def add(dst: Dict[str, float], src: Dict[str, Any]) -> None:
            for k, v in (src or {}).items():
                if isinstance(v, (int, float)):
                    dst[k] = dst.get(k, 0.0) + float(v)

        # Handle structure separately to recalculate avg_sentence_len_words properly
        structure_data = dict(rep.structure or {})
        if "avg_sentence_len_words" in structure_data:
            # Remove avg_sentence_len_words from structure_data as we'll recalculate it
            structure_data.pop("avg_sentence_len_words")
        add(self.structure, structure_data)
        
        # Recalculate avg_sentence_len_words from total tokens and sentences
        total_tokens = self.totals.get("tokens", 0.0)
        total_sentences = self.totals.get("sentences", 0.0)
        if total_sentences > 0:
            self.structure["avg_sentence_len_words"] = round(total_tokens / total_sentences, 6)
        
        add(self.punctuation, rep.punctuation)
        add(self.casing, rep.casing)
        add(self.emoji, rep.emoji)
        add(self.elongation, rep.elongation)
        add(self.repetition, rep.repetition)
        add(self.negatives, rep.negatives)

        # lexical sub-dicts (lifetime)
        for cat in ("fillers", "hedges", "slang", "affective", "user_frequent", "user_rare", "user_signature"):
            src = rep.lexical.get(cat, {}) if rep.lexical else {}
            dst = self.lexical.get(cat, {})
            for w, c in src.items():
                dst[w] = float(dst.get(w, 0.0) + float(c))
            self.lexical[cat] = dst
        # session EMA update
        decay = self._ema_decay
        for cat in ("fillers", "hedges", "slang", "affective", "user_frequent", "user_rare", "user_signature"):
            src = rep.lexical.get(cat, {}) if rep.lexical else {}
            sess = self.session_lexical.get(cat, {})
            for w, c in src.items():
                prev = float(sess.get(w, 0.0)) * decay
                sess[w] = prev + (1.0 - decay) * float(c)
            self.session_lexical[cat] = sess

    def update_phrase_counts(self, counts: Dict[str, int]) -> None:
        """Update phrase counts with intelligent deduplication."""
        if not counts:
            return
        if not self._deduplicator:
            raise RuntimeError("Pattern deduplicator not initialized. Call initialize_deduplicator() before updates.")

        # Apply recency decay to session structures
        self.apply_recency_decay(self._ema_decay)

        # Extract patterns by category for deduplication
        # Organize patterns by category
        pattern_categories = {
            'elongations': {},
            'fillers': {},
            'hedges': {},
            'slang': {},
            'affective': {},
            'signature_words': {},
            'rare_words': {},
            'frequent_words': {},
            'emojis': {}
        }

        # Parse phrase counts into categories
        for phrase_key, count in counts.items():
            if phrase_key.startswith('[elongation] '):
                word = phrase_key[13:]
                pattern_categories['elongations'][word] = count
            elif phrase_key.startswith('[filler] '):
                word = phrase_key[9:]
                pattern_categories['fillers'][word] = count
            elif phrase_key.startswith('[hedge] '):
                word = phrase_key[8:]
                pattern_categories['hedges'][word] = count
            elif phrase_key.startswith('[slang] '):
                word = phrase_key[8:]
                pattern_categories['slang'][word] = count
            elif phrase_key.startswith('[affective] '):
                word = phrase_key[12:]
                pattern_categories['affective'][word] = count
            elif phrase_key.startswith('[emoji] ') and phrase_key != '[emoji] any':
                emoji = phrase_key[8:]
                pattern_categories['emojis'][emoji] = count
            elif phrase_key.startswith('[user_frequent] '):
                word = phrase_key[16:]
                pattern_categories['frequent_words'][word] = count
            elif phrase_key.startswith('[user_rare] '):
                word = phrase_key[12:]
                pattern_categories['rare_words'][word] = count
            elif phrase_key.startswith('[user_signature] '):
                word = phrase_key[17:]
                pattern_categories['signature_words'][word] = count
            else:
                # Keep other phrases as-is
                self.phrase_counts[phrase_key] = self.phrase_counts.get(phrase_key, 0) + count

        # Deduplicate each category
        deduplicated = self._deduplicator.deduplicate_patterns(pattern_categories)

        # Update lexical categories with deduplicated results
        for category, word_counts in deduplicated.items():
            if category == 'elongations':
                for word, count in word_counts.items():
                    key = f'[elongation] {word}'
                    self.phrase_counts[key] = self.phrase_counts.get(key, 0) + count
                    # session update
                    prev = int(round(self.session_phrase_counts.get(key, 0) * self._ema_decay))
                    self.session_phrase_counts[key] = prev + int(round((1.0 - self._ema_decay) * count))
            elif category == 'emojis':
                for emoji, count in word_counts.items():
                    key = f'[emoji] {emoji}'
                    self.phrase_counts[key] = self.phrase_counts.get(key, 0) + count
                    prev = int(round(self.session_phrase_counts.get(key, 0) * self._ema_decay))
                    self.session_phrase_counts[key] = prev + int(round((1.0 - self._ema_decay) * count))
            elif category in ['fillers', 'hedges', 'slang', 'affective']:
                lexical_cat = category
                if lexical_cat not in self.lexical:
                    self.lexical[lexical_cat] = {}
                for word, count in word_counts.items():
                    self.lexical[lexical_cat][word] = self.lexical[lexical_cat].get(word, 0) + count
                    sess = self.session_lexical.get(lexical_cat, {})
                    prev = float(sess.get(word, 0.0)) * self._ema_decay
                    sess[word] = prev + (1.0 - self._ema_decay) * float(count)
                    self.session_lexical[lexical_cat] = sess
            elif category in ['signature_words', 'rare_words', 'frequent_words']:
                mapping = {
                    'signature_words': 'user_signature',
                    'rare_words': 'user_rare',
                    'frequent_words': 'user_frequent',
                }
                user_cat = mapping[category]
                if user_cat not in self.lexical:
                    self.lexical[user_cat] = {}
                for word, count in word_counts.items():
                    self.lexical[user_cat][word] = self.lexical[user_cat].get(word, 0) + count
                    sess = self.session_lexical.get(user_cat, {})
                    prev = float(sess.get(word, 0.0)) * self._ema_decay
                    sess[word] = prev + (1.0 - self._ema_decay) * float(count)
                    self.session_lexical[user_cat] = sess

        # Persist dedup state if paths configured
        try:
            if hasattr(self._deduplicator, 'save_memory') and self._dedup_memory_path:
                os.makedirs(os.path.dirname(self._dedup_memory_path), exist_ok=True)
                self._deduplicator.save_memory(self._dedup_memory_path)
            if hasattr(self._deduplicator, 'faiss_store') and self._dedup_store_dir:
                self._deduplicator.faiss_store.save(self._dedup_store_dir, self._dedup_store_basename)
        except Exception:
            # Strict mode: do not silence; raise to caller for visibility
            raise

    def apply_recency_decay(self, decay: float = 0.9) -> None:
        """Apply exponential decay to session (recency) structures only."""
        if decay <= 0.0 or decay >= 1.0:
            decay = self._ema_decay
        # Decay session lexical
        for cat in ("fillers", "hedges", "slang", "affective", "user_frequent", "user_rare", "user_signature"):
            bucket = self.session_lexical.get(cat, {}) or {}
            for k in list(bucket.keys()):
                bucket[k] = float(bucket.get(k, 0.0)) * decay
            self.session_lexical[cat] = bucket
        # Decay session phrase counts
        for k in list(self.session_phrase_counts.keys()):
            self.session_phrase_counts[k] = int(round(self.session_phrase_counts.get(k, 0) * decay))

    def update_emotions(self, scores: Dict[str, float]) -> None:
        if not scores:
            return
        # scores are expected as probabilities summing to 1.0
        total = float(sum(scores.values()))
        if total <= 0:
            return
        # Store per-message percentages (0..100), normalized to sum exactly 100.00
        pct_entry: Dict[str, float] = {k: round(100.0 * float(v) / total, 2) for k, v in scores.items()}
        # Adjust rounding drift on the max entry
        if pct_entry:
            sum_pct = round(sum(pct_entry.values()), 2)
            drift = round(100.0 - sum_pct, 2)
            if abs(drift) >= 0.01:
                max_key = max(pct_entry.items(), key=lambda kv: kv[1])[0]
                pct_entry[max_key] = round(pct_entry[max_key] + drift, 2)
        self.emotion_history.append(pct_entry)
        # Accumulate running sums as probabilities
        for k, v in scores.items():
            if k not in self.emotions:
                self.emotions[k] = 0.0
            self.emotions[k] += float(v) / total  # normalize defensively
        self.emotion_updates += 1

    # Back-compat: accept pattern_data from old extractor wrapper
    def update_pattern_data(self, pattern_data: Dict[str, Dict[str, List[str]]]) -> None:
        if not pattern_data:
            return
        # Map minimal cues to punctuation/casing/emoji/elongation buckets
        punct = pattern_data.get("punctuation", {})
        for k, examples in punct.items():
            if isinstance(examples, list):
                self.punctuation[k] = self.punctuation.get(k, 0.0) + float(len(examples))
        casing = pattern_data.get("casing", {})
        if casing.get("ALL_CAPS"):
            self.casing["ALL_CAPS_hits"] = self.casing.get("ALL_CAPS_hits", 0.0) + float(len(casing.get("ALL_CAPS", [])))
        if casing.get("lowercase_sentence_starts"):
            self.casing["lowercase_sentence_starts"] = self.casing.get("lowercase_sentence_starts", 0.0) + 1.0
        li = casing.get("lowercase_i", [])
        if isinstance(li, list) and li:
            self.casing["lowercase_i_count"] = self.casing.get("lowercase_i_count", 0.0) + float(len(li))
        em = pattern_data.get("emoji", {})
        if em.get("examples"):
            self.emoji["emoji_count"] = self.emoji.get("emoji_count", 0.0) + float(len(em.get("examples", [])))
        el = pattern_data.get("elongations", {})
        if el.get("examples"):
            self.elongation["count"] = self.elongation.get("count", 0.0) + float(len(el.get("examples", [])))
        neg = pattern_data.get("negatives", {})
        for k, v in neg.items():
            if v:
                self.negatives[k] = self.negatives.get(k, 0.0) + 1.0

    # Conditional mini-profiles (topic_id + emotion bucket)
    # Note: conditional and temporal fields are maintained for future use; no update methods are required here.

    # -----------------------
    # Export sections
    # -----------------------
    def section1_percentages(self) -> List[Tuple[str, str, float]]:
        """Return list of (label, cue, percent) sorted desc.

        Heuristic mapping from aggregates -> interpretable style signals.
        Percentages are normalized within Section 1.
        """
        # Raw feature signals (non-negative)
        sig: List[Tuple[str, float]] = []
        tokens = max(1.0, self.totals.get("tokens", 1.0))
        sentences = max(1.0, self.totals.get("sentences", 1.0))

        # Core cues
        lower_start = float(self.casing.get("lowercase_sentence_start_ratio", 0.0)) * sentences
        sig.append(("lowercase sentence starts", lower_start))

        all_caps_hits = float(self.casing.get("ALL_CAPS_hits", 0.0)) + float(self.casing.get("all_caps_token_ratio", 0.0) * tokens)
        sig.append(("ALL-CAPS emphasis", all_caps_hits))

        elong = float(self.elongation.get("count", 0.0)) + float(self.elongation.get("elongation_token_ratio", 0.0) * tokens)
        sig.append(("elongations", elong))

        emoji_any = float(self.emoji.get("emoji_count", 0.0)) + float(self.emoji.get("emoji_per_token", 0.0) * tokens)
        sig.append(("emojis", emoji_any))

        lower_i = float(self.casing.get("lowercase_i_count", 0.0))
        sig.append(("lowercase 'i' pronoun", lower_i))

        # Fillers as proportion of lexical items encountered
        filler_sum = sum(self.lexical.get("fillers", {}).values())
        sig.append(("fillers", float(filler_sum)))

        # Repeated punctuation
        repeat_punct = 0.0
        for k in ("!!!", "??", "...", "!!??"):
            repeat_punct += float(self.punctuation.get(k, 0.0))
        sig.append(("repeated punctuation", repeat_punct))

        # Normalize to percentages
        total = sum(max(0.0, v) for _, v in sig)
        if total <= 0:
            return []
        def _fmt(val: float) -> str:
            if val >= 10:
                return f"{val:.0f}×"
            if val >= 1:
                return f"{val:.1f}×"
            return f"{val:.2f}×"
        out = [(label, _fmt(val), round(100.0 * val / total, 2)) for (label, val) in sig]
        out.sort(key=lambda x: x[2], reverse=True)
        return out

    def section2_percentages(self) -> List[Tuple[str, float]]:
        # Average emotion distribution across updates
        if self.emotion_updates <= 0:
            return []
        avg = {k: (v / max(1.0, self.emotion_updates)) for k, v in self.emotions.items()}
        total = sum(avg.values())
        if total <= 0:
            return []
        items = [(k, 100.0 * v / total) for k, v in avg.items()]
        items.sort(key=lambda x: x[1], reverse=True)
        return [(k, round(p, 2)) for k, p in items]

    # -----------------------
    # Dynamic LLM System Prompt Generation
    # -----------------------
    def get_style_blueprint_percentages(self) -> Dict[str, float]:
        """Generate comprehensive Style Blueprint percentages for LLM system prompt."""
        total_samples = max(1.0, self.totals.get("sentences", 1.0))
        blueprint = {}
        
        # 1. Casing Patterns
        lowercase_starts = self.casing.get("lowercase_sentence_starts", 0)
        if lowercase_starts > 0:
            blueprint['lowercase_starts'] = round((lowercase_starts / total_samples) * 100, 1)
        
        # Prefer ratio-based all-caps if available
        all_caps_ratio = self.casing.get("all_caps_token_ratio", 0)
        if all_caps_ratio > 0:
            blueprint['all_caps'] = round(all_caps_ratio * 100, 1)
        else:
            all_caps = self.casing.get("ALL_CAPS_hits", 0)
            if all_caps > 0:
                blueprint['all_caps'] = round((all_caps / total_samples) * 100, 1)
        
        lowercase_i = self.casing.get("lowercase_i_count", 0)
        if lowercase_i > 0:
            blueprint['lowercase_i'] = round((lowercase_i / total_samples) * 100, 1)
        
        # 2. Elongation Patterns
        elongations = self.elongation.get("count", 0)
        if elongations > 0:
            blueprint['elongations'] = round((elongations / total_samples) * 100, 1)
        
        # 3. Emoji Patterns  
        emojis = self.emoji.get("emoji_count", 0)
        if emojis > 0:
            blueprint['emojis'] = round((emojis / total_samples) * 100, 1)
        
        # 4. Punctuation Patterns
        # Sum canonical repeated punctuation counters
        repeated_punct = 0
        for k in ("!!!", "??", "...", "!!??"):
            repeated_punct += int(self.punctuation.get(k, 0))
        if repeated_punct > 0:
            blueprint['repeated_punct'] = round((repeated_punct / total_samples) * 100, 1)
        
        exclamations = self.structure.get("exclaims_ratio", 0)
        if exclamations > 0:
            blueprint['exclamations'] = round(exclamations * 100, 1)
        
        # 5. Lexical Patterns
        fillers = float(sum((self.lexical.get("fillers", {}) or {}).values()))
        if fillers > 0:
            blueprint['fillers'] = round((fillers / total_samples) * 100, 1)
        
        affective = float(sum((self.lexical.get("affective", {}) or {}).values()))
        if affective > 0:
            blueprint['affective'] = round((affective / total_samples) * 100, 1)
        
        slang = float(sum((self.lexical.get("slang", {}) or {}).values()))
        if slang > 0:
            blueprint['slang'] = round((slang / total_samples) * 100, 1)
        
        # 6. Repetition Patterns
        repeats = self.repetition.get("adjacent_repeat_word_count", 0)
        if repeats > 0:
            blueprint['word_repeats'] = round((repeats / total_samples) * 100, 1)
        
        # 7. Structure Patterns
        avg_length = self.structure.get("avg_sentence_len_words", 0)
        if avg_length > 0 and avg_length != 15.0:  # Don't include if default
            blueprint['avg_sentence_length'] = round(avg_length, 1)
        
        # 8. Grammar Negatives (as boolean percentages)
        no_capital = self.negatives.get("never_capital_start", 0)
        total_updates = max(1.0, self.totals.get("sentences", 1.0))
        if no_capital > total_updates * 0.3:  # If 30%+ of texts avoid capitals
            blueprint['avoids_capitals'] = round((no_capital / total_updates) * 100, 1)
        
        no_oxford = self.negatives.get("avoids_oxford_comma", 0)
        if no_oxford > total_updates * 0.3:
            blueprint['avoids_oxford'] = round((no_oxford / total_updates) * 100, 1)
        
        # Return all non-negative patterns; no visibility cutoff in strict mode
        return {k: v for k, v in blueprint.items() if v >= 0.0}

    def get_emotion_percentages(self) -> Dict[str, float]:
        """Get dynamic emotion percentages based on total emotional mass (top 15)."""
        if not self.emotions:
            return {}
        total_mass = sum(self.emotions.values())
        if total_mass == 0:
            return {}
        # Convert to percentages and pick top 15
        percentages: Dict[str, float] = {
            emotion: round((value / total_mass) * 100, 2) for emotion, value in self.emotions.items()
        }
        sorted_emotions = dict(sorted(percentages.items(), key=lambda x: x[1], reverse=True))
        top_n = min(15, len(sorted_emotions))
        return dict(list(sorted_emotions.items())[:top_n])

    def _topn_from_counts(self, counts: Dict[str, float], n: int = 5) -> list[tuple[str, int]]:
        items = sorted(((k, int(v)) for k, v in counts.items()), key=lambda x: x[1], reverse=True)
        return items[:n]

    def format_for_llm_system_prompt(self) -> str:
        """Format Emotional Palette and a rich Style Blueprint with exemplars for LLM."""
        sections: list[str] = []

        # Emotional Palette (at least 10 if possible)
        emotions = self.get_emotion_percentages()
        if emotions:
            emotion_pairs = [f"{emotion}: {pct}%" for emotion, pct in emotions.items()]
            sections.append(f"Emotional Palette → {', '.join(emotion_pairs)}")

        # Style Blueprint (percentages + exemplars)
        style_patterns = self.get_style_blueprint_percentages()
        exemplars: list[str] = []

        # Top fillers/hedges/slang/affective words + user-specific patterns
        if self.lexical.get("fillers"):
            tops = self._topn_from_counts(self.lexical["fillers"], 5)
            exemplars.append("fillers: " + ", ".join([f"{w}×{c}" for w, c in tops]))
        if self.lexical.get("hedges"):
            tops = self._topn_from_counts(self.lexical["hedges"], 3)
            exemplars.append("hedges: " + ", ".join([f"{w}×{c}" for w, c in tops]))
        if self.lexical.get("slang"):
            tops = self._topn_from_counts(self.lexical["slang"], 5)
            exemplars.append("slang: " + ", ".join([f"{w}×{c}" for w, c in tops]))
        if self.lexical.get("affective"):
            tops = self._topn_from_counts(self.lexical["affective"], 5)
            exemplars.append("affective: " + ", ".join([f"{w}×{c}" for w, c in tops]))
            
        # User-specific frequent words (personal habits)
        if self.lexical.get("user_frequent"):
            tops = self._topn_from_counts(self.lexical["user_frequent"], 5)
            exemplars.append("user_frequent: " + ", ".join([f"{w}×{c}" for w, c in tops]))
            
        # User signature words (unique/uncommon words they use)
        if self.lexical.get("user_signature"):
            tops = self._topn_from_counts(self.lexical["user_signature"], 3)
            exemplars.append("user_signature: " + ", ".join([f"{w}×{c}" for w, c in tops]))
            
        # User rare words (technical terms, unusual words)
        if self.lexical.get("user_rare"):
            tops = self._topn_from_counts(self.lexical["user_rare"], 3)
            exemplars.append("user_rare: " + ", ".join([f"{w}×{c}" for w, c in tops]))

        # Top emojis (from phrase_counts: keys like "[emoji] 😊")
        top_emojis: list[tuple[str, int]] = []
        if self.phrase_counts:
            emoji_items = [(k[8:], v) for k, v in self.phrase_counts.items() if k.startswith("[emoji] ") and k != "[emoji] any"]
            emoji_items.sort(key=lambda x: x[1], reverse=True)
            top_emojis = emoji_items[:5]
        if top_emojis:
            exemplars.append("emojis: " + ", ".join([f"{e}×{c}" for e, c in top_emojis]))

        # Common elongations (from phrase_counts keys like "[elongation] sooo")
        el_items = [(k.split('] ',1)[1], v) for k, v in self.phrase_counts.items() if k.startswith("[elongation] ")]
        el_items.sort(key=lambda x: x[1], reverse=True)
        if el_items:
            exemplars.append("elongations: " + ", ".join([f"{w}×{c}" for w, c in el_items[:5]]))

        # Average sentence length, exclamation/question usage
        avg_len = self.structure.get("avg_sentence_len_words", 0)
        if avg_len:
            exemplars.append(f"avg_sentence_len_words: {avg_len:.1f}")
        if "exclaims_ratio" in self.structure:
            exemplars.append(f"exclaims_ratio: {self.structure['exclaims_ratio']*100:.1f}%")
        if "questions_ratio" in self.structure:
            exemplars.append(f"questions_ratio: {self.structure['questions_ratio']*100:.1f}%")

        # Compose blueprint line
        parts: list[str] = []
        if style_patterns:
            parts.append(", ".join([f"{pattern}: {pct}%" for pattern, pct in style_patterns.items()]))
        if exemplars:
            parts.append("examples → " + " | ".join(exemplars))
        if parts:
            sections.append("Style Blueprint → " + "; ".join(parts))

        return "\n".join(sections)
    
    # -----------------------
    # Detailed, per-category blueprint with dynamic exemplars
    # -----------------------
    def _topn_pct(self, counts: Dict[str, float], n: int = 5) -> list[tuple[str, float]]:
        """Return top-n items as (key, pct of category total)."""
        if not counts:
            return []
        total = float(sum(counts.values())) or 1.0
        items = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)[:n]
        return [(k, round(100.0 * float(v) / total, 2)) for k, v in items]

    def _extract_prefixed_counts(self, prefix: str) -> Dict[str, float]:
        out: Dict[str, float] = {}
        for k, v in self.phrase_counts.items():
            if k.startswith(prefix):
                key = k[len(prefix):]
                out[key] = out.get(key, 0.0) + float(v)
        return out

    def get_style_blueprint_detailed(self) -> Dict[str, object]:
        """Build a detailed, per-category blueprint with:
        - overall_pct: category strength (capped at 100)
        - top: up to 5 dynamic exemplars with within-category percentages
        - value: for scalar metrics like avg_sentence_length
        """
        total_samples = max(1.0, self.totals.get("sentences", 1.0))

        # Category: all_caps
        all_caps_ratio = float(self.casing.get("all_caps_token_ratio", 0.0)) * 100.0
        all_caps_ratio = max(0.0, min(100.0, round(all_caps_ratio, 2)))
        all_caps_top = [("ALL_CAPS", 100.0)] if all_caps_ratio > 0 else []

        # Category: elongations
        el_counts = self._extract_prefixed_counts("[elongation] ")
        el_total = float(sum(el_counts.values()))
        el_overall = round(min(100.0, (self.elongation.get("count", el_total) / total_samples) * 100.0), 2) if (self.elongation.get("count", el_total) or el_total) else 0.0
        el_top = self._topn_pct(el_counts, 5)

        # Category: emojis
        emoji_counts = self._extract_prefixed_counts("[emoji] ")
        if "any" in emoji_counts:
            emoji_counts.pop("any", None)
        emoji_total = float(sum(emoji_counts.values()))
        emoji_overall = round(min(100.0, (self.emoji.get("emoji_count", emoji_total) / total_samples) * 100.0), 2) if (self.emoji.get("emoji_count", emoji_total) or emoji_total) else 0.0
        emoji_top = self._topn_pct(emoji_counts, 5)

        # Category: repeated punctuation
        rep_keys = ("!!!", "??", "...", "!!??")
        rep_counts = {k: float(self.punctuation.get(k, 0.0)) for k in rep_keys if self.punctuation.get(k, 0.0)}
        rep_total = float(sum(rep_counts.values()))
        rep_overall = round(min(100.0, (rep_total / total_samples) * 100.0), 2) if rep_total else 0.0
        rep_top = self._topn_pct(rep_counts, 5)

        # Category: exclamations (use structure ratio; fall back to repeated !!!)
        ex_ratio = float(self.structure.get("exclaims_ratio", 0.0)) * 100.0
        ex_overall = round(min(100.0, ex_ratio), 2) if ex_ratio > 0 else (rep_overall if rep_overall > 0 else 0.0)
        ex_top = [(k, p) for k, p in rep_top if "!" in k][:5]

        # Category: fillers (blend lifetime + session)
        def _blend_counts(primary: Dict[str, float], session: Dict[str, float], w_session: float = 0.5) -> Dict[str, float]:
            out: Dict[str, float] = dict(primary or {})
            for k, v in (session or {}).items():
                out[k] = float(out.get(k, 0.0)) + float(w_session) * float(v)
            return out

        fillers_counts = _blend_counts(self.lexical.get("fillers", {}), self.session_lexical.get("fillers", {}))
        fillers_total = float(sum(fillers_counts.values()))
        fillers_overall = round(min(100.0, (fillers_total / total_samples) * 100.0), 2) if fillers_total else 0.0
        fillers_top = self._topn_pct(fillers_counts, 5)

        # Category: affective
        affect_counts = _blend_counts(self.lexical.get("affective", {}), self.session_lexical.get("affective", {}))
        affect_total = float(sum(affect_counts.values()))
        affect_overall = round(min(100.0, (affect_total / total_samples) * 100.0), 2) if affect_total else 0.0
        affect_top = self._topn_pct(affect_counts, 5)

        # Category: slang
        slang_counts = _blend_counts(self.lexical.get("slang", {}), self.session_lexical.get("slang", {}))
        slang_total = float(sum(slang_counts.values()))
        slang_overall = round(min(100.0, (slang_total / total_samples) * 100.0), 2) if slang_total else 0.0
        slang_top = self._topn_pct(slang_counts, 5)

        # Scalar: avg_sentence_length
        avg_len = float(self.structure.get("avg_sentence_len_words", 0.0))
        avg_len_val = round(avg_len, 1) if avg_len > 0 else 0.0

        # Booleans: avoids_capitals, avoids_oxford (percent of updates + examples)
        total_updates = max(1.0, self.totals.get("sentences", 1.0))
        
        # avoids_capitals: use lowercase_sentence_starts ratio as overall, create synthetic examples
        lowercase_ratio = float(self.casing.get("lowercase_sentence_start_ratio", 0.0)) * 100.0
        avoids_capitals_overall = round(min(100.0, lowercase_ratio), 2) if lowercase_ratio > 0 else 0.0
        avoids_capitals_top = [("lowercase_starts", 100.0)] if avoids_capitals_overall > 0 else []
        
        # avoids_oxford: use avoids_oxford_comma ratio, create synthetic examples  
        avoids_oxford = float(self.negatives.get("avoids_oxford_comma", 0.0))
        avoids_oxford_overall = round(min(100.0, (avoids_oxford / total_updates) * 100.0), 2) if avoids_oxford else 0.0
        avoids_oxford_top = [("no_oxford_comma", 100.0)] if avoids_oxford_overall > 0 else []

        # Category: user_frequent (novel frequent words)
        user_frequent_counts = _blend_counts(self.lexical.get("user_frequent", {}), self.session_lexical.get("user_frequent", {}))
        user_frequent_total = float(sum(user_frequent_counts.values()))
        user_frequent_overall = round(min(100.0, (user_frequent_total / total_samples) * 100.0), 2) if user_frequent_total else 0.0
        user_frequent_top = self._topn_pct(user_frequent_counts, 5)

        # Category: user_signature (distinctive/uncommon words)
        user_signature_counts = _blend_counts(self.lexical.get("user_signature", {}), self.session_lexical.get("user_signature", {}))
        user_signature_total = float(sum(user_signature_counts.values()))
        user_signature_overall = round(min(100.0, (user_signature_total / total_samples) * 100.0), 2) if user_signature_total else 0.0
        user_signature_top = self._topn_pct(user_signature_counts, 5)

        # Category: user_rare (technical/unusual words)
        user_rare_counts = _blend_counts(self.lexical.get("user_rare", {}), self.session_lexical.get("user_rare", {}))
        user_rare_total = float(sum(user_rare_counts.values()))
        user_rare_overall = round(min(100.0, (user_rare_total / total_samples) * 100.0), 2) if user_rare_total else 0.0
        user_rare_top = self._topn_pct(user_rare_counts, 5)

        return {
            "all_caps": {"overall_pct": all_caps_ratio, "top": all_caps_top},
            "elongations": {"overall_pct": el_overall, "top": el_top},
            "emojis": {"overall_pct": emoji_overall, "top": emoji_top},
            "repeated_punct": {"overall_pct": rep_overall, "top": rep_top},
            "exclamations": {"overall_pct": ex_overall, "top": ex_top},
            "fillers": {"overall_pct": fillers_overall, "top": fillers_top},
            "affective": {"overall_pct": affect_overall, "top": affect_top},
            "slang": {"overall_pct": slang_overall, "top": slang_top},
            "user_frequent": {"overall_pct": user_frequent_overall, "top": user_frequent_top},
            "user_signature": {"overall_pct": user_signature_overall, "top": user_signature_top},
            "user_rare": {"overall_pct": user_rare_overall, "top": user_rare_top},
            "avg_sentence_length": {"value": avg_len_val},
            "avoids_capitals": {"overall_pct": avoids_capitals_overall, "top": avoids_capitals_top},
            "avoids_oxford": {"overall_pct": avoids_oxford_overall, "top": avoids_oxford_top},
        }

    def format_for_llm_system_prompt_detailed(self) -> str:
        """Enhanced system prompt with per-category top-5 examples and overall % caps."""
        lines: list[str] = []

        # Emotional Palette (top 15)
        emotions = self.get_emotion_percentages()
        if emotions:
            pairs = ", ".join([f"{k}: {v}%" for k, v in emotions.items()])
            lines.append(f"Emotional Palette → {pairs}")

        # Detailed Style Blueprint
        lines.append("Style Blueprint:")
        bp = self.get_style_blueprint_detailed()

        def _pad(items: list[str], n: int = 5) -> list[str]:
            out = list(items)[:n]
            while len(out) < n:
                out.append("-")
            return out

        def _rank_examples(entry: dict, cat: str) -> list[str]:
            # Use actual frequency data from lexical dictionaries, not blueprint "top"
            primary = dict(self.lexical.get(cat, {}) or {})
            session = dict(self.session_lexical.get(cat, {}) or {})
            
            # Combine all words from both primary and session
            all_words = set(primary.keys()) | set(session.keys())
            if not all_words:
                return _pad([], 5)
            
            # Score based on actual usage frequency (lifetime + recent)
            scores: Dict[str, float] = {}
            for word in all_words:
                freq = float(primary.get(word, 0.0))
                rec = float(session.get(word, 0.0))
                scores[word] = 0.7 * freq + 0.3 * rec
            
            # Sort by total usage frequency, take top 5
            ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
            return _pad([k for k, _ in ranked[:5]], 5)

        def _get_elongation_examples() -> list[str]:
            # Get elongations from phrase_counts (most frequent elongated words)
            elongations = {k[13:]: v for k, v in self.phrase_counts.items() if k.startswith("[elongation] ")}
            if not elongations:
                return _pad([], 5)
            ranked = sorted(elongations.items(), key=lambda kv: kv[1], reverse=True)
            return _pad([k for k, _ in ranked[:5]], 5)

        def _get_emoji_examples() -> list[str]:
            # Get emojis from phrase_counts (most frequent emojis)
            emojis = {k[8:]: v for k, v in self.phrase_counts.items() if k.startswith("[emoji] ") and k != "[emoji] any"}
            if not emojis:
                return _pad([], 5)
            ranked = sorted(emojis.items(), key=lambda kv: kv[1], reverse=True)
            return _pad([k for k, _ in ranked[:5]], 5)

        # Build normalized overall percentages across categories (excluding avg_sentence_length)
        category_keys = [
            "all_caps", "elongations", "emojis", "repeated_punct", "exclamations",
            "fillers", "affective", "slang", "user_frequent", "user_signature", "user_rare",
            "avoids_capitals", "avoids_oxford"
        ]
        
        # Consider categories with any non-zero percentage regardless of example availability
        active_overalls = {}
        for k in category_keys:
            entry = bp.get(k, {})
            raw_pct = float(entry.get("overall_pct", 0.0))
            if raw_pct > 0.0:
                active_overalls[k] = raw_pct
        
        total_overall = sum(active_overalls.values()) or 1.0
        
        # Normalize only the active categories to sum to 100%
        norm_overalls = {}
        for k in category_keys:
            if k in active_overalls:
                norm_overalls[k] = round(100.0 * active_overalls[k] / total_overall, 1)
            else:
                norm_overalls[k] = 0.0

        # Emit in the requested order
        ordering = [
            "all_caps", "elongations", "emojis", "repeated_punct", "exclamations",
            "fillers", "affective", "slang", "user_frequent", "user_signature", "user_rare",
            "avg_sentence_length", "avoids_capitals", "avoids_oxford"
        ]
        for key in ordering:
            entry = bp.get(key, {}) if isinstance(bp, dict) else {}
            if key == "avg_sentence_length":
                val = entry.get("value", 0.0)
                lines.append(f"  - {key}: [{val}]")
            elif key in ["all_caps", "repeated_punct", "exclamations", "avoids_capitals", "avoids_oxford"]:
                # These categories show only percentage, no examples
                overall_pct = norm_overalls.get(key, 0.0)
                lines.append(f"  - {key}: {{{overall_pct}%}}")
            else:
                # Use actual frequency-based examples for each category
                if key == "elongations":
                    examples = _get_elongation_examples()
                elif key == "emojis":
                    examples = _get_emoji_examples()
                elif key in {"fillers", "affective", "slang", "user_frequent", "user_signature", "user_rare"}:
                    examples = _rank_examples(entry, key)
                else:
                    examples = _pad([k for k,_ in entry.get("top", [])], 5)
                overall_pct = norm_overalls.get(key, 0.0)
                lines.append(f"  - {key}: [{', '.join(examples)}] {{{overall_pct}%}}")

        # Add n-grams and emoji categories sections (derived from phrase_counts)
        # n-grams
        ngram2 = {k[9:]: v for k, v in self.phrase_counts.items() if k.startswith("[ngram2] ")}
        ngram3 = {k[9:]: v for k, v in self.phrase_counts.items() if k.startswith("[ngram3] ")}
        def _top_k(d: Dict[str, int], k: int = 5) -> list[str]:
            return [f"{w}×{c}" for w, c in sorted(d.items(), key=lambda kv: kv[1], reverse=True)[:k]]
        if ngram2 or ngram3:
            lines.append(f"  - ngrams: [{', '.join((_top_k(ngram2)+_top_k(ngram3))[:5])}]")

        # emoji categories already included in main emoji section above, skip duplicates

        return "\n".join(lines)


# =============================================================
# Formatting helpers
# =============================================================

def format_sections(
    se_profile: StyleEmotionProfile,
    max_phrases: int = 40,
    min_phrase_pct: float = 1.0,
    max_emotions: Optional[int] = None,
    min_emotion_pct: float = 1.0,
    min_phrase_count: int = 2,
    hide_low_signal: bool = False,
) -> str:
    # Section 1 (style patterns)
    s1 = se_profile.section1_percentages()
    s1_lines = ["Section 1: Style Patterns (normalized %)"]
    for label, cue, pct in s1:
        if hide_low_signal and pct < 1.0:
            continue
        s1_lines.append(f"- [pattern] {label} û {pct}%")
    s1_block = "\n".join(s1_lines)

    # Section 2 (emotions)
    s2 = se_profile.section2_percentages()
    s2_lines = ["Section 2: Emotions by Sentiment Analysis"]
    used = 0
    for emo, pct in s2:
        if pct < min_emotion_pct:
            continue
        s2_lines.append(f"- {emo} û {pct}%")
        used += 1
        if max_emotions and used >= max_emotions:
            break
    s2_block = "\n".join(s2_lines)

    # Phrase list (optional, filtered)
    phrase_lines = []
    if se_profile.phrase_counts:
        total_phr = sum(v for v in se_profile.phrase_counts.values() if v >= min_phrase_count)
        if total_phr > 0:
            phrase_lines.append("\nTop phrases:")
            for k, v in sorted(se_profile.phrase_counts.items(), key=lambda kv: kv[1], reverse=True):
                if v < min_phrase_count:
                    continue
                pct = 100.0 * v / total_phr
                if pct < min_phrase_pct:
                    continue
                phrase_lines.append(f"- {k}: {v} ({pct:.2f}%)")

    return "\n".join([s1_block, "", s2_block, "", *phrase_lines])


# =============================================================
# Dedupe + update helper
# =============================================================

def dedupe_and_add_phrase_counts(
    phrase_counts: Dict[str, int],
    embed_model: EmbeddingModel,
    store: FAISSStore,
    sim_threshold: float = 0.85,
) -> Dict[str, int]:
    """Embed phrase keys, search nearest in store, merge if similarity >= threshold.

    Returns a dict of canonical phrase -> merged count to reflect the update,
    and **also** mutates the store by adding vectors for any new phrases.
    """
    if not phrase_counts:
        return {}

    keys = list(phrase_counts.keys())
    vecs = embed_model.encode(keys)

    merged: Dict[str, int] = {}
    new_vecs: List[List[float]] = []
    new_keys: List[str] = []

    for key, vec in zip(keys, vecs):
        # search top-1
        idxs, sims = store.search(vec, k=1)
        if idxs and sims:
            i, sim = idxs[0], sims[0]
            # if above threshold, merge counts into existing key
            if i is not None and i >= 0 and sim >= sim_threshold and i < len(store.keys):
                base_key = store.keys[i]
                merged[base_key] = merged.get(base_key, 0) + int(phrase_counts[key])
                continue
        # else: treat as new canonical phrase
        merged[key] = merged.get(key, 0) + int(phrase_counts[key])
        new_vecs.append(vec)
        new_keys.append(key)

    if new_keys:
        store.add(new_vecs, new_keys)

    return merged
