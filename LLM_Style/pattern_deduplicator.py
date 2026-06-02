#!/usr/bin/env python3
"""
pattern_deduplicator.py — Advanced Pattern Deduplication Engine

Multi-layer approach for intelligent pattern clustering:
1. Exact matching (fastest)
2. Edit distance clustering (style-aware) 
3. FAISS semantic similarity (content-aware)
4. Category-specific strategies
5. Incremental learning with memory
"""

from typing import Dict, List, Tuple, Set, Optional, Any
from dataclasses import dataclass, field
import re
from collections import defaultdict
import json
import os

# Edit distance calculation
def edit_distance(s1: str, s2: str) -> int:
    """Calculate Levenshtein distance between two strings."""
    if len(s1) < len(s2):
        return edit_distance(s2, s1)
    
    if len(s2) == 0:
        return len(s1)
    
    previous_row = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    
    return previous_row[-1]

def phonetic_similarity(s1: str, s2: str) -> float:
    """Simple phonetic similarity for fillers (um, umm, ummm)."""
    # Remove repeated characters for comparison
    def normalize_repeats(s: str) -> str:
        return re.sub(r'(.)\1+', r'\1', s.lower())
    
    norm1, norm2 = normalize_repeats(s1), normalize_repeats(s2)
    if norm1 == norm2:
        # If both normalize to same base, check if one is just elongated
        if s1.lower() == s2.lower():
            return 1.0  # Exact match
        # Check if they're elongation variants of same base sound
        base_len_diff = abs(len(s1) - len(s2))
        if base_len_diff <= 2:  # Allow some elongation difference
            return 0.8
        return 0.6  # More different but same base
    
    # Check if one is a subset of the other (for um/umm cases)
    if norm1 in norm2 or norm2 in norm1:
        return 0.7
    
    return 0.0

def is_elongation_variant(s1: str, s2: str) -> bool:
    """Check if two strings are elongation variants (sooo vs soooo)."""
    # Remove repeated characters and compare
    def collapse_repeats(s: str) -> str:
        return re.sub(r'(.)\1+', r'\1', s.lower())
    
    collapsed1 = collapse_repeats(s1)
    collapsed2 = collapse_repeats(s2)
    
    return collapsed1 == collapsed2 and len(s1) != len(s2)

@dataclass
class ClusterNode:
    """Represents a cluster of similar patterns."""
    canonical: str  # The representative word for this cluster
    variants: Set[str] = field(default_factory=set)  # All variants in this cluster
    total_count: int = 0  # Sum of all variant counts
    category: str = ""  # Pattern category (elongation, filler, etc.)
    
    def add_variant(self, word: str, count: int = 1) -> None:
        """Add a variant to this cluster."""
        self.variants.add(word)
        self.total_count += count
        
    def merge_with(self, other: 'ClusterNode') -> None:
        """Merge another cluster into this one."""
        self.variants.update(other.variants)
        self.total_count += other.total_count

@dataclass 
class PatternMemory:
    """Memory system for incremental learning."""
    clusters: Dict[str, ClusterNode] = field(default_factory=dict)  # canonical -> cluster
    word_to_canonical: Dict[str, str] = field(default_factory=dict)  # word -> canonical
    category_totals: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    
    def get_canonical(self, word: str) -> Optional[str]:
        """Get canonical form for a word if it exists."""
        return self.word_to_canonical.get(word)
    
    def add_to_cluster(self, word: str, canonical: str, count: int, category: str) -> None:
        """Add word to existing cluster or create new one."""
        if canonical not in self.clusters:
            self.clusters[canonical] = ClusterNode(canonical=canonical, category=category)
        
        self.clusters[canonical].add_variant(word, count)
        self.word_to_canonical[word] = canonical
        self.category_totals[category] += count
    
    def get_category_percentages(self, category: str) -> List[Tuple[str, float, int]]:
        """Get (canonical, percentage, count) for category sorted by count."""
        category_clusters = [(canonical, cluster) for canonical, cluster in self.clusters.items() 
                           if cluster.category == category]
        
        total = sum(cluster.total_count for _, cluster in category_clusters)
        if total == 0:
            return []
        
        results = []
        for canonical, cluster in category_clusters:
            percentage = (cluster.total_count / total) * 100
            results.append((canonical, percentage, cluster.total_count))
        
        return sorted(results, key=lambda x: x[2], reverse=True)

class PatternDeduplicator:
    """Advanced multi-layer pattern deduplication engine (strict; no fallbacks)."""
    
    def __init__(self, embed_model, faiss_store):
        if embed_model is None or faiss_store is None:
            raise ValueError("embed_model and faiss_store are required in strict mode")
        self.embed_model = embed_model
        self.faiss_store = faiss_store
        self.memory = PatternMemory()
        
        # Thresholds for different strategies
        self.edit_distance_threshold = 1  # Max edit distance for clustering
        self.phonetic_threshold = 0.8     # Min similarity for phonetic matching
        self.faiss_threshold = 0.85       # Min similarity for semantic clustering
        
    def deduplicate_patterns(self, new_patterns: Dict[str, Dict[str, int]]) -> Dict[str, Dict[str, int]]:
        """
        Main deduplication function.
        Input: {'elongations': {'sooo': 2, 'soooo': 1}, 'fillers': {'umm': 3, 'ummm': 1}}
        Output: Deduplicated patterns with canonical forms
        """
        deduplicated = {}
        
        for category, word_counts in new_patterns.items():
            deduplicated[category] = self._deduplicate_category(category, word_counts)
        
        return deduplicated
    
    def _deduplicate_category(self, category: str, word_counts: Dict[str, int]) -> Dict[str, int]:
        """Deduplicate patterns within a specific category."""
        result = {}
        unprocessed = dict(word_counts)  # Copy to avoid modifying original
        
        # Layer 1: Check against existing memory (fastest)
        for word, count in list(unprocessed.items()):
            canonical = self.memory.get_canonical(word)
            if canonical:
                # Word already has a canonical form
                self.memory.add_to_cluster(word, canonical, count, category)
                if canonical in result:
                    result[canonical] += count
                else:
                    result[canonical] = self.memory.clusters[canonical].total_count
                del unprocessed[word]
        
        # Layer 2: Category-specific clustering for remaining words
        if category == "elongations":
            result.update(self._cluster_elongations(unprocessed, category))
        elif category in ["fillers", "hedges"]:
            result.update(self._cluster_fillers(unprocessed, category))
        elif category in ["signature_words", "rare_words", "frequent_words"]:
            result.update(self._cluster_semantic(unprocessed, category))
        else:
            # Default: edit distance clustering
            result.update(self._cluster_edit_distance(unprocessed, category))
        
        return result
    
    def _cluster_elongations(self, word_counts: Dict[str, int], category: str) -> Dict[str, int]:
        """Cluster elongation variants (sooo, soooo, etc.)."""
        result = {}
        processed = set()
        
        words = list(word_counts.keys())
        for i, word1 in enumerate(words):
            if word1 in processed:
                continue
                
            # Start a new cluster with word1 as canonical
            canonical = word1
            cluster_count = word_counts[word1]
            cluster_words = [word1]
            
            # Find all elongation variants
            for j, word2 in enumerate(words[i+1:], i+1):
                if word2 in processed:
                    continue
                    
                if is_elongation_variant(word1, word2):
                    cluster_count += word_counts[word2]
                    cluster_words.append(word2)
                    processed.add(word2)
                    
                    # Choose the shortest form as canonical (sooo vs soooo)
                    if len(word2) < len(canonical):
                        canonical = word2
            
            # Add cluster to memory and result
            for word in cluster_words:
                self.memory.add_to_cluster(word, canonical, word_counts[word], category)
            
            result[canonical] = cluster_count
            processed.add(word1)
        
        return result
    
    def _cluster_fillers(self, word_counts: Dict[str, int], category: str) -> Dict[str, int]:
        """Cluster filler words using phonetic similarity."""
        result = {}
        processed = set()
        
        words = list(word_counts.keys())
        for i, word1 in enumerate(words):
            if word1 in processed:
                continue
                
            canonical = word1
            cluster_count = word_counts[word1]
            cluster_words = [word1]
            
            # Find phonetically similar words
            for j, word2 in enumerate(words[i+1:], i+1):
                if word2 in processed:
                    continue
                    
                similarity = phonetic_similarity(word1, word2)
                if similarity >= self.phonetic_threshold:
                    cluster_count += word_counts[word2]
                    cluster_words.append(word2)
                    processed.add(word2)
                    
                    # Choose shorter form as canonical
                    if len(word2) < len(canonical):
                        canonical = word2
            
            # Add cluster to memory and result
            for word in cluster_words:
                self.memory.add_to_cluster(word, canonical, word_counts[word], category)
            
            result[canonical] = cluster_count
            processed.add(word1)
        
        return result
    
    def _cluster_edit_distance(self, word_counts: Dict[str, int], category: str) -> Dict[str, int]:
        """Cluster words using edit distance."""
        result = {}
        processed = set()
        
        words = list(word_counts.keys())
        for i, word1 in enumerate(words):
            if word1 in processed:
                continue
                
            canonical = word1
            cluster_count = word_counts[word1]
            cluster_words = [word1]
            
            # Find words within edit distance threshold
            for j, word2 in enumerate(words[i+1:], i+1):
                if word2 in processed:
                    continue
                    
                if edit_distance(word1, word2) <= self.edit_distance_threshold:
                    cluster_count += word_counts[word2]
                    cluster_words.append(word2)
                    processed.add(word2)
                    
                    # Choose shorter form as canonical
                    if len(word2) < len(canonical):
                        canonical = word2
            
            # Add cluster to memory and result
            for word in cluster_words:
                self.memory.add_to_cluster(word, canonical, word_counts[word], category)
            
            result[canonical] = cluster_count
            processed.add(word1)
        
        return result
    
    def _cluster_semantic(self, word_counts: Dict[str, int], category: str) -> Dict[str, int]:
        """Cluster words using FAISS semantic similarity (required)."""
        result: Dict[str, int] = {}
        if not word_counts:
            return result
        words = list(word_counts.keys())
        vecs = self.embed_model.encode(words)
        # For each word, attempt merge with nearest existing key; else add as new
        for word, vec in zip(words, vecs):
            # search nearest
            idxs, sims = self.faiss_store.search(vec, k=1)
            if idxs and sims:
                i, sim = idxs[0], sims[0]
                can_merge = i is not None and i >= 0 and i < len(self.faiss_store.keys) and sim is not None and sim >= 0.85
                if can_merge:
                    base_key = self.faiss_store.keys[i]
                    canonical = self.memory.get_canonical(base_key) or base_key
                    self.memory.add_to_cluster(word, canonical, word_counts[word], category)
                    result[canonical] = result.get(canonical, 0) + word_counts[word]
                    continue
            # treat as new canonical
            canonical = word
            self.memory.add_to_cluster(word, canonical, word_counts[word], category)
            result[canonical] = result.get(canonical, 0) + word_counts[word]
            # add vector for this new canonical
            self.faiss_store.add([vec], [canonical])
        return result
    
    def get_formatted_percentages(self, categories: List[str] = None) -> Dict[str, str]:
        """Get formatted percentage strings for display."""
        if categories is None:
            categories = list(set(cluster.category for cluster in self.memory.clusters.values()))
        
        results = {}
        
        for category in categories:
            percentages = self.memory.get_category_percentages(category)
            if not percentages:
                continue
                
            # Format: 'word'(percentage%, count×)
            formatted_items = []
            for canonical, percentage, count in percentages[:10]:  # Top 10
                formatted_items.append(f"'{canonical}'({percentage:.1f}%, {count}×)")
            
            results[category] = ", ".join(formatted_items)
        
        return results
    
    def save_memory(self, filepath: str) -> None:
        """Save pattern memory to file."""
        data = {
            'clusters': {
                canonical: {
                    'canonical': cluster.canonical,
                    'variants': list(cluster.variants),
                    'total_count': cluster.total_count,
                    'category': cluster.category
                }
                for canonical, cluster in self.memory.clusters.items()
            },
            'word_to_canonical': dict(self.memory.word_to_canonical),
            'category_totals': dict(self.memory.category_totals)
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    
    def load_memory(self, filepath: str) -> None:
        """Load pattern memory from file."""
        if not os.path.exists(filepath):
            return
            
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Reconstruct clusters
        self.memory.clusters = {}
        for canonical, cluster_data in data.get('clusters', {}).items():
            cluster = ClusterNode(
                canonical=cluster_data['canonical'],
                category=cluster_data['category']
            )
            cluster.variants = set(cluster_data['variants'])
            cluster.total_count = cluster_data['total_count']
            self.memory.clusters[canonical] = cluster
        
        self.memory.word_to_canonical = data.get('word_to_canonical', {})
        self.memory.category_totals = defaultdict(int, data.get('category_totals', {}))
