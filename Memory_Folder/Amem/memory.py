from __future__ import annotations

import uuid
from typing import Any, Dict, List, Optional, Tuple

from .local_config import OfflineMemoryConfig
from .local_embedding import LocalEmbedding
# LocalLLM imported conditionally to avoid heavy dependencies when using ServerLLM
from .vector_store import LocalVectorStore
from .graph_store import GraphStore, Relation
from .entity_index import EntityIndex, ENTITY_PATTERN
from .utils import md5_hash, utc_now_iso, merge_texts
from .node_vectors import NodeVectorIndex
from .storage import SQLiteManager


class OfflineMemory:
    """Rule-based offline memory system with hybrid retrieval and graph reasoning.

    Features:
    - Deterministic parsing of entities (no LLM prompts) and graph building
    - FAISS vector search for semantic recall
    - Graph k-hop expansion for indirect retrieval and re-scoring
    - Merge/update logic using similarity thresholds
    - SQLite history tracking
    """

    def __init__(self, config: Optional[OfflineMemoryConfig] = None):
        self.config = config or OfflineMemoryConfig()
        self.embedder = LocalEmbedding(self.config.embedder)
        self.vectors = LocalVectorStore(self.config.vector_store)
        self.graph = GraphStore(self.config.graph)
        self.entities = EntityIndex(self.config.entity_index)
        # Import LocalLLM only when needed to avoid heavy dependencies
        llm_backend = getattr(self.config.llm, "backend", None)
        if llm_backend is None or str(llm_backend).lower() == "none":
            # Defer LLM creation until a backend is explicitly selected
            self.llm = None
        else:
            from .local_llm import LocalLLM
            self.llm = LocalLLM(self.config.llm)
        self.db = SQLiteManager(self.config.history_db_path)
        self.node_vectors = NodeVectorIndex(self.config.node_vectors)
        # Caches for traversal speed-ups
        self._neighbor_cache: Dict[str, List[Relation]] = {}
        self._degree_cache: Dict[str, int] = {}

    def _clear_traverse_caches(self) -> None:
        self._neighbor_cache.clear()
        self._degree_cache.clear()

    # ---------- profile management for name personalization ----------
    def _get_profile_path(self) -> str:
        """Get path to user profile file."""
        import os
        base_dir = os.path.dirname(self.config.vector_store.path)
        return os.path.join(base_dir, "user_profile.json")

    def _load_profile(self) -> Dict[str, Any]:
        """Load user/assistant profile from disk."""
        import json
        import os
        path = self._get_profile_path()
        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                pass
        return {"user_name": None, "user_age": None, "user_location": None, "assistant_name": "ARAISHA"}

    def _save_profile(self, profile: Dict[str, Any]) -> None:
        """Save profile to disk."""
        import json
        import os
        path = self._get_profile_path()
        os.makedirs(os.path.dirname(path), exist_ok=True)
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(profile, f, indent=2)
        except Exception:
            pass

    def _get_user_name(self) -> Optional[str]:
        """Get persisted user name."""
        return self._load_profile().get("user_name")

    def _get_assistant_name(self) -> str:
        """Get persisted assistant name (default: ARAISHA)."""
        return self._load_profile().get("assistant_name") or "ARAISHA"

    def _update_profile(self, **kwargs) -> None:
        """Update specific profile fields."""
        profile = self._load_profile()
        for key, value in kwargs.items():
            if value is not None:
                profile[key] = value
        self._save_profile(profile)

    def _detect_and_store_names(self, text: str) -> None:
        """Detect user/assistant name declarations and store them."""
        import re
        lower = text.lower()
        
        # Detect user name: "my name is X" or "I am X" or "call me X"
        user_name_patterns = [
            r"my name is\s+([a-zA-Z]+)",
            r"i am\s+([a-zA-Z]+)",
            r"call me\s+([a-zA-Z]+)",
            r"i'm\s+([a-zA-Z]+)",
        ]
        for pattern in user_name_patterns:
            match = re.search(pattern, lower)
            if match:
                name = match.group(1).capitalize()
                # Avoid common words that aren't names
                if name.lower() not in {"the", "a", "an", "very", "really", "just", "so", "here", "there", "fine", "good", "okay", "happy", "sad"}:
                    self._update_profile(user_name=name.upper())
                    break
        
        # Detect assistant name: "your name is X" or "call you X" or "your name will be X"
        assistant_name_patterns = [
            r"your name (?:is|will be)\s+([a-zA-Z]+)",
            r"call you\s+([a-zA-Z]+)",
            r"i(?:'ll| will) call you\s+([a-zA-Z]+)",
        ]
        for pattern in assistant_name_patterns:
            match = re.search(pattern, lower)
            if match:
                name = match.group(1).capitalize()
                if name.lower() not in {"the", "a", "an", "very", "really", "just", "so"}:
                    self._update_profile(assistant_name=name.upper())
                    break
        
        # Detect user age: "I am X years old" or "my age is X"
        age_patterns = [
            r"i am\s+(\d{1,3})\s+years? old",
            r"i'm\s+(\d{1,3})\s+years? old",
            r"my age is\s+(\d{1,3})",
        ]
        for pattern in age_patterns:
            match = re.search(pattern, lower)
            if match:
                age = int(match.group(1))
                if 1 <= age <= 150:
                    self._update_profile(user_age=age)
                    break
        
        # Detect user location: "I live in X" or "I'm from X"
        location_patterns = [
            r"i live in\s+([a-zA-Z\s]+?)(?:\.|,|$)",
            r"i'm from\s+([a-zA-Z\s]+?)(?:\.|,|$)",
            r"i am from\s+([a-zA-Z\s]+?)(?:\.|,|$)",
        ]
        for pattern in location_patterns:
            match = re.search(pattern, lower)
            if match:
                location = match.group(1).strip().title()
                if len(location) >= 2:
                    self._update_profile(user_location=location.upper())
                    break


    def _cache_neighbors(self, node: str, base_limit: int = 200) -> List[Relation]:
        try:
            if node in self._neighbor_cache:
                return self._neighbor_cache[node]
            rels = self.graph.neighbors(node, direction="both", limit=base_limit)
            self._neighbor_cache[node] = rels
            self._degree_cache[node] = len(rels)
            return rels
        except Exception:
            return []

    # ---------- storage helpers ----------
    def _make_payload(self, memory_id: str, text: str, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        now = utc_now_iso()
        meta = metadata.copy() if metadata else {}
        meta.setdefault("created_at", now)
        meta["updated_at"] = now
        return {
            "id": memory_id,
            "memory": text,
            "hash": md5_hash(text),
            "metadata": meta,
        }

    def _index_entities_and_graph(self, memory_id: str, text: str, metadata: Optional[Dict[str, Any]], messages: Optional[List[Dict[str, str]]] = None):
        # STEP 0: Detect and store user/assistant names from the text
        self._detect_and_store_names(text)
        
        # Entities
        additional_entities = []
        # Heuristic for self-naming: "your name will be X" or "call you X"
        lower = text.lower()
        if "your name" in lower and (" will be " in lower or " is " in lower):
            try:
                after = text.split("your name", 1)[1]
                for tok in ["will be", "is"]:
                    if tok in after.lower():
                        name = after.lower().split(tok, 1)[1].strip().split(" ")[0]
                        if name:
                            additional_entities.append(name.capitalize())
                            break
            except Exception:
                pass
        if "call you" in lower:
            try:
                name = lower.split("call you", 1)[1].strip().split(" ")[0]
                if name:
                    additional_entities.append(name.capitalize())
            except Exception:
                pass
        self.entities.add_memory_entities(memory_id, text, additional_entities)
        # Dynamic relationship extraction using LLM once during store; normal chat unaffected
        now_ts = utc_now_iso()
        import re as _re
        
        # Simple entity extraction for entity index (fallback for non-relationship entities)
        capitalized_entities = _re.findall(r"\b([A-Z][a-zA-Z0-9_\-]+)\b", text)
        entities = list(set(e.capitalize() for e in capitalized_entities if e))

        # Extract multiple triplets (src, relation, dst) using LLM
        # CRITICAL: Build graph context FIRST before any LLM calls
        # Hold the user message until graph context is fully built
        extracted_triplets = []
        
        try:
            # Prepare context for LLM
            ctx = messages[-2:] if messages and len(messages) >= 2 else []
            ctx = ctx + [{"role": "user", "content": text}]
            
            if getattr(self.config.retrieval, "debug", False):
                print(f"[timing_debug] Starting graph context building - LLM call BLOCKED until ready")
            
            # STEP 1: Build graph context BEFORE calling LLM
            # This ensures the model has knowledge of existing entities when extracting relationships
            def _norm_rel_for_dedup(r: str) -> str:
                r = (r or "").upper().replace("-", "_")
                tokens = [t for t in r.split("_") if t not in ("OF", "IS", "ARE", "WAS", "WERE")]
                out = "_".join(tokens)
                if len(out) > 3 and out.endswith("S"):
                    out = out[:-1]
                return out

            def _trip_key(src: str, rel: str, dst: str) -> tuple:
                return ((src or "").upper(), _norm_rel_for_dedup(rel or ""), (dst or "").upper())

            # STEP 2: Collect ONLY related entities from the graph (not all entities)
            related_entities = set(entities)  # Start with entities found in text
            neighbor_edges: list[tuple[str, str, str]] = []
            seen_keys: set[tuple] = set()
            
            # Debug: Show extracted entities
            if getattr(self.config.retrieval, "debug", False):
                print(f"[graph_context_debug] Extracted entities from text: {entities}")
            
            # STEP 2.1: Enhanced entity detection for possessive patterns
            # Look for patterns like "X's friend" and try to resolve actual friend names
            possessive_patterns = _re.findall(r"(\w+)'s\s+(\w+)", text.lower())
            if possessive_patterns:
                if getattr(self.config.retrieval, "debug", False):
                    print(f"[graph_context_debug] Found possessive patterns: {possessive_patterns}")
                
                # Try to find if the possessed entity exists in our graph
                for owner, possessed in possessive_patterns:
                    owner_cap = owner.capitalize()
                    # Look for existing relationships that might identify the actual person
                    try:
                        owner_neighbors = list(self.graph.neighbors(owner_cap, direction="both", limit=50))
                        if not owner_neighbors:
                            # Try case variants
                            for case_variant in [owner.upper(), owner.lower()]:
                                owner_neighbors.extend(list(self.graph.neighbors(case_variant, direction="both", limit=50)))
                        
                        # Look for friend relationships
                        friend_entities = []
                        for rel in owner_neighbors:
                            if "FRIEND" in rel.relationship or "BUDDY" in rel.relationship or "PAL" in rel.relationship:
                                friend_entities.append(rel.destination if rel.source.upper() == owner_cap.upper() else rel.source)
                        
                        if friend_entities:
                            entities.extend(friend_entities)
                            if getattr(self.config.retrieval, "debug", False):
                                print(f"[graph_context_debug] Resolved {owner}'s friends: {friend_entities}")
                    except:
                        pass
            
            # Find neighbors for entities mentioned in current text
            for ent in entities[:5]:  # Check up to 5 entities from text
                try:
                    # Try both direct neighbors and k-hop traversal with case variations
                    neighbors = list(self.graph.neighbors(ent, direction="both", limit=15))
                    
                    # Try case variations if no neighbors found
                    if len(neighbors) == 0:
                        for case_variant in [ent.upper(), ent.lower(), ent.capitalize()]:
                            if case_variant != ent:
                                alt_neighbors = list(self.graph.neighbors(case_variant, direction="both", limit=15))
                                if alt_neighbors:
                                    neighbors.extend(alt_neighbors)
                                    if getattr(self.config.retrieval, "debug", False):
                                        print(f"[graph_context_debug] Found {len(alt_neighbors)} neighbors for '{case_variant}' (variant of '{ent}')")
                    
                    # Add related entities to our set
                    for r in neighbors:
                        related_entities.add(r.source)
                        related_entities.add(r.destination)
                        
                        key = _trip_key(r.source, r.relationship, r.destination)
                        if key not in seen_keys:
                            seen_keys.add(key)
                            neighbor_edges.append((r.source.upper(), r.relationship.upper(), r.destination.upper()))
                            
                    if getattr(self.config.retrieval, "debug", False):
                        print(f"[graph_context_debug] Entity '{ent}' contributed {len(neighbors)} relationships")
                        
                except Exception as e:
                    if getattr(self.config.retrieval, "debug", False):
                        print(f"[graph_context_debug] Error getting neighbors for '{ent}': {e}")
                    continue

            # STEP 3: Filter additional edges to only include related entities
            related_edges: list[tuple[str, str, str]] = []
            try:
                all_edges = self.graph.query_edges(limit=1000)
                for e in all_edges:
                    # Only include edges that involve entities related to current text
                    if e.source in related_entities or e.destination in related_entities:
                        key = _trip_key(e.source, e.relationship, e.destination)
                        if key not in seen_keys:
                            seen_keys.add(key)
                            related_edges.append((e.source.upper(), e.relationship.upper(), e.destination.upper()))
            except Exception:
                pass

            # STEP 4: Assemble FOCUSED graph context (only related entities)
            max_total = 12  # Focused context
            # Prioritize direct neighbors of mentioned entities
            chosen_neighbors = neighbor_edges[:min(8, len(neighbor_edges))]
            remaining_slots = max_total - len(chosen_neighbors)
            chosen_related = related_edges[:remaining_slots] if remaining_slots > 0 else []
            
            graph_lines = [f"{s} --{r}-- {d}" for (s, r, d) in (chosen_neighbors + chosen_related)]

            # Use focused examples relevant to the relationships found
            if graph_lines:
                examples = "RELATED EXAMPLES:\n" + "\n".join(graph_lines[:5]) + "\n"
            else:
                examples = (
                "EXAMPLES:\n"
                "JOHN - WORKS_AT - MICROSOFT\nJOHN - LOVES - CODING\n\n"
                "ALICE - LIKES - BOB\nBOB - LOVES - CAROL\nCAROL - MARRIED_TO - DAVID\n\n"
                "ABHISHEK - LOVES - ZANI\nZANI - LOVES - VEYRION\nVEYRON - LIKES - ARIA\nARIA - WIFE_OF - ABHISHEK\n"
                )

            graph_context_block = "EXISTING RELATIONSHIPS:\n" + ("\n".join(graph_lines) if graph_lines else "(none)")

            # Debug: Show focused graph context
            if getattr(self.config.retrieval, "debug", False):
                print(f"[graph_context_debug] Related entities found: {len(related_entities)}")
                print(f"[graph_context_debug] Focused graph_lines count: {len(graph_lines)}")
                print(f"[graph_context_debug] Graph context:\n{graph_context_block}")

            # STEP 5: NOW build system prompt with proper context and call LLM
            sys_prompt = (
                """(<think>Detailed Thinking: OFF </think> You are a GRAPH MEMORY RELATIONSHIP EXTRACTOR.

Your task is to extract ONLY long-term, relevant, user-memory-worthy relationships
from input text and represent them as graph edges.

Most inputs may be irrelevant, noisy, emotional, exaggerated, temporary, or meaningless.
You must decide whether a memory deserves to be stored.

────────────────────────────────────────
OUTPUT FORMAT (MANDATORY)
────────────────────────────────────────

Output ONLY one of the following:
1) Valid relationship lines
2) A single full stop:
.

Each valid relationship must follow EXACTLY one format:

Either: SOURCE -- RELATION -- DESTINATION
Or: DESTINATION -- RELATION -- SOURCE

Rules:
- UPPERCASE ONLY
- One relationship per line (If there are multiple relationships, print them on separate lines)
- No explanations
- No commentary
- No extra text

────────────────────────────────────────
PRONOUN REPLACEMENT (STRICT)
────────────────────────────────────────

Replace:
- I / MY / ME   → """ + (self._get_user_name() or "USER") + """
- YOU / YOUR   → """ + (self._get_assistant_name() or "ASSISTANT") + """

Pronouns must NEVER appear in the output.

────────────────────────────────────────
ENTITY RULES
────────────────────────────────────────

SOURCE and DESTINATION must be:
- A person, place, object, concept, or attribute
- OR a verb / noun / adverb phrase that represents a STABLE memory node

Rules:
- Use EXACT spelling for entities already present in EXISTING RELATIONSHIPS
- If an entity already exists:
  - DO NOT recreate it and print (.)
  - ONLY connect or modify relationships which are partially present in EXISTING RELATIONSHIPS.

────────────────────────────────────────
RELATION RULES
────────────────────────────────────────

RELATION must be a clear, meaningful connector verb.

Examples:
LIKES, USES, STUDIES, WORKS_AT, OWNS, BUILT, CREATED, PREFERS

NEVER use:
- Pronouns or auxiliaries (I, YOU, MY, IS, ARE, OF)
- Vague relations (LIKES_THINGS, ENJOYS_WORK, INTERESTED_IN_STUFF)

Relations must be SPECIFIC and CONCRETE.

────────────────────────────────────────
POSSESSIVE & SPOUSE NORMALIZATION
────────────────────────────────────────

Directionality is STRICT.

If X is the wife of Y:
X -- WIFE_OF -- Y

If Y is the husband of X:
Y -- HUSBAND_OF -- X

If marriage is stated without gender roles:
X -- MARRIED_TO -- Y

If a possessive refers to a person but the name is unknown:
- Connect the memory  with user.

If a real name can be inferred from context:
- ALWAYS prefer the real name

────────────────────────────────────────
IRRELEVANT INPUT BEHAVIOR (MANDATORY)
────────────────────────────────────────

If NO relationship qualifies for memory storage, output:

. (FULL STOP)

This applies to:
- Empty input
- Pure chatter
- Non-personal facts
- Irrelevant or unstable information
- Vague or general statements

────────────────────────────────────────
GRAPH CONSISTENCY RULES
────────────────────────────────────────

- Prefer linking to EXISTING graph entities
- Avoid duplicates and reversed duplicates
- Respect relationship direction by Gender and logic.
- Output a MAXIMUM of 6 relationships if multiple present.

────────────────────────────────────────
CONTEXT
────────────────────────────────────────

You may receive:
- This system prompt
- Existing stored relationships (graph state)


DO NOT OUTPUT ANYTHING EXCEPT RELATIONSHIPS OR A FULL STOP.
"""
                + examples + "\n" + "These are stored relationships in the graph:" + graph_context_block + f"\n\n USER_INPUT: {text}\n YOUR_OUTPUT:"
            )
            
            # STEP 6: Call LLM with prepared context
            if getattr(self.config.retrieval, "debug", False):
                print(f"[timing_debug] Graph context building COMPLETE - calling LLM")
                print(f"[graph_context_debug] Calling LLM with {len(graph_lines)} related relationships in context")
            
            # Build LLM messages
            llm_msgs = [{"role": "system", "content": sys_prompt}]
            if ctx:
                llm_msgs.extend(ctx)
            else:
                llm_msgs.append({"role": "user", "content": text})
                
            # No max_new_tokens limit to support reasoning models that need more tokens
            raw_response = self.llm.generate_messages(llm_msgs, temperature=0.1) or ""
            
            # Handle reasoning models: strip <think>...</think> or <thinking>...</thinking> blocks
            # These models output their reasoning before the actual relationship output
            thinking_pattern = _re.compile(r'<(?:think|thinking|THINK|THINKING)>.*?</(?:think|thinking|THINK|THINKING)>', _re.DOTALL | _re.IGNORECASE)
            raw_response = thinking_pattern.sub('', raw_response).strip()
            
            # Parse the response to extract multiple triplets (accept '-' or '--' separators)
            lines = raw_response.strip().split('\n')
            for line in lines:
                raw_line = line.strip()
                # Skip empty lines and boilerplate
                if not raw_line or raw_line.startswith("Analysis:") or raw_line.startswith("Input:") or raw_line.startswith("Output:"):
                    continue
                # Remove common special tokens if present
                raw_line = raw_line.replace('<|eot_id|>', '').replace('<|assistant|>', '').replace('<|user|>', '')

                # Flexible separator: one or two hyphens/en-dash/em-dash
                m_sep = _re.match(r"^\s*([A-Za-z0-9_\s]+)\s*[-–—]{1,2}\s*([A-Za-z0-9_\s]+)\s*[-–—]{1,2}\s*([A-Za-z0-9_\s]+)\s*$", raw_line)
                if m_sep:
                    src, rel_label, dst = m_sep.group(1).strip(), m_sep.group(2).strip(), m_sep.group(3).strip()
                else:
                    if " - " not in raw_line:
                        continue
                    parts = raw_line.split(" - ")
                    if len(parts) < 3:
                        continue
                    src, rel_label, dst = parts[0].strip(), parts[1].strip(), " - ".join(parts[2:]).strip()

                # Clean and validate while preserving names better
                src = _re.sub(r"[^a-zA-Z0-9_\s]+", "", src).strip()
                rel_label = _re.sub(r"[^a-zA-Z0-9_]+", "", rel_label.replace(" ", "_"))
                dst = _re.sub(r"[^a-zA-Z0-9_\s]+", "", dst).strip()

                # Convert to uppercase
                src = src.upper()
                rel_label = rel_label.upper()
                dst = dst.upper()

                # Canonicalize/validate relation (dynamic; no predefined list)
                def _canon_rel(r: str) -> Optional[str]:
                    raw = r.upper()
                    norm = raw.replace("-", "_")
                    base = norm.rstrip('S') if len(norm) > 3 else norm
                    # quick synonym map
                    syn = {
                        "WIFE": "WIFE_OF", "WIFEOF": "WIFE_OF", "WIVES": "WIFE_OF",
                        "HUSBAND": "HUSBAND_OF", "HUSBANDOF": "HUSBAND_OF",
                        "MARRIED": "MARRIED_TO", "MARRIEDTO": "MARRIED_TO",
                        "LIKE": "LIKES", "LOVE": "LOVES", "HATE": "HATES",
                        "WORKSAT": "WORKS_AT", "WORKS_AT": "WORKS_AT", "WORKSFOR": "WORKS_FOR",
                        "FAVOURITE": "FAVORITE", "FAVORITES": "FAVORITE",
                        "LIVEIN": "LIVES_IN", "LIVESIN": "LIVES_IN",
                    }
                    cand = syn[base] if base in syn else base
                    # strip leftover underscores
                    cand = cand.replace("__", "_")
                    # Deny only pronouns and articles, but allow "IS" for valid relationships like "IS_A", "IS_TYPE"
                    deny = {"I","YOU","MY","YOUR","A","AN","THE"}
                    if cand in deny or len(cand) < 2:  # Allow 2-char relations like "IS"
                        return None
                    return cand

                canon_rel = _canon_rel(rel_label)
                if not canon_rel:
                    continue

                # Ensure all parts are valid and not too long and limit total lines
                if src and dst and len(src) <= 50 and len(canon_rel) <= 30 and len(dst) <= 50:
                    extracted_triplets.append((src, canon_rel, dst))

                if len(extracted_triplets) >= 6:
                    break
                        
        except Exception as e:
            # Log the actual error instead of silently failing
            print(f"[relationship_extraction_error] {e}")
            if getattr(self.config.retrieval, "debug", False):
                print(f"[timing_debug] Graph context ready status: {graph_context_ready}")
                import traceback
                traceback.print_exc()
            extracted_triplets = []

        # Debug output for relationship extraction
        if getattr(self.config.retrieval, "debug", False):
            try:
                if extracted_triplets:
                    print(f"[relationships] Extracted {len(extracted_triplets)} relationships:")
                    for src, rel, dst in extracted_triplets:
                        print(f"[relationships] {src} --{rel}--> {dst}")
                else:
                    print("[extractor_debug] no triplets parsed; raw LLM output:", repr(raw_response))
            except Exception:
                pass

        # Deduplicate extracted triplets before persisting
        unique_triplets: List[Tuple[str, str, str]] = []
        seen_trips: set[Tuple[str, str, str]] = set()
        for src, rel_label, dst in extracted_triplets:
            key = (src.upper(), rel_label.upper(), dst.upper())
            if key in seen_trips:
                continue
            seen_trips.add(key)
            unique_triplets.append((src, rel_label, dst))

        # Store all unique triplets (batch embed path fragments for speed)
        stored_relationships = []
        all_entities_from_triplets = []
        path_frags: List[Tuple[str, str, str]] = []

        # Debug: Show what relationships are being stored
        if unique_triplets and getattr(self.config.retrieval, "debug", False):
            print(f"[relationship_storage] Storing {len(unique_triplets)} relationships: {unique_triplets}")

        for src, rel_label, dst in unique_triplets:
            rel_upper = rel_label.upper()
            try:
                self.graph.upsert(Relation(source=src, relationship=rel_upper, destination=dst, created_at=now_ts, updated_at=now_ts, memory_id=memory_id))
                stored_relationships.append(rel_upper)
                all_entities_from_triplets.extend([src, dst])
                path_frags.append((src, rel_upper, dst))
                if getattr(self.config.retrieval, "debug", False):
                    print(f"[relationship_stored] {src} --{rel_upper}--> {dst}")
            except Exception as e:
                print(f"[relationship_storage_error] Failed to store {src} --{rel_upper}--> {dst}: {e}")
                continue

            # attach minimal props for auditing (no embedding here)
            try:
                self.graph.set_edge_prop(src, rel_upper, dst, "memory_id", memory_id)
                # Temporal extraction: detect simple time mentions and attach window props
                import re as _tre
                low = text.lower()
                time_props: Dict[str, str] = {}
                if any(t in low for t in ["yesterday", "last night"]):
                    time_props["time_window"] = "YESTERDAY"
                elif any(t in low for t in ["today", "this morning", "this evening", "tonight"]):
                    time_props["time_window"] = "TODAY"
                elif "last week" in low:
                    time_props["time_window"] = "LAST_WEEK"
                elif "last month" in low:
                    time_props["time_window"] = "LAST_MONTH"
                m = _tre.search(r"(20\d{2}-\d{2}-\d{2})", text)
                if m:
                    time_props["date"] = m.group(1)
                for k, v in time_props.items():
                    self.graph.set_edge_prop(src, rel_upper, dst, k, v)
            except Exception:
                pass
        # Name assignment edge: USER --calls_assistant--> Name
        if ("your name" in lower and (" will be " in lower or " is " in lower)) or ("call you" in lower):
            # resolve user id and assistant name
            user = (metadata or {}).get("user_id") or "USER_ID"
            name = additional_entities[0] if additional_entities else "Assistant"
            now = utc_now_iso()
            self.graph.upsert(Relation(source=user, relationship="CALLS_ASSISTANT", destination=name, created_at=now, updated_at=now, memory_id=memory_id))
            stored_relationships.append("CALLS_ASSISTANT")

        # Promote nodes and clusters from ONLY relationship-extracted entities (NOT raw text entities)
        # IMPORTANT: Only create nodes for entities that appear in LLM-extracted relationships
        # Do NOT use self.entities.entities_for_memory() - those are raw text entities that pollute the graph
        try:
            import json as _json
            now = utc_now_iso()
            # Only create nodes for entities from LLM-extracted triplets
            # Deduplicate to avoid multiple node creations
            unique_triplet_entities = list(set(all_entities_from_triplets))
            for e in unique_triplet_entities:
                nid = f"ent::{e}"
                self.graph.upsert_node(node_id=nid, name=e, label="Entity", props_json=_json.dumps({}), created_at=now, updated_at=now)
            # Batch-embed node names for vector-native node retrieval
            try:
                if unique_triplet_entities:
                    node_vecs = self.embedder.embed_many(unique_triplet_entities)
                    for e, vec in zip(unique_triplet_entities, node_vecs):
                        nid = f"ent::{e}"
                        self.node_vectors.upsert(nid, vec, {"type": "entity", "name": e})
            except Exception:
                pass
            # Set edge node ids when edge connects user or known entity names
            rels = self.graph.k_hop(user_id, k=1, limit_per_hop=50) if user_id else []
            for r in rels:
                sid = user_id if r.source == user_id else (f"ent::{r.source}" if r.source in all_graph_entities else None)
                did = user_id if r.destination == user_id else (f"ent::{r.destination}" if r.destination in all_graph_entities else None)
                if sid or did:
                    self.graph.set_edge_node_ids(r.source, r.destination, sid, did)
            self.graph.recompute_clusters()

            # After nodes exist, batch-embed path fragments for this memory
            try:
                if path_frags:
                    path_texts = [f"{s} {r} {d}" for (s, r, d) in path_frags]
                    pvecs = self.embedder.embed_many(path_texts)
                    from uuid import uuid4 as _uuid4
                    for (src, rel_upper, dst), pvec in zip(path_frags, pvecs):
                        pid = f"path::{str(_uuid4())}"
                        self.node_vectors.upsert(pid, pvec, {"type": "path", "src": src, "rel": rel_upper, "dst": dst, "memory_id": memory_id})
            except Exception:
                pass
            
            # Perform similarity-based merging after initial storage
            all_entities = entities + all_entities_from_triplets
            all_entities = [e for e in all_entities if e]  # Remove None values
            all_entities = list(set(all_entities))  # Deduplicate
            
            if all_entities or stored_relationships:
                merge_results = self._merge_similar_entities_and_relationships(
                    memory_id, all_entities, stored_relationships
                )
                # Store merge results in metadata for transparency
                if merge_results["total_entity_merges"] > 0 or merge_results["total_relationship_merges"] > 0:
                    metadata = metadata or {}
                    metadata["merge_results"] = merge_results
                    
        except Exception:
            pass

        # Graph changed → clear traversal caches
        try:
            self._clear_traverse_caches()
        except Exception:
            pass

    def _merge_similar_entities_and_relationships(self, memory_id: str, entities: List[str], relationships: List[str]) -> Dict[str, Any]:
        """Check for similar entities and relationships, merge if similarity is above threshold."""
        merge_results = {
            "entities_merged": [],
            "relationships_merged": [],
            "total_entity_merges": 0,
            "total_relationship_merges": 0
        }
        
        if not self.config.entity_merging.enable_entity_merging and not self.config.entity_merging.enable_relationship_merging:
            return merge_results
        
        try:
            # 1. Entity merging
            if self.config.entity_merging.enable_entity_merging:
                for entity in entities:
                    similar_entities = self.graph.find_similar_entities(
                        entity, 
                        self.config.entity_merging.entity_similarity_threshold,
                        self.config.entity_merging.max_merge_candidates
                    )
                    
                    for similar_entity, score in similar_entities:
                        # Only merge if the score is very high to avoid false positives
                        if score >= self.config.entity_merging.entity_similarity_threshold:
                            # Merge the current entity into the existing similar one
                            # (Keep the existing entity as the target to preserve history)
                            merge_stats = self.graph.merge_entities(entity, similar_entity)
                            
                            if merge_stats["nodes_merged"] > 0 or merge_stats["edges_updated"] > 0:
                                merge_results["entities_merged"].append({
                                    "source": entity,
                                    "target": similar_entity,
                                    "score": score,
                                    "stats": merge_stats
                                })
                                merge_results["total_entity_merges"] += 1
                                
                                # Update node vectors - remove old, keep existing
                                old_node_id = f"ent::{entity}"
                                existing_node_id = f"ent::{similar_entity}"
                                if self.node_vectors.get(old_node_id):
                                    self.node_vectors.delete(old_node_id)
                                # Keep the existing entity's vector
                                
                                # Update any FAISS memories that reference the merged entity
                                self._update_faiss_entity_references(entity, similar_entity)
                            
                            break  # Only merge with the first (highest score) match
            
            # 2. Relationship merging
            if self.config.entity_merging.enable_relationship_merging:
                for relationship in relationships:
                    similar_rels = self.graph.find_similar_relationships(
                        relationship,
                        self.config.entity_merging.relationship_similarity_threshold,
                        self.config.entity_merging.max_merge_candidates
                    )
                    
                    for similar_rel, score in similar_rels:
                        if score >= self.config.entity_merging.relationship_similarity_threshold:
                            # Merge current relationship into existing similar one
                            merge_stats = self.graph.merge_relationships(relationship, similar_rel)
                            
                            if merge_stats["relationships_merged"] > 0:
                                merge_results["relationships_merged"].append({
                                    "source": relationship,
                                    "target": similar_rel,
                                    "score": score,
                                    "stats": merge_stats
                                })
                                merge_results["total_relationship_merges"] += 1
                            
                            break  # Only merge with the first match
                            
        except Exception as e:
            # Log error but don't fail the entire memory storage process
            print(f"[merge_warning] Similarity merging failed: {e}")
        
        return merge_results

    def _update_faiss_entity_references(self, old_entity: str, new_entity: str):
        """Update FAISS vector payloads when an entity is merged - optimized version."""
        try:
            # Use entity index to efficiently find memories containing the old entity
            memory_ids = self.entities.memories_for_entity(old_entity)
            
            # Only update memories that actually contain this entity
            for memory_id in memory_ids:
                record = self.vectors.get(memory_id)
                if record and record.payload:
                    memory_text = record.payload.get("memory", "")
                    
                    # Update entity references in the text
                    updated_text = memory_text
                    # Replace all case variations efficiently
                    case_variants = [
                        (old_entity, new_entity),
                        (old_entity.lower(), new_entity.lower()),
                        (old_entity.upper(), new_entity.upper()),
                        (old_entity.capitalize(), new_entity.capitalize())
                    ]
                    
                    for old_variant, new_variant in case_variants:
                        if old_variant in updated_text:
                            updated_text = updated_text.replace(old_variant, new_variant)
                    
                    # Only re-embed if text actually changed
                    if updated_text != memory_text:
                        new_vec = self.embedder.embed(updated_text)
                        updated_payload = record.payload.copy()
                        updated_payload["memory"] = updated_text
                        updated_payload["metadata"]["updated_at"] = utc_now_iso()
                        
                        # Update in FAISS
                        self.vectors.update(memory_id, vector=new_vec, payload=updated_payload)
                        
                        # Update entity index to reflect the change
                        self.entities.add_memory_entities(memory_id, updated_text)
            
        except Exception as e:
            print(f"[faiss_update_warning] Failed to update FAISS entity references: {e}")

    def _prune_path_vectors_for_memory(self, memory_id: str) -> None:
        """Remove path-fragment vectors associated with a memory id from node vector index."""
        try:
            # Iterate through node vector store entries and delete matching 'path' payloads
            for vid, payload in list(self.node_vectors.items()):
                if (payload or {}).get("type") == "path" and (payload or {}).get("memory_id") == memory_id:
                    self.node_vectors.delete(vid)
        except Exception:
            pass

    # ---------- public API ----------
    def add(self, text_or_messages: str | List[Dict[str, Any]], *, user_id: Optional[str] = None, agent_id: Optional[str] = None, run_id: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None, infer: bool = False, messages: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        if isinstance(text_or_messages, list):
            # concatenate user/assistant texts for simple rule-based extraction
            text = "\n".join([m.get("content", "") for m in text_or_messages if m.get("role") in ("user", "assistant")])
        else:
            text = text_or_messages

        base_meta = dict(metadata or {})
        if user_id:
            base_meta["user_id"] = user_id
        if agent_id:
            base_meta["agent_id"] = agent_id
        if run_id:
            base_meta["run_id"] = run_id

        # Source of truth: build graph first, then embed text and push to vector store
        # Embed and attempt merge/update before adding a new record
        qres_for_store = self.embedder.embed_with_lexical([text])
        vec = qres_for_store["dense"][0] if qres_for_store.get("dense") else self.embedder.embed(text)
        lex = (qres_for_store.get("lexical") or [{}])[0] if qres_for_store else {}
        if isinstance(lex, dict):
            try:
                lex = {k: float(v) for k, v in lex.items()}
            except Exception:
                pass
        # Speed: smaller candidate pool during add; we'll still build graph fully
        initial_k = max(10, min(30, self.config.retrieval.initial_top_k))
        candidates = self.vectors.search(vec, limit=initial_k, filters=base_meta)
        merged_id = None
        for vid, score, payload in candidates:
            if score >= self.config.retrieval.update_similarity_threshold:
                # Update existing memory
                new_text = merge_texts(payload.get("memory", ""), text)
                new_res = self.embedder.embed_with_lexical([new_text])
                new_vec = new_res["dense"][0] if new_res.get("dense") else self.embedder.embed(new_text)
                new_lex = (new_res.get("lexical") or [{}])[0] if new_res else {}
                if isinstance(new_lex, dict):
                    try:
                        new_lex = {k: float(v) for k, v in new_lex.items()}
                    except Exception:
                        pass
                new_payload = self._make_payload(vid, new_text, payload.get("metadata"))
                new_payload["vector"] = new_vec
                new_payload["lexical"] = new_lex
                self.vectors.update(vid, vector=new_vec, payload=new_payload)
                self.db.add_history(vid, payload.get("memory"), new_text, "UPDATE", updated_at=new_payload["metadata"]["updated_at"], actor_id=base_meta.get("user_id"))
                self._index_entities_and_graph(vid, new_text, base_meta)
                merged_id = vid
                break
            elif score >= self.config.retrieval.merge_similarity_threshold:
                # Merge as new fact grouped with existing (optional grouping field)
                continue

        if merged_id is None:
            memory_id = str(uuid.uuid4())
            payload = self._make_payload(memory_id, text, base_meta)
            payload["vector"] = vec
            payload["lexical"] = lex
            # 1) Build graph memory FIRST with relation extraction (LLM once)
            # If the caller passed in explicit context messages, use them; otherwise if a list of messages
            # was provided as the primary input, use that. This ensures we only store the user's text
            # while still giving the relation extractor sufficient conversational context.
            conv_msgs = text_or_messages if isinstance(text_or_messages, list) else messages
            self._index_entities_and_graph(memory_id, text, base_meta, messages=conv_msgs)
            # 2) Store vector SECOND to ensure graph built regardless of FAISS
            self.vectors.add([vec], [payload], [memory_id])
            self.db.add_history(memory_id, None, text, "ADD", created_at=payload["metadata"]["created_at"], updated_at=payload["metadata"]["updated_at"], actor_id=base_meta.get("user_id"))
            return {"results": [payload]}
        else:
            return {"results": [self.vectors.get(merged_id).payload]}

    def search(self, query: str, *, user_id: Optional[str] = None, agent_id: Optional[str] = None, run_id: Optional[str] = None, limit: int = 10) -> Dict[str, Any]:
        q_res = self.embedder.embed_with_lexical([query])
        vec = q_res["dense"][0] if q_res.get("dense") else self.embedder.embed(query)
        q_lex = (q_res.get("lexical") or [{}])[0] if q_res else {}
        filters: Dict[str, Any] = {}
        if user_id:
            filters["user_id"] = user_id
        if agent_id:
            filters["agent_id"] = agent_id
        if run_id:
            filters["run_id"] = run_id

        # 1) Vector search - full power
        initial_k = max(50, self.config.retrieval.initial_top_k)  # No artificial cap
        vector_hits = self.vectors.search(vec, limit=initial_k, filters=filters)

        # 2) Expand via graph around detected query entities (k-hop)
        # Extract entities from query using same heuristic as entity index

        q_entities = list(set(ENTITY_PATTERN.findall(query)))
        # MANDATORY case-insensitive entity recovery: map lowercase tokens to known node names
        if not q_entities:
            import re as _re_ci
            tokens = _re_ci.findall(r"[A-Za-z][A-Za-z0-9_]+", query)
            if tokens:
                nodes = self.graph.query_nodes()  # Fail fast if graph is broken
                name_map = { (n.name or "").lower(): (n.name or "") for n in nodes if getattr(n, "name", None) }
                for tok in tokens:
                    cand = name_map.get(tok.lower())
                    if cand:
                        q_entities.append(cand)
                q_entities = list(set(q_entities))
        # Recognize special intents for name recall
        if "what do i call you" in query.lower() or "your name" in query.lower():
            # Add user node to seed entities if provided to expand through calls_assistant edges
            if user_id:
                q_entities.append(user_id)
        # MANDATORY graph k-hop expansion - full depth for accuracy
        graph_relations = []
        for e in q_entities:
            graph_relations.extend(self.graph.k_hop(e, k=4, limit_per_hop=100))  # Deeper traversal
        # Prepare relations dicts early so it's always defined
        rels_dicts = [r.__dict__ for r in graph_relations]

        # 3) MANDATORY collection of related memory ids via entity index for the graph entities
        related_memory_ids = set()
        for rel in graph_relations:
            # Fail fast if entity index is broken
            related_memory_ids.update(self.entities.memories_for_entity(rel.source))
            related_memory_ids.update(self.entities.memories_for_entity(rel.destination))

        # 3.5) MANDATORY traversal for comprehensive graph-based retrieval
        traverse_nodes: List[Tuple[str, float]] = []
        # MANDATORY traversal - full power for maximum accuracy
        t_res = self.traverse(query, user_id=user_id, max_hops=8, limit=self.config.retrieval.final_top_k * 3)
        if getattr(self.config.retrieval, "traverse_debug", False):
            print("[traverse_debug]", t_res)
        for ent in t_res.get("results", []) or []:
            node = ent.get("node")
            score = float(ent.get("score", 0.0) or 0.0)
            if not node:
                continue
            traverse_nodes.append((node, score))
            # MANDATORY entity lookup - fail fast if entity index is broken
            mids = self.entities.memories_for_entity(node)
            for mid in mids:
                related_memory_ids.add(mid)

        # 4) Combine candidates: vector + entity-linked memories
        candidate_map: Dict[str, Dict[str, Any]] = {}
        candidate_scores: Dict[str, float] = {}
        for vid, score, payload in vector_hits:
            candidate_map[vid] = payload
            candidate_scores[vid] = max(candidate_scores.get(vid, 0.0), score)
        for mid in related_memory_ids:
            rec = self.vectors.get(mid)
            if rec and rec.payload:
                candidate_map[mid] = rec.payload
                candidate_scores[mid] = max(candidate_scores.get(mid, 0.0), 0.0)
        
        # Fallback: if no candidates found and user_id filter was used, retry without it
        if not candidate_map and user_id:
            try:
                # Retry vector search without user_id filter
                vector_hits_fallback = self.vectors.search(vec, limit=initial_k, filters={})
                for vid, score, payload in vector_hits_fallback:
                    candidate_map[vid] = payload
                    candidate_scores[vid] = max(candidate_scores.get(vid, 0.0), score)
                if getattr(self.config.retrieval, "debug", False):
                    print(f"[search_fallback] Retried without user_id filter, found {len(candidate_map)} candidates")
            except Exception:
                pass

        # Boost candidates coming from traverse nodes to ensure visibility in hybrid rerank
        for node, nscore in traverse_nodes:
            try:
                mids = self.entities.memories_for_entity(node)
            except Exception:
                mids = []
            # map node score to a conservative boost
            boost = max(0.0, min(1.0, 0.4 + 0.3 * nscore))
            for mid in mids:
                candidate_scores[mid] = max(candidate_scores.get(mid, 0.0), boost)

        # Temporal intent boost (lightweight): if query mentions time window/date, boost memories
        # whose edges have matching time props
        qlow = query.lower()
        desired_window = None
        if any(t in qlow for t in ["yesterday", "last night"]):
            desired_window = "YESTERDAY"
        elif any(t in qlow for t in ["today", "this morning", "this evening", "tonight"]):
            desired_window = "TODAY"
        elif "last week" in qlow:
            desired_window = "LAST_WEEK"
        elif "last month" in qlow:
            desired_window = "LAST_MONTH"
        import re as _t_re
        date_match = _t_re.search(r"(20\d{2}-\d{2}-\d{2})", query)
        desired_date = date_match.group(1) if date_match else None
        if (desired_window or desired_date) and candidate_map:
            try:
                edges = self.graph.query_edges()  # unified API, no limit
                cand_ids = set(candidate_map.keys())
                for e in edges:
                    if getattr(e, "memory_id", None) in cand_ids:
                        if desired_window:
                            tw = self.graph.get_edge_prop(e.source, e.relationship, e.destination, "time_window")
                            if (tw or "").upper() == desired_window:
                                candidate_scores[e.memory_id] = max(candidate_scores.get(e.memory_id, 0.0), 0.2)
                        if desired_date and candidate_scores.get(getattr(e, "memory_id", None), None) is not None:
                            dt = self.graph.get_edge_prop(e.source, e.relationship, e.destination, "date")
                            if dt == desired_date:
                                candidate_scores[e.memory_id] = max(candidate_scores.get(e.memory_id, 0.0), 0.25)
            except Exception:
                pass

        # 5) Re-score candidates using hybrid scoring
        rescored: List[Tuple[str, float]] = []
        w = self.config.retrieval.scoring
        for vid, pl in candidate_map.items():
            sim = candidate_scores.get(vid, 0.0)
            # graph score: if memory connects to any query entity via graph
            mem_entities = self.entities.entities_for_memory(vid)
            graph_overlap = len(set(mem_entities) & set(q_entities))
            graph_score = min(1.0, graph_overlap / 3.0)
            # recency score (no decay)
            ts = pl.get("metadata", {}).get("updated_at") or pl.get("metadata", {}).get("created_at")
            recency_score = 0.5 if ts else 0.0
            # evidence (currently count of relations touching its entities)
            evidence = 0.0
            for e in mem_entities:
                evidence += len(self.graph.neighbors(e, direction="both", limit=5)) * 0.02
            evidence = min(evidence, 1.0)
            # lexical score using BGE-M3 lexical weights (if available)
            lexical_score = 0.0
            doc_lex = pl.get("lexical") if isinstance(pl, dict) else None
            if isinstance(doc_lex, dict) and isinstance(q_lex, dict) and q_lex:
                try:
                    lexical_score = float(self.embedder.lexical_score(q_lex, doc_lex))
                except Exception:
                    lexical_score = 0.0
            w_lex = getattr(w, "w_lexical", 0.2)
            final = (w.w_similarity * sim
                     + w.w_graph * graph_score
                     + w.w_recency * recency_score
                     + w.w_evidence * evidence
                     + w_lex * lexical_score)
            rescored.append((vid, final))

        rescored.sort(key=lambda x: x[1], reverse=True)
        # Map of final hybrid scores per vid
        final_score_map: Dict[str, float] = {vid: sc for vid, sc in rescored}
        top_ids = [vid for vid, _ in rescored[: self.config.retrieval.final_top_k]]
        results = []
        for vid in top_ids:
            rec = self.vectors.get(vid)
            if rec and rec.payload:
                results.append({
                    "id": vid,
                    "memory": rec.payload.get("memory"),
                    "metadata": rec.payload.get("metadata"),
                    # Expose the final hybrid score, not just vector similarity
                    "score": final_score_map.get(vid, candidate_scores.get(vid, 0.0)),
                })

        # GRAPH-ENHANCED BGE-M3 reranking with hybrid vector-graph intelligence
        # Only apply if reranker is enabled in config
        if results and getattr(self.config.reranker, "enabled", True):
            # Extract candidates for graph-enhanced reranking
            candidates = results[: self.config.reranker.top_k]
            mpl = getattr(self.config.reranker, "max_passage_length", 96)
            
            # Build comprehensive graph contexts for each candidate
            graph_contexts = []
            graph_debug_info = []
            
            for candidate in candidates:
                memory_id = candidate.get("id", "")
                if memory_id:
                    # Build rich graph context using existing storage structure
                    graph_ctx = self.build_comprehensive_graph_context(
                        memory_id=memory_id, 
                        query_entities=q_entities, 
                        query=query
                    )
                    graph_contexts.append(graph_ctx)
                    
                    # Store debug info if enabled
                    if getattr(self.config.reranker, "debug_scores", False):
                        graph_debug_info.append({
                            "memory_id": memory_id,
                            "context_quality": graph_ctx.get("context_quality", {}),
                            "entity_overlap": graph_ctx.get("entity_overlap", 0),
                            "graph_confidence": graph_ctx.get("graph_confidence", 0.0)
                        })
                else:
                    # Fallback for candidates without memory_id
                    graph_contexts.append({
                        "memory_entities": [],
                        "direct_relationships": [],
                        "entity_overlap": 0,
                        "graph_confidence": 0.0,
                        "traversal_strength": 0.0
                    })
            
            # Use graph-enhanced BGE-M3 reranking
            try:
                graph_rerank_results = self.embedder.graph_enhanced_rerank_m3(
                    query=query, 
                    memory_records=candidates, 
                    graph_contexts=graph_contexts, 
                    max_passage_length=mpl
                )
                
                # Integrate graph-enhanced scores with original vector scores
                for i, (candidate, graph_rerank_result) in enumerate(zip(candidates, graph_rerank_results)):
                    if i < len(results):
                        original_score = float(results[i]["score"])
                        graph_enhanced_score = float(graph_rerank_result["final_score"])
                        
                        # Adaptive blending based on graph context quality
                        graph_quality = graph_contexts[i].get("context_quality", {}).get("overall_score", 0.0)
                        entity_overlap = graph_contexts[i].get("entity_overlap", 0)
                        
                        # Determine blend strategy based on graph intelligence
                        if entity_overlap > 1 and graph_quality > 0.7:
                            # High-quality graph context: trust graph enhancement more
                            blend_weight = 0.25  # 75% graph-enhanced, 25% original
                            fusion_strategy = "graph_dominant"
                        elif entity_overlap > 0 and graph_quality > 0.4:
                            # Medium-quality graph context: balanced approach
                            blend_weight = 0.4  # 60% graph-enhanced, 40% original
                            fusion_strategy = "graph_balanced"
                        elif graph_quality > 0.2:
                            # Low-quality graph context: slight graph bias
                            blend_weight = 0.6  # 40% graph-enhanced, 60% original
                            fusion_strategy = "graph_assisted"
                        else:
                            # No meaningful graph context: primarily original
                            blend_weight = 0.8  # 20% graph-enhanced, 80% original
                            fusion_strategy = "vector_dominant"
                        
                        # Compute final hybrid score
                        final_score = blend_weight * original_score + (1 - blend_weight) * graph_enhanced_score
                        
                        # Store comprehensive scoring information
                        results[i]["score"] = final_score
                        results[i]["graph_enhanced_details"] = {
                            "original_score": original_score,
                            "graph_enhanced_score": graph_enhanced_score,
                            "fusion_strategy": fusion_strategy,
                            "graph_quality": graph_quality,
                            "entity_overlap": entity_overlap,
                            "blend_weight": blend_weight,
                            "fusion_type": graph_rerank_result.get("fusion_type", "unknown"),
                            "graph_confidence": graph_contexts[i].get("graph_confidence", 0.0),
                            "traversal_strength": graph_contexts[i].get("traversal_strength", 0.0),
                            "bgem3_scores": {
                                "dense": graph_rerank_result.get("dense", 0.0),
                                "sparse": graph_rerank_result.get("sparse", 0.0),
                                "colbert": graph_rerank_result.get("colbert", 0.0)
                            },
                            "graph_boosts": {
                                "entity_boost": graph_rerank_result.get("entity_boost", 0.0),
                                "confidence_boost": graph_rerank_result.get("confidence_boost", 0.0),
                                "traversal_boost": graph_rerank_result.get("traversal_boost", 0.0)
                            }
                        }
                        
                        # Debug output if enabled
                        if getattr(self.config.reranker, "debug_scores", False):
                            memory_preview = results[i]["memory"][:100] + "..." if len(results[i]["memory"]) > 100 else results[i]["memory"]
                            print(f"[graph_enhanced_debug] Doc {i+1}: {final_score:.3f}")
                            print(f"  Original: {original_score:.3f}, Graph-Enhanced: {graph_enhanced_score:.3f}")
                            print(f"  Strategy: {fusion_strategy}, Graph Quality: {graph_quality:.3f}")
                            print(f"  Entity Overlap: {entity_overlap}, Graph Confidence: {graph_contexts[i].get('graph_confidence', 0.0):.3f}")
                            print(f"  BGE-M3 D/S/C: {graph_rerank_result.get('dense', 0.0):.3f}/{graph_rerank_result.get('sparse', 0.0):.3f}/{graph_rerank_result.get('colbert', 0.0):.3f}")
                            print(f"  Text: {memory_preview}")
                            
                            # Show graph context if available
                            if graph_contexts[i].get("direct_relationships"):
                                print(f"  Relationships: {graph_contexts[i]['direct_relationships'][:2]}")
                
            except Exception as e:
                # Fallback to basic reranking if graph enhancement fails
                print(f"[graph_enhanced_warning] Graph enhancement failed: {e}, falling back to basic reranking")
                basic_rerank_results = self.embedder.advanced_rerank_m3(
                    query, [r["memory"] for r in candidates], max_passage_length=mpl
                )
                
                for i, rerank_result in enumerate(basic_rerank_results):
                    if i < len(results):
                        original_score = float(results[i]["score"])
                        rerank_score = float(rerank_result["final_score"])
                        final_score = 0.5 * original_score + 0.5 * rerank_score
                        
                        results[i]["score"] = final_score
                        results[i]["fallback_rerank_details"] = {
                            "original_score": original_score,
                            "rerank_score": rerank_score,
                            "fusion_type": rerank_result["fusion_type"]
                        }
            
            # Sort by final hybrid scores
            results.sort(key=lambda x: x["score"], reverse=True)

        # Confidence gating / no-answer policy - but skip for "who is" queries which should always return results
        is_who_query = any(pattern in query.lower() for pattern in ["who is", "who are", "what is"])
        # NOTE: Removed confidence gating that returned empty results silently
        # Results are already ranked by quality; forcing empty return hides data from traversal/graph
        # Let the caller decide filtering based on presence/absence of results
        if not results and graph_relations:
            # pick memories connected to the first few relations' entities
            fallback_ids = list(related_memory_ids)[: self.config.retrieval.final_top_k]
            for mid in fallback_ids:
                rec = self.vectors.get(mid)
                if rec and rec.payload:
                    results.append({
                        "id": mid,
                        "memory": rec.payload.get("memory"),
                        "metadata": rec.payload.get("metadata"),
                        "score": 0.6,  # conservative fallback score
                    })
        # If still no relations, derive from candidates' edges to help UI and augmentation
        if not rels_dicts and candidate_map:
            try:
                cand_ids = set(candidate_map.keys())
                edges = self.graph.query_edges()
                # Deduplicate relations using a set of tuples
                seen_rels = set()
                unique_rels = []
                for e in edges:
                    if getattr(e, "memory_id", None) in cand_ids:
                        rel_key = (e.source, e.relationship, e.destination)
                        if rel_key not in seen_rels:
                            seen_rels.add(rel_key)
                            unique_rels.append(e.__dict__)
                            if len(unique_rels) >= 20:  # Limit to prevent too many relations
                                break
                rels_dicts = unique_rels
            except Exception:
                pass
        
        # Also deduplicate the main rels_dicts if it has duplicates
        if rels_dicts:
            seen_main = set()
            deduped_rels = []
            for rel in rels_dicts:
                rel_key = (rel.get("source"), rel.get("relationship"), rel.get("destination"))
                if rel_key not in seen_main:
                    seen_main.add(rel_key)
                    deduped_rels.append(rel)
            rels_dicts = deduped_rels
        
        return {"results": results, "relations": rels_dicts}

    # ---------- graph traversal for complex relational queries (A*) ----------
    def traverse(self, query: str, *, user_id: Optional[str] = None, max_hops: int = 4, limit: int = 5) -> Dict[str, Any]:
        """Traverse graph using dynamic relation labels derived from possessive chains.

        - Parses a seed entity and a chain of terms (e.g., X's friend's father's friend → [FRIEND, FATHER, FRIEND])
        - Uses A*-style search that favors better label matches, recency, and stronger edges,
          while penalizing hubs and cycles. No predefined relation vocabulary.
        - Returns top endpoints with path evidence.
        """
        import re as _re
        import heapq as _heapq
        import math as _math
        from datetime import datetime as _dt

        def _upper(s: Optional[str]) -> Optional[str]:
            return s.upper() if isinstance(s, str) else s

        def _edge_match_score(label: str, term: str) -> float:
            lab = (label or "").upper()
            tok = (term or "").upper().replace(" ", "_")
            if not lab or not tok:
                return 0.0
            if tok == lab or tok in lab:
                return 1.0
            if tok.replace("_", "") in lab.replace("_", ""):
                return 0.85
            inter = len(set(tok) & set(lab))
            return max(0.0, min(0.6, inter / max(1, len(tok))))

        def _recency_score(ts: Optional[str]) -> float:
            if not ts:
                return 0.0
            try:
                dt = _dt.fromisoformat(ts)
                now = _dt.utcnow()
                days = (now - dt).total_seconds() / 86400.0
                return max(0.0, 1.0 / (1.0 + (days / 7.0)))  # 1 week half-life-ish
            except Exception:
                return 0.0

        def _degree_penalty(node: str) -> float:
            try:
                deg = len(self.graph.neighbors(node, direction="both", limit=200))
            except Exception:
                deg = 0
            # 0 for small degree, up to ~0.3 for very high
            return min(0.3, _math.log(1.0 + max(0, deg)) / 20.0)

        # 1) Seed entity detection with case-insensitive fallback
        quoted = _re.findall(r'"([^"]+)"', query)
        seed = quoted[0] if quoted else None
        if not seed:
            try:
                from Amem.entity_index import ENTITY_PATTERN
                ents = ENTITY_PATTERN.findall(query)
                seed = ents[0] if ents else None
            except Exception:
                seed = None
        
        # Fallback: try lowercase token matching against known nodes
        if not seed:
            try:
                nodes = self.graph.query_nodes()
                name_map = {(n.name or "").lower(): (n.name or "") for n in nodes if getattr(n, "name", None)}
                tokens = _re.findall(r"\b([a-zA-Z][a-zA-Z0-9_]+)\b", query)
                for tok in tokens:
                    cand = name_map.get(tok.lower())
                    if cand:
                        seed = cand
                        break
            except Exception:
                pass
        
        if not seed and user_id:
            seed = user_id
        if not seed:
            return {"results": [], "paths": [], "note": "no seed entity detected"}
        
        seed = _upper(seed)

        # 2) Chain term parsing from possessives (supports ' and ')
        lower = query.lower()
        raw_terms = _re.findall(r"[\u2019']s\s+([a-zA-Z_]+)", lower)
        stop = {"name","age","where","what","who","whom","when","why","how","is","are","was","were","do","did","does"}
        chain: List[str] = []
        for term in raw_terms:
            t = term.strip().upper()
            if len(t) > 3 and t.endswith("S"):
                t = t[:-1]
            if t in stop:
                continue
            chain.append(t)

        # If no chain: list neighbors of seed ranked
        if not chain or len(chain) > max_hops:
            rels = self._cache_neighbors(seed, base_limit=200)
            neighbor_scores: Dict[str, float] = {}
            neighbor_paths: Dict[str, List[Relation]] = {}
            for r in rels:
                other = r.destination if _upper(r.source) == seed else r.source
                other_u = _upper(other)
                neighbor_paths.setdefault(other_u, []).append(r)
                score = 0.5 + 0.3 * (r.weight or 1.0) + 0.2 * _recency_score(getattr(r, "updated_at", None))
                neighbor_scores[other_u] = max(neighbor_scores.get(other_u, 0.0), score) - _degree_penalty(other_u)
            ranked = sorted(neighbor_scores.items(), key=lambda x: x[1], reverse=True)[:limit]
            results = [{"node": n, "score": sc} for n, sc in ranked]
            paths = [[e.__dict__ for e in neighbor_paths.get(n, [])[:1]] for n, _ in ranked]
            return {"results": results, "paths": paths}

        # 3) A* over relation chain
        class _State:
            __slots__ = ("node", "idx", "path", "g")
            def __init__(self, node: str, idx: int, path: List[Relation], g: float):
                self.node = node
                self.idx = idx
                self.path = path
                self.g = g

        def _heuristic(idx: int) -> float:
            # optimistic remaining gain: each remaining step could match perfectly
            return float(len(chain) - idx)

        open_heap: List[Tuple[float, int, _State]] = []
        counter = 0
        start = _State(seed, 0, [], 0.0)
        _heapq.heappush(open_heap, (-(start.g + _heuristic(0)), counter, start))
        counter += 1
        best_g: Dict[Tuple[str, int], float] = {(seed, 0): 0.0}
        candidates: List[Tuple[str, float, List[Relation]]] = []
        expansions = 0
        max_expansions = 2000

        while open_heap and expansions < max_expansions and len(candidates) < (limit * 3):
            _, _, st = _heapq.heappop(open_heap)
            expansions += 1
            if st.idx >= len(chain):
                # terminal: full chain matched
                candidates.append((st.node, st.g, st.path))
                continue
            term = chain[st.idx]
            neighs = self._cache_neighbors(st.node, base_limit=200)
            for r in neighs:
                s = _edge_match_score(r.relationship, term)
                if s < 0.2:
                    continue
                other = r.destination if _upper(r.source) == st.node else r.source
                other_u = _upper(other)
                # edge quality
                w = float(getattr(r, "weight", 1.0) or 1.0)
                rec = _recency_score(getattr(r, "updated_at", None))
                score_edge = 0.65 * s + 0.2 * min(1.0, w) + 0.15 * rec - _degree_penalty(other_u)
                new_g = st.g + max(0.0, score_edge)
                new_idx = st.idx + 1
                key = (other_u, new_idx)
                if new_g <= best_g.get(key, -1e9):
                    continue
                best_g[key] = new_g
                nxt = _State(other_u, new_idx, st.path + [r], new_g)
                f = new_g + _heuristic(new_idx)
                _heapq.heappush(open_heap, (-f, counter, nxt))
                counter += 1

        # 4) Rerank candidates and return top-N with paths
        # Deduplicate by node keeping best score
        node_best: Dict[str, Tuple[float, List[Relation]]] = {}
        for n, sc, p in candidates:
            if n not in node_best or sc > node_best[n][0]:
                node_best[n] = (sc, p)
        ranked = sorted(node_best.items(), key=lambda x: x[1][0], reverse=True)[:limit]
        results = [{"node": n, "score": scp[0]} for n, scp in ranked]
        paths = [[e.__dict__ for e in scp[1]] for _, scp in ranked]
        return {"results": results, "paths": paths}

    def build_comprehensive_graph_context(self, memory_id: str, query_entities: List[str], query: str = "") -> Dict[str, Any]:
        """
        Build comprehensive graph context for a specific memory using existing storage structure.
        Integrates memory-graph relationships, entity mappings, and traversal paths.
        """
        try:
            # 1. Get direct relationships for this memory (existing approach)
            memory_rels = []
            memory_entities = set()
            
            try:
                all_edges = self.graph.query_edges()
                for edge in all_edges:
                    if hasattr(edge, 'memory_id') and edge.memory_id == memory_id:
                        rel_line = f"{edge.source} --{edge.relationship}--> {edge.destination}"
                        memory_rels.append(rel_line)
                        memory_entities.add(edge.source)
                        memory_entities.add(edge.destination)
            except Exception:
                pass
            
            # 2. Calculate entity overlap with query
            query_entities_set = set(q.upper() for q in query_entities)
            memory_entities_upper = set(e.upper() for e in memory_entities)
            entity_overlap = len(query_entities_set & memory_entities_upper)
            
            # 3. Find traversal paths involving memory entities
            relevant_paths = []
            traversal_scores = []
            
            # Limit traversal to top 5 entities for performance
            traversal_count = 0
            for entity in memory_entities:
                if traversal_count >= 5:
                    break
                if entity.upper() in query_entities_set:
                    traversal_count += 1
                    try:
                        # Use existing traverse method to find paths
                        traverse_result = self.traverse(
                            f'"{entity}"',  # Quote entity for precise matching
                            max_hops=5, 
                            limit=8
                        )
                        
                        paths = traverse_result.get("paths", [])
                        results = traverse_result.get("results", [])
                        
                        relevant_paths.extend(paths)
                        
                        # Extract traversal confidence scores
                        for result in results:
                            score = result.get("score", 0.0)
                            if score > 0:
                                traversal_scores.append(score)
                                
                    except Exception:
                        continue
            
            # 4. Calculate graph confidence metrics
            graph_confidence = 0.0
            traversal_strength = 0.0
            
            # Base confidence from direct relationships
            if memory_rels:
                graph_confidence += min(0.6, len(memory_rels) * 0.2)
            
            # Entity overlap bonus
            if entity_overlap > 0:
                graph_confidence += min(0.4, entity_overlap * 0.2)
            
            # Traversal path strength
            if traversal_scores:
                avg_traversal_score = sum(traversal_scores) / len(traversal_scores)
                traversal_strength = min(1.0, avg_traversal_score)
                graph_confidence += min(0.3, traversal_strength * 0.3)
            
            # Cap confidence at 1.0
            graph_confidence = min(1.0, graph_confidence)
            
            # 5. Build connected entity network
            connected_entities = set(memory_entities)
            
            # Add entities from traversal paths
            for path in relevant_paths:
                for edge in path:
                    try:
                        if isinstance(edge, dict):
                            connected_entities.add(edge.get("source", ""))
                            connected_entities.add(edge.get("destination", ""))
                        else:
                            connected_entities.add(getattr(edge, "source", ""))
                            connected_entities.add(getattr(edge, "destination", ""))
                    except Exception:
                        continue
            
            # Remove empty strings
            connected_entities = [e for e in connected_entities if e.strip()]
            
            # 6. Find related memories through shared entities
            related_memory_ids = set()
            for entity in connected_entities:
                try:
                    entity_memories = self.entities.memories_for_entity(entity)
                    related_memory_ids.update(entity_memories)
                except Exception:
                    continue
            
            # Remove the current memory from related set
            related_memory_ids.discard(memory_id)
            
            # 7. Extract relationship patterns for enhanced understanding
            relationship_types = set()
            entity_roles = {}  # entity -> [roles as source, roles as destination]
            
            for rel_line in memory_rels:
                try:
                    if "--" in rel_line and "-->" in rel_line:
                        parts = rel_line.split("--")
                        if len(parts) >= 3:
                            src = parts[0].strip()
                            rel_type = parts[1].strip()
                            dst = parts[2].split("-->")[1].strip() if "-->" in parts[2] else parts[2].strip()
                            
                            relationship_types.add(rel_type)
                            
                            # Track entity roles
                            if src not in entity_roles:
                                entity_roles[src] = {"as_source": [], "as_destination": []}
                            if dst not in entity_roles:
                                entity_roles[dst] = {"as_source": [], "as_destination": []}
                            
                            entity_roles[src]["as_source"].append(rel_type)
                            entity_roles[dst]["as_destination"].append(rel_type)
                except Exception:
                    continue
            
            # 8. Query-specific relevance scoring
            query_relevance = 0.0
            if query:
                query_words = set(query.lower().split())
                
                # Check if query words appear in entities
                for entity in memory_entities:
                    entity_words = set(entity.lower().split())
                    overlap = len(query_words & entity_words)
                    if overlap > 0:
                        query_relevance += overlap * 0.2
                
                # Check if query words appear in relationship types
                for rel_type in relationship_types:
                    rel_words = set(rel_type.lower().replace("_", " ").split())
                    overlap = len(query_words & rel_words)
                    if overlap > 0:
                        query_relevance += overlap * 0.1
                
                query_relevance = min(1.0, query_relevance)
            
            # 9. Return comprehensive context
            return {
                "memory_entities": list(memory_entities),
                "direct_relationships": memory_rels,
                "entity_overlap": entity_overlap,
                "graph_confidence": graph_confidence,
                "traversal_strength": traversal_strength,
                "relevant_paths": relevant_paths[:5],  # Top 5 most relevant paths
                "connected_entities": connected_entities[:10],  # Top 10 connected entities
                "related_memory_ids": list(related_memory_ids)[:8],  # Top 8 related memories
                "relationship_types": list(relationship_types),
                "entity_roles": entity_roles,
                "query_relevance": query_relevance,
                "context_quality": {
                    "has_relationships": len(memory_rels) > 0,
                    "has_entity_overlap": entity_overlap > 0,
                    "has_traversal_paths": len(relevant_paths) > 0,
                    "has_connected_entities": len(connected_entities) > 1,
                    "relationship_diversity": len(relationship_types),
                    "overall_score": (graph_confidence + traversal_strength + query_relevance) / 3
                }
            }
            
        except Exception as e:
            # Return minimal context on error
            return {
                "memory_entities": [],
                "direct_relationships": [],
                "entity_overlap": 0,
                "graph_confidence": 0.0,
                "traversal_strength": 0.0,
                "relevant_paths": [],
                "connected_entities": [],
                "related_memory_ids": [],
                "relationship_types": [],
                "entity_roles": {},
                "query_relevance": 0.0,
                "context_quality": {
                    "has_relationships": False,
                    "has_entity_overlap": False,
                    "has_traversal_paths": False,
                    "has_connected_entities": False,
                    "relationship_diversity": 0,
                    "overall_score": 0.0
                },
                                "error": str(e)
            }

    def update(self, memory_id: str, new_text: str):
        rec = self.vectors.get(memory_id)
        if not rec:
            return None
        old_text = rec.payload.get("memory")
        new_text = merge_texts(old_text, new_text)
        new_vec = self.embedder.embed(new_text)
        new_payload = self._make_payload(memory_id, new_text, rec.payload.get("metadata"))
        self.vectors.update(memory_id, vector=new_vec, payload=new_payload)
        self.db.add_history(memory_id, old_text, new_text, "UPDATE", updated_at=new_payload["metadata"]["updated_at"])
        # Remove old graph edges for this memory id and re-index
        try:
            self.graph.delete_by_memory_id(memory_id)
        except Exception:
            pass
        # Remove old path fragments tied to this memory
        self._prune_path_vectors_for_memory(memory_id)
        self._index_entities_and_graph(memory_id, new_text, rec.payload.get("metadata"))
        return {"id": memory_id, "memory": new_text}

    def delete(self, memory_id: str):
        """Delete a single memory by its ID from all stores."""
        rec = self.vectors.get(memory_id)
        if rec and rec.payload:
            self.db.add_history(memory_id, rec.payload.get("memory"), None, "DELETE", updated_at=utc_now_iso())
        self.vectors.delete(memory_id)
        try:
            self.graph.delete_by_memory_id(memory_id)
        except Exception:
            pass
        try:
            self.entities.delete_by_memory_id(memory_id)
        except Exception:
            pass
        # Also prune path fragment vectors associated with this memory
        self._prune_path_vectors_for_memory(memory_id)
        return {"id": memory_id, "deleted": True}

    def delete_all(self) -> Dict[str, Any]:
        """Delete ALL memories from all stores (FAISS, graph, entity_index, history)."""
        deleted_count = 0
        # Get all memory IDs before deleting
        all_ids = self.vectors.list()
        deleted_count = len(all_ids)
        
        # Delete all vectors from FAISS
        for mid in all_ids:
            try:
                rec = self.vectors.get(mid)
                if rec and rec.payload:
                    self.db.add_history(mid, rec.payload.get("memory"), None, "DELETE", updated_at=utc_now_iso())
                self.vectors.delete(mid)
            except Exception:
                pass
        
        # Delete all from graph
        try:
            self.graph.delete_all()
        except Exception:
            pass
        
        # Delete all from entity index
        try:
            self.entities.delete_all()
        except Exception:
            pass
        
        # Delete all node vectors
        try:
            self.node_vectors.reset()
        except Exception:
            pass
        
        # Clear caches
        self._clear_traverse_caches()
        
        return {"deleted_count": deleted_count, "status": "all_deleted"}


    # ---------- advanced delete/update by query ----------
    def delete_by_query(self, query: str, *, user_id: Optional[str] = None, max_candidates: int = 200) -> Dict[str, Any]:
        q = query.strip().lower()
        deleted_ids: List[str] = []

        # delete all for user
        if any(term in q for term in ["delete all", "delete everything", "wipe all", "reset all", "delete all the memories"]):
            # Remove all vector docs for this user and their graph edges
            for vid, payload in list(self.vectors.items()):
                meta = (payload or {}).get("metadata", {})
                if user_id and meta.get("user_id") != user_id:
                    continue
                self.vectors.delete(vid)
                try:
                    self.graph.delete_by_memory_id(vid)
                except Exception:
                    pass
                deleted_ids.append(vid)
            # Reset index because we removed many
            try:
                self.vectors.reset_index()
            except Exception:
                pass
            return {"deleted_ids": deleted_ids}

        # Otherwise, delete by textual and entity match
        tokens = [t for t in q.replace(",", " ").split() if len(t) > 2]

        # Candidate by vector search
        vec = self.embedder.embed(query)
        filters: Dict[str, Any] = {}
        if user_id:
            filters["user_id"] = user_id
        candidates = self.vectors.search(vec, limit=max_candidates, filters=filters)

        def matches(payload_memory: str) -> bool:
            mem_low = (payload_memory or "").lower()
            return any(tok in mem_low for tok in tokens)

        for vid, score, payload in list(candidates):
            text = (payload or {}).get("memory", "")
            if matches(text):
                self.vectors.delete(vid)
                try:
                    self.graph.delete_by_memory_id(vid)
                except Exception:
                    pass
                try:
                    self.entities.delete_by_memory_id(vid)
                except Exception:
                    pass
                deleted_ids.append(vid)

        # Also prune graph edges matching entities specified in the query (case-insensitive)
        for tok in tokens:
            try:
                # Use case-insensitive entity deletion
                self.graph.delete_by_entity_case_insensitive(tok)
            except Exception:
                pass

        # If we deleted a lot, consider resetting index
        if len(deleted_ids) > max(50, len(self.vectors.docstore) // 3):
            try:
                self.vectors.reset_index()
            except Exception:
                pass

        return {"deleted_ids": deleted_ids}

    def history(self, memory_id: str) -> List[Dict[str, Any]]:
        return self.db.get_history(memory_id)

