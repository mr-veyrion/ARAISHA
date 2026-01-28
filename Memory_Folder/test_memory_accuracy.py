
import os
import sys
import uuid
import json
import time

# ============================================================================
# AMEM MEMORY SYSTEM IMPORTS
# ============================================================================
try:
    from Amem.memory import OfflineMemory
    from Amem.local_config import OfflineMemoryConfig
    from Amem.graph_store import Relation
    from Amem.utils import utc_now_iso, md5_hash
    from Amem.memory_system import graph_only_search, unified_memory_search
except ImportError:
    # Add extraction path to sys.path if running from separate dir
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    from Amem.memory import OfflineMemory
    from Amem.local_config import OfflineMemoryConfig
    from Amem.graph_store import Relation
    from Amem.utils import utc_now_iso, md5_hash
    from Amem.memory_system import graph_only_search, unified_memory_search



#============================================================================
# DATA: 50 MEMORIES
# ============================================================================
MEMORIES = [
    # Person: Abhishek
    "Abhishek — LIKES — Black color",
    "Abhishek — STUDIES — Artificial Intelligence",
    "Abhishek — USES — Linux",
    "Abhishek — PREFERS — Dark mode interfaces",
    "Abhishek — WORKS_ON — After Human project",
    "Abhishek — INTERESTED_IN — Digital immortality",
    "Abhishek — KNOWS — Python",
    "Abhishek — KNOWS — System design",
    "Abhishek — DISLIKES — Loud environments",
    "Abhishek — PREFERS — Minimal UI",

    # Person: Riya
    "Riya — LIVES_IN — Bangalore",
    "Riya — LIKES — Street photography",
    "Riya — USES — iPhone",
    "Riya — PREFERS — iOS ecosystem",
    "Riya — WORKS_AS — UX designer",
    "Riya — DISLIKES — Cluttered layouts",
    "Riya — LIKES — Pastel colors",
    "Riya — INTERESTED_IN — Human-centered design",

    # Person: Karan
    "Karan — LIVES_IN — Delhi",
    "Karan — WORKS_AS — Data analyst",
    "Karan — KNOWS — SQL",
    "Karan — KNOWS — Power BI",
    "Karan — LIKES — Football",
    "Karan — SUPPORTS — Manchester United",
    "Karan — DISLIKES — Manual data entry",

    # Person: Meera
    "Meera — LIVES_IN — Pune",
    "Meera — STUDIES — Psychology",
    "Meera — INTERESTED_IN — Cognitive behavior",
    "Meera — LIKES — Journaling",
    "Meera — PREFERS — Quiet cafés",
    "Meera — DISLIKES — Social media overload",

    # Person: Aarav
    "Aarav — LIVES_IN — Mumbai",
    "Aarav — WORKS_AS — Startup founder",
    "Aarav — INTERESTED_IN — Venture capital",
    "Aarav — LIKES — Networking events",
    "Aarav — USES — MacBook",
    "Aarav — PREFERS — Remote work",

    # Shared / Relational
    "Abhishek — COLLABORATES_WITH — Riya",
    "Abhishek — DISCUSSES — Ethics of AI",
    "Riya — WORKS_WITH — Aarav",
    "Karan — CONSULTS — Aarav",
    "Meera — READS — Research papers",
    "Aarav — INVESTS_IN — Tech startups",
    "Riya — AVOIDS — Dark color palettes",
    "Abhishek — AVOIDS — Vendor lock-in",
    "Karan — PREFERS — Structured datasets",
    "Meera — PRACTICES — Mindfulness",
    "Aarav — FOLLOWS — Startup accelerators",
    "Abhishek — BELIEVES_IN — Open-source philosophy",
    "Riya — BELIEVES_IN — Design empathy",
]

# ============================================================================
# DATA: 50 QUERIES WITH GROUND TRUTH
# ============================================================================
QUERIES = [
    # Direct
    {"q": "Who likes black color?", "a": ["Abhishek"]},
    {"q": "Who studies Artificial Intelligence?", "a": ["Abhishek"]},
    {"q": "Who prefers minimal UI?", "a": ["Abhishek"]},
    {"q": "Who lives in Pune?", "a": ["Meera"]},
    {"q": "Who uses a MacBook?", "a": ["Aarav"]},

    # Attribute-based
    {"q": "Which people dislike clutter or overload?", "a": ["Riya", "Meera"]}, # Cluttered layouts, Social media overload
    {"q": "Who prefers quiet environments?", "a": ["Meera", "Abhishek"]}, # Quiet cafes, Dislikes loud environments
    {"q": "Who avoids dark color palettes?", "a": ["Riya"]},
    {"q": "Who prefers remote work?", "a": ["Aarav"]},
    {"q": "Who dislikes manual processes?", "a": ["Karan"]}, # Manual data entry

    # Skill / Knowledge
    {"q": "Who knows Python?", "a": ["Abhishek"]},
    {"q": "Who has system design knowledge?", "a": ["Abhishek"]},
    {"q": "Who works with data tools?", "a": ["Karan"]}, # SQL, Power BI
    {"q": "Who reads research papers?", "a": ["Meera"]},
    {"q": "Who understands human behavior?", "a": ["Meera", "Riya"]}, # Human-centered design, Psychology

    # Multi-hop (Graph Traversal)
    {"q": "Who collaborates with someone interested in design?", "a": ["Abhishek"]}, # Riya is UX designer/Human-centered
    {"q": "Who works with a startup founder?", "a": ["Riya", "Karan"]}, # Riya works with Aarav, Karan consults Aarav
    {"q": "Who is connected to venture capital interests?", "a": ["Aarav", "Riya", "Karan"]}, # Aarav direct, others indirect
    {"q": "Who is indirectly linked to tech startups?", "a": ["Riya", "Karan", "Abhishek"]}, # Via Aarav or Interests
    {"q": "Who collaborates with a UX designer?", "a": ["Abhishek"]},

    # Preference + Role
    {"q": "Which designer dislikes clutter?", "a": ["Riya"]},
    {"q": "Which AI student prefers dark interfaces?", "a": ["Abhishek"]},
    {"q": "Which founder prefers remote work?", "a": ["Aarav"]},
    {"q": "Which analyst prefers structured data?", "a": ["Karan"]},
    {"q": "Which psychology student avoids overload?", "a": ["Meera"]},

    # Belief / Philosophy
    {"q": "Who believes in open-source principles?", "a": ["Abhishek"]},
    {"q": "Who believes in empathy-driven work?", "a": ["Riya"]},
    {"q": "Who discusses ethics-related topics?", "a": ["Abhishek"]},
    {"q": "Who is interested in digital immortality?", "a": ["Abhishek"]},
    {"q": "Who avoids vendor lock-in?", "a": ["Abhishek"]},

    # Environment / Lifestyle
    {"q": "Who prefers quiet cafés?", "a": ["Meera"]},
    {"q": "Who dislikes loud environments?", "a": ["Abhishek"]},
    {"q": "Who enjoys journaling?", "a": ["Meera"]},
    {"q": "Who attends networking events?", "a": ["Aarav"]},
    {"q": "Who practices mindfulness?", "a": ["Meera"]},

    # Indirect / Compound
    {"q": "Find people interested in AI or cognitive science", "a": ["Abhishek", "Meera"]},
    {"q": "Find people who design or analyze systems", "a": ["Abhishek", "Riya"]}, # System design, UX design
    {"q": "Who is connected to football indirectly?", "a": ["Karan"]}, # Manchester United
    {"q": "Who works in roles related to product or startups?", "a": ["Aarav", "Riya"]}, # Founder, UX
    {"q": "Who might prefer clean UI experiences?", "a": ["Abhishek", "Riya"]}, # Minimal UI, dislikes cluttered

    # Stress Queries (Hard)
    {"q": "Who would most likely support open tools and Linux?", "a": ["Abhishek"]},
    {"q": "Who would dislike cluttered dashboards?", "a": ["Riya"]},
    {"q": "Who could advise a startup on data visualization?", "a": ["Karan"]}, # Power BI
    {"q": "Who aligns with ethical AI discussions?", "a": ["Abhishek"]},
    {"q": "Who might enjoy calm workspaces?", "a": ["Meera", "Abhishek"]}, # Quiet cafes, Dislikes loud

    # Abstract / Semantic
    {"q": "Who values structure over chaos?", "a": ["Karan", "Riya", "Abhishek"]}, # Structured datasets, Dislikes clutter, Dislikes loud
    {"q": "Who is least likely to enjoy social media noise?", "a": ["Meera"]}, # Dislikes social media overload
    {"q": "Who combines technical skill with philosophy?", "a": ["Abhishek"]}, # Ethics, Open-source
    {"q": "Who bridges design and technology?", "a": ["Riya", "Abhishek"]},
    {"q": "Who is most aligned with human-centric thinking?", "a": ["Riya", "Meera"]},
]


# ============================================================================
# HELPER: BYPASS EXTRACTION & STORE DIRECTLY
# ============================================================================
def store_memory_direct(mem: OfflineMemory, text_line: str, user_id: str = "tester"):
    """
    Parses 'Source — Relation — Destination' from text.
    1. Upserts nodes to Graph.
    2. Upserts edge to Graph.
    3. Adds text to Vector Store.
    """
    try:
        parts = text_line.split(" — ")
        if len(parts) != 3:
            print(f"⚠️  Skipping invalid format: {text_line}")
            return
        
        src, rel, dst = parts[0].strip(), parts[1].strip(), parts[2].strip()
        
        # Standardize
        src_node = src.upper()
        rel_edge = rel.upper().replace(" ", "_")
        dst_node = dst.upper()
        
        memory_id = str(uuid.uuid4())
        now = utc_now_iso()
        
        # 1. Upsert Nodes
        for node_name in [src_node, dst_node]:
            mem.graph.upsert_node(
                node_id=f"ent::{node_name}", 
                name=node_name, 
                label="Entity", 
                props_json="{}", 
                created_at=now, 
                updated_at=now
            )
            
        # 2. Upsert Edge
        mem.graph.upsert(Relation(
            source=src_node, 
            relationship=rel_edge, 
            destination=dst_node, 
            created_at=now, 
            updated_at=now, 
            memory_id=memory_id
        ))
        
        # 3. Add to Vector Store
        # Generate embedding
        vec = mem.embedder.embed(text_line)
        
        # Metadata
        meta = {
            "user_id": user_id,
            "created_at": now,
            "updated_at": now,
            "source": "test_script"
        }
        
        payload = {
            "id": memory_id,
            "memory": text_line,
            "hash": md5_hash(text_line),
            "metadata": meta,
            "vector": vec
        }
        
        mem.vectors.add([vec], [payload], [memory_id])
        
        # Add to Entity Index for quicker lookup
        mem.entities.add_memory_entities(memory_id, text_line, [src.capitalize(), dst.capitalize()])
        
        # Also need to ensure the user ID is known if used in queries
        # But for this test, we might not rely on user_id context heavily
        
    except Exception as e:
        print(f"❌ Error storing '{text_line}': {e}")
        import traceback
        traceback.print_exc()

# ============================================================================
# MAIN TEST LOGIC
# ============================================================================
def main():
    print("🚀 Starting Memory Accuracy Test...")
    
    # 1. Initialize Memory System
    # We use a temp config to avoid messing with real user data if possible, 
    # but here we'll just use a test user_id namespace.
    
    # Check if config exists, else mock it
    config_path = "configs/offline.yaml"
    if not os.path.exists(config_path) and os.path.exists("Amem/local_config.py"):
         # Try to rely on defaults if config file missing, but typically we need one.
         pass

    # Create Memory instance
    # We assume defaults work or config file is present at relative path
    class Args:
        config = config_path
        llm_backend = "none"
        max_new_tokens = 100
        temperature = 0.1
        top_p = 0.9
        index_type = "FLAT" # Precise for small test
        llm_model = None
        embed_model = None
        faiss_path = "test_memory_data" # Separate DB for testing
        reranker_top_k = 10
        debug = False
        traverse_debug = False

    args = Args()
    
    # Ensure test dir exists/clean
    if os.path.exists(args.faiss_path):
        import shutil
        try:
            shutil.rmtree(args.faiss_path)
            print(f"🧹 Cleaned up existing test data at {args.faiss_path}")
        except Exception as e:
            print(f"⚠️ Could not clean test data: {e}")
            
    if not os.path.exists(args.faiss_path):
        os.makedirs(args.faiss_path)
    
    # Hack: Monkey patch to force new DB/Graph for test isolation if desired
    # For now, we'll just rely on separate user_id "TEST_USER" to filter? 
    # But graph is shared. Better to use separate vector_store path.
    # The OfflineMemoryConfig loading might override paths. 
    # Let's trust the user wants to test on the current system or a safe one.
    # The prompt implied "create a python file Which goes through integrated_system.py",
    # so we should use the standard initialization logic.
    
    # Override config vector store path to avoid corrupting main memory?
    # The prompt inputs 50 memories. If we add them to main DB, we pollute it.
    # Standard practice: Use a test path.
    
    print("📦 Building Memory System (Backend: NONE)...")
    
    # We have to build it manually to inject overrides since build_memory loads from file
    cfg = OfflineMemoryConfig.from_file(args.config)
    cfg.vector_store.path = args.faiss_path # Override storage
    cfg.graph.db_path = os.path.join(args.faiss_path, "graph.db") # Override graph
    cfg.history_db_path = os.path.join(args.faiss_path, "history.db") # Override history
    cfg.llm.backend = "none"
    
    mem = OfflineMemory(cfg)
    
    # 2. Populate Memories
    print(f"\n📥 Storing {len(MEMORIES)} memories (Direct Injection)...")
    try:
        # Clear existing test data first?
        # mem.vectors.reset? Not easily available. 
        # We'll just assume fresh directory or append.
        pass
    except:
        pass

    for i, line in enumerate(MEMORIES):
        store_memory_direct(mem, line, user_id="tester")
        if (i+1) % 10 == 0:
            print(f"   Stored {i+1}/{len(MEMORIES)}...")
    
    print("✅ Storage Complete.")
    
    # Helper to check partial matches
    def check_result(results_text, expected_list):
        lower_res = results_text.lower()
        # It's a match if ANY of the expected keywords appear? 
        # Or ALL? Or majority? 
        # For "Who likes black?", expected "Abhishek".
        # If result contains "Abhishek", it's correct.
        for ans in expected_list:
            if ans.lower() in lower_res:
                return True
        return False

    # 3. Phase 1: No @remember (Graph Only)
    print("\n🔍 PHASE 1: Retrieval WITHOUT @remember (Graph Only)")
    p1_score = 0
    p1_results = []
    
    for i, item in enumerate(QUERIES):
        q = item["q"]
        gold = item["a"]
        
        # Use graph_only_search
        # Note: integrated_system.graph_only_search uses mem, query, user_id
        res = graph_only_search(mem, q, user_id="tester", limit=5)
        
        # Extract text from results
        # results structure: {"results": [{"memory": "...", "score": ...}], ...}
        found_texts = [r.get("memory", "") for r in res.get("results", [])]
        found_blob = " ".join(found_texts)
        
        is_correct = check_result(found_blob, gold)
        if is_correct:
            p1_score += 1
        
        p1_results.append({
            "id": i+1,
            "q": q,
            "correct": is_correct,
            "gold": gold,
            "found": found_texts[:2] # Top 2 for brevity
        })
        print(f"   Q{i+1}: {'✅' if is_correct else '❌'} | {q}")
        for k, ft in enumerate(found_texts[:2]):
             print(f"      [{k+1}] {ft}")

    # 4. Phase 2: With @remember (Unified Search)
    print("\n🧠 PHASE 2: Retrieval WITH @remember (Unified Search)")
    p2_score = 0
    p2_results = []
    
    for i, item in enumerate(QUERIES):
        q = item["q"]
        gold = item["a"]
        
        # if i >= 5: break # LIMIT TO 5 FOR DEBUG
        
        # Use unified_memory_search
        res = unified_memory_search(mem, q, user_id="tester", limit=5, debug=False)
        
        found_texts = [r.get("memory", "") for r in res.get("results", [])]
        found_blob = " ".join(found_texts)
        
        is_correct = check_result(found_blob, gold)
        if is_correct:
            p2_score += 1
            
        p2_results.append({
            "id": i+1,
            "q": q,
            "correct": is_correct,
            "gold": gold,
            "found": found_texts[:2]
        })
        print(f"   Q{i+1}: {'✅' if is_correct else '❌'} | {q}")
        for k, ft in enumerate(found_texts[:2]):
             print(f"      [{k+1}] {ft}")

    # 5. Report
    print("\n" + "="*60)
    print("📊 ACCURACY REPORT")
    print("="*60)
    print(f"Phase 1 (Graph Only): {p1_score}/{len(QUERIES)} ({p1_score/len(QUERIES)*100:.1f}%)")
    print(f"Phase 2 (Unified):    {p2_score}/{len(QUERIES)} ({p2_score/len(QUERIES)*100:.1f}%)")
    print("="*60)
    
    # Detailed failure log
    print("\n❌ MISSED QUERIES (Phase 2):")
    failures = []
    for r in p2_results:
        if not r["correct"]:
            print(f"Q{r['id']}: {r['q']}")
            print(f"   Expected: {r['gold']}")
            print(f"   Got Top-2: {r['found']}")
            failures.append({
                "id": r["id"],
                "q": r["q"],
                "expected": r["gold"],
                "got": r["found"]
            })
    
    with open("failures.json", "w", encoding="utf-8") as f:
        json.dump(failures, f, indent=2)
    print(f"\nSaved {len(failures)} failures to failures.json")

if __name__ == "__main__":
    main()
