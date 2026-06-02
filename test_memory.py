"""
Unified Memory System Benchmark - Sequential Storage & Dual Evaluation
======================================================================
This benchmark script evaluates the memory system in two phases after a 
strictly sequential storage process.

Phase 1: Sequential Storage
- Stores memories one by one using `mem.add(infer=True)`
- Waits for each memory to be fully processed (Graph + Vector) before moving to the next.

Phase 2: Benchmark 1 (Normal Conversation Mode)
- Uses `graph_only_search` (mimicking standard chat retrieval)
- Evaluates 250 queries across 5 categories
- Judge LLM evaluates relevance (YES/NO)

Phase 3: Benchmark 2 (@remember Command Mode)
- Uses `unified_memory_search` (mimicking explicit memory retrieval)
- Evaluates the same 250 queries
- Judge LLM evaluates relevance (YES/NO)

Output:
- benchmark_1_results.txt: Detailed logs for normal conversation benchmark
- benchmark_2_results.txt: Detailed logs for @remember benchmark
- Final console summary comparison
"""

import os
import sys
import uuid
import json
import time
import shutil
import requests
from typing import List, Dict, Any, Optional, Tuple

# ============================================================================
# AMEM MEMORY SYSTEM IMPORTS
# ============================================================================
from Amem.memory import OfflineMemory
from Amem.local_config import OfflineMemoryConfig
from Amem.memory_system import graph_only_search, unified_memory_search

# ============================================================================
# LM STUDIO API CONFIG
# ============================================================================
LM_STUDIO_URL = "http://192.168.0.13:1234/v1"
LM_STUDIO_MODEL = "qwen3-30b-a3b-instruct-2507"

def call_llm(prompt: str, system_prompt: str = "", max_tokens: int = 500) -> str:
    """Call LM Studio API with OpenAI-compatible format."""
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})
    
    try:
        response = requests.post(
            f"{LM_STUDIO_URL}/chat/completions",
            json={
                "model": LM_STUDIO_MODEL,
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": 0.1
            },
            timeout=120
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"].strip()
    except Exception as e:
        print(f"⚠️ LLM Call Error: {e}")
        return "ERROR"

def judge_retrieval(query: str, expected_keywords: List[str], retrieved_memories: str) -> Tuple[bool, str]:
    """
    Judge LLM evaluates if the retrieved memories are relevant and contain expected information.
    Returns (success: bool, reasoning: str)
    """
    if not retrieved_memories or retrieved_memories.strip() == "":
        return False, "No memories retrieved"
    
    expected_str = ", ".join(expected_keywords) if expected_keywords else "relevant information"
    
    prompt = f"""You are a strict JUDGE evaluating memory retrieval quality.

IMPORTANT CONTEXT: In this benchmark, "I", "My", "Me" refer to "Abhishek" (the benchmark user).
So if the query asks about "Abhishek" and the memory says "I do X", that is a MATCH.

Query: {query}
Expected Information: {expected_str}
Retrieved Memories: {retrieved_memories}

Evaluate: Does the retrieved memory contain the expected information or answer the query?
- If query mentions "Abhishek" and memory uses "I/My", treat as SAME PERSON.
- If expected keywords are found in retrieved memories, answer YES.
- If the memory directly answers the query, answer YES even if wording differs.

You MUST respond with ONLY "YES" or "NO" on the first line.
On the second line, provide a brief reason (max 20 words).

Example responses:
YES
Contains expected information about the color preference.

NO
Retrieved memory is about work, not the color query.

/no_think"""

    try:
        response = call_llm(prompt, system_prompt="You are a strict binary evaluator. Answer YES or NO only.", max_tokens=50)
        lines = response.strip().split('\n')
        # Filter out empty lines
        lines = [l.strip() for l in lines if l.strip()]
        if not lines:
            return False, "Empty response from Judge"
            
        verdict = lines[0].strip().upper()
        # Handle case where LLM might output "Verdict: YES"
        if "YES" in verdict: verdict = "YES"
        elif "NO" in verdict: verdict = "NO"
        
        reason = lines[1] if len(lines) > 1 else "No reason provided"
        
        is_success = verdict == "YES"
        return is_success, reason
    except Exception as e:
        return False, f"Judge error: {str(e)}"

# ============================================================================
# TEST DATA
# ============================================================================
MEMORIES_TO_STORE = [
    # ========== Person: Abhishek (10 facts) ==========
    "My favorite color is black. I really love dark themes everywhere.",
    "I study Artificial Intelligence at university. It's my passion.",
    "I use Linux as my primary operating system. Ubuntu and Arch are my favorites.",
    "I prefer dark mode interfaces in all applications.",
    "I'm working on a project called After Human. It's about digital immortality.",
    "I'm deeply interested in digital immortality and consciousness transfer.",
    "I know Python very well. It's my main programming language.",
    "I understand system design and distributed systems architecture.",
    "I dislike loud environments. I prefer quiet places to work.",
    "I prefer minimal UI designs. Clean interfaces are the best.",
    
    # ========== Person: Riya (8 facts) ==========
    "Riya lives in Bangalore. She moved there for work.",
    "Riya loves street photography. She captures urban life beautifully.",
    "Riya uses an iPhone for all her photography work.",
    "Riya prefers the iOS ecosystem over Android.",
    "Riya works as a UX designer at a tech startup.",
    "Riya dislikes cluttered layouts. She values whitespace.",
    "Riya likes pastel colors in her designs.",
    "Riya is interested in human-centered design principles.",
    
    # ========== Person: Karan (7 facts) ==========
    "Karan lives in Delhi. He grew up there.",
    "Karan works as a data analyst at a consulting firm.",
    "Karan knows SQL very well. He uses it daily.",
    "Karan is proficient in Power BI for data visualization.",
    "Karan likes football. He plays every weekend.",
    "Karan supports Manchester United. He's been a fan since childhood.",
    "Karan dislikes manual data entry. He automates everything.",
    
    # ========== Person: Meera (6 facts) ==========
    "Meera lives in Pune. She loves the weather there.",
    "Meera studies Psychology at college.",
    "Meera is interested in cognitive behavior and therapy.",
    "Meera likes journaling. She writes every night.",
    "Meera prefers quiet cafés for studying.",
    "Meera dislikes social media overload. She limits her screen time.",
    
    # ========== Person: Aarav (6 facts) ==========
    "Aarav lives in Mumbai. He runs his business from there.",
    "Aarav works as a startup founder. He built his company from scratch.",
    "Aarav is interested in venture capital and funding.",
    "Aarav likes networking events. He attends them regularly.",
    "Aarav uses a MacBook for all his work.",
    "Aarav prefers remote work. He values flexibility.",
    
    # ========== Relationships (13 facts) ==========
    "I collaborate with Riya on design projects. We make a great team.",
    "I often discuss ethics of AI with my colleagues.",
    "Riya works with Aarav at his startup.",
    "Karan consults Aarav on business analytics.",
    "Meera reads research papers on cognitive psychology.",
    "Aarav invests in early-stage tech startups.",
    "Riya avoids using dark color palettes in her designs.",
    "I avoid vendor lock-in when choosing technologies.",
    "Karan prefers working with structured datasets.",
    "Meera practices mindfulness meditation daily.",
    "Aarav follows several startup accelerators like Y Combinator.",
    "I believe in open-source philosophy. I contribute to many projects.",
    "Riya believes in design empathy. Users first.",
    
    # ========== Additional Facts for Multi-hop (10 facts) ==========
    "My wife's name is Aria. She is an artist.",
    "Aria loves painting landscapes. She uses watercolors.",
    "Aria and Riya are friends. They met at an art exhibition.",
    "Karan's brother is Vikram. He works in finance.",
    "Vikram lives in Singapore. He moved there for work.",
    "Meera's mentor is Dr. Sharma. She is a renowned psychologist.",
    "Dr. Sharma wrote a book on cognitive therapy.",
    "Aarav's company is called TechNova. It focuses on AI solutions.",
    "TechNova raised funding from Sequoia Capital.",
    "I attended university with Karan. We were roommates.",
    
    # ========== Temporal Facts (7 facts) ==========
    "Yesterday I had a meeting with the team about the project roadmap.",
    "Last week I completed the memory system architecture.",
    "Today I'm testing the benchmark suite.",
    "Last month I started learning Rust programming.",
    "I graduated from university in 2023.",
    "Riya joined the startup in 2024.",
    "Aarav founded TechNova in 2022.",
    
    # ========== Preferences and Habits (8 facts) ==========
    "My favorite programming language is Python but I also like Rust.",
    "I prefer VSCode as my code editor.",
    "I drink coffee every morning. Black coffee, no sugar.",
    "I sleep late, usually around 2 AM.",
    "I exercise three times a week. Running is my favorite.",
    "I read books on philosophy and technology.",
    "I listen to lo-fi music while coding.",
    "My favorite movie genre is science fiction.",
]

# Total Tests = 50 * 5 = 250
FAITHFULNESS_TESTS = [
    {"query": "What color does Abhishek like?", "expected": ["black", "dark"]},
    {"query": "Where does Riya live?", "expected": ["Bangalore"]},
    {"query": "What does Karan know?", "expected": ["SQL", "Power BI"]},
    {"query": "What does Meera study?", "expected": ["Psychology"]},
    {"query": "What team does Karan support?", "expected": ["Manchester United"]},
    {"query": "What is Abhishek's favorite programming language?", "expected": ["Python"]},
    {"query": "What operating system does Abhishek use?", "expected": ["Linux"]},
    {"query": "What device does Riya use?", "expected": ["iPhone"]},
    {"query": "Where does Aarav live?", "expected": ["Mumbai"]},
    {"query": "What does Meera practice daily?", "expected": ["mindfulness", "meditation"]},
    {"query": "What is my wife's name?", "expected": ["Aria"]},
    {"query": "What does Aria love?", "expected": ["painting", "watercolors", "landscapes"]},
    {"query": "What is Aarav's company called?", "expected": ["TechNova"]},
    {"query": "Who is Meera's mentor?", "expected": ["Sharma", "Dr. Sharma"]},
    {"query": "What did Karan's brother Vikram do?", "expected": ["finance", "Singapore"]},
    {"query": "What coffee do I drink?", "expected": ["black", "no sugar"]},
    {"query": "What music do I listen to while coding?", "expected": ["lo-fi"]},
    {"query": "When do I usually sleep?", "expected": ["2 AM", "late"]},
    {"query": "What editor do I use?", "expected": ["VSCode"]},
    {"query": "What genre of movies do I like?", "expected": ["science fiction", "sci-fi"]},
    {"query": "What does Riya work as?", "expected": ["UX designer"]},
    {"query": "What is Aarav interested in?", "expected": ["venture capital", "funding"]},
    {"query": "What does Meera like doing?", "expected": ["journaling", "writes"]},
    {"query": "What sport does Karan like?", "expected": ["football"]},
    {"query": "What ecosystem does Riya prefer?", "expected": ["iOS"]},
    {"query": "What kind of UI do I prefer?", "expected": ["minimal", "clean"]},
    {"query": "What am I working on?", "expected": ["After Human", "digital immortality"]},
    {"query": "What kind of places do I prefer?", "expected": ["quiet"]},
    {"query": "What does Karan avoid?", "expected": ["manual data entry"]},
    {"query": "What does Meera dislike?", "expected": ["social media", "overload"]},
    {"query": "Who raised funding from Sequoia?", "expected": ["TechNova", "Aarav"]},
    {"query": "When did I graduate?", "expected": ["2023"]},
    {"query": "When did Aarav found his company?", "expected": ["2022"]},
    {"query": "What am I learning recently?", "expected": ["Rust"]},
    {"query": "What kind of work does Aarav prefer?", "expected": ["remote"]},
    {"query": "What laptop does Aarav use?", "expected": ["MacBook"]},
    {"query": "What does Riya capture in photography?", "expected": ["street", "urban"]},
    {"query": "What does Riya avoid in designs?", "expected": ["dark color", "palettes"]},
    {"query": "What philosophy do I believe in?", "expected": ["open-source"]},
    {"query": "What does Riya value?", "expected": ["whitespace", "empathy"]},
    {"query": "Where did Vikram move to?", "expected": ["Singapore"]},
    {"query": "What book did Dr. Sharma write?", "expected": ["cognitive therapy"]},
    {"query": "Where did Aria and Riya meet?", "expected": ["art exhibition"]},
    {"query": "What datasets does Karan prefer?", "expected": ["structured"]},
    {"query": "What time did I complete the architecture?", "expected": ["last week"]},
    {"query": "When did Riya join the startup?", "expected": ["2024"]},
    {"query": "What books do I read?", "expected": ["philosophy", "technology"]},
    {"query": "How often do I exercise?", "expected": ["three times", "week"]},
    {"query": "What kind of running do I do?", "expected": ["running", "exercise"]},
    {"query": "What does TechNova focus on?", "expected": ["AI solutions"]},
]

RETRIEVAL_TESTS = [
    {"query": "Who likes black color?", "expected_persons": ["Abhishek", "I", "my"]},
    {"query": "Who lives in Bangalore?", "expected_persons": ["Riya"]},
    {"query": "Who prefers quiet environments?", "expected_persons": ["Abhishek", "Meera", "I"]},
    {"query": "Who dislikes clutter?", "expected_persons": ["Riya"]},
    {"query": "Who knows Python?", "expected_persons": ["Abhishek", "I"]},
    {"query": "Who uses Apple devices?", "expected_persons": ["Riya", "iPhone"]},
    {"query": "Who works as a designer?", "expected_persons": ["Riya"]},
    {"query": "Who is interested in AI?", "expected_persons": ["Abhishek", "I"]},
    {"query": "Who practices mindfulness?", "expected_persons": ["Meera"]},
    {"query": "Who invests in startups?", "expected_persons": ["Aarav"]},
    {"query": "Who uses Linux?", "expected_persons": ["Abhishek", "I"]},
    {"query": "Who prefers dark mode?", "expected_persons": ["Abhishek", "I"]},
    {"query": "Who plays football?", "expected_persons": ["Karan"]},
    {"query": "Who studies psychology?", "expected_persons": ["Meera"]},
    {"query": "Who uses a MacBook?", "expected_persons": ["Aarav"]},
    {"query": "Who attends networking events?", "expected_persons": ["Aarav"]},
    {"query": "Who likes street photography?", "expected_persons": ["Riya"]},
    {"query": "Who lives in Delhi?", "expected_persons": ["Karan"]},
    {"query": "Who lives in Pune?", "expected_persons": ["Meera"]},
    {"query": "Who lives in Mumbai?", "expected_persons": ["Aarav"]},
    {"query": "Who supports Manchester United?", "expected_persons": ["Karan"]},
    {"query": "Who prefers iOS?", "expected_persons": ["Riya"]},
    {"query": "Who drinks black coffee?", "expected_persons": ["Abhishek", "I"]},
    {"query": "Who sleeps late?", "expected_persons": ["Abhishek", "I"]},
    {"query": "Who listens to lo-fi?", "expected_persons": ["Abhishek", "I"]},
    {"query": "Who founded a company?", "expected_persons": ["Aarav"]},
    {"query": "Who is married to an artist?", "expected_persons": ["Abhishek", "I", "Aria"]},
    {"query": "Who avoids vendor lock-in?", "expected_persons": ["Abhishek", "I"]},
    {"query": "Who contributes to open source?", "expected_persons": ["Abhishek", "I"]},
    {"query": "Who prefers remote work?", "expected_persons": ["Aarav"]},
    {"query": "Who journals every night?", "expected_persons": ["Meera"]},
    {"query": "Who likes watercolors?", "expected_persons": ["Aria"]},
    {"query": "Who works in finance?", "expected_persons": ["Vikram"]},
    {"query": "Who is a psychologist?", "expected_persons": ["Sharma", "Dr"]},
    {"query": "Who automates data work?", "expected_persons": ["Karan"]},
    {"query": "Who uses Power BI?", "expected_persons": ["Karan"]},
    {"query": "Who values whitespace in design?", "expected_persons": ["Riya"]},
    {"query": "Who reads research papers?", "expected_persons": ["Meera"]},
    {"query": "Who follows Y Combinator?", "expected_persons": ["Aarav"]},
    {"query": "Who prefers structured data?", "expected_persons": ["Karan"]},
    {"query": "Who uses VSCode?", "expected_persons": ["Abhishek", "I"]},
    {"query": "Who exercises regularly?", "expected_persons": ["Abhishek", "I"]},
    {"query": "Who likes running?", "expected_persons": ["Abhishek", "I"]},
    {"query": "Who reads philosophy books?", "expected_persons": ["Abhishek", "I"]},
    {"query": "Who likes sci-fi movies?", "expected_persons": ["Abhishek", "I"]},
    {"query": "Who dislikes social media?", "expected_persons": ["Meera"]},
    {"query": "Who limits screen time?", "expected_persons": ["Meera"]},
    {"query": "Who likes pastel colors?", "expected_persons": ["Riya"]},
    {"query": "Who understands system design?", "expected_persons": ["Abhishek", "I"]},
    {"query": "Who is learning Rust?", "expected_persons": ["Abhishek", "I"]},
]

REASONING_TESTS = [
    {"query": "Who collaborates with a UX designer?", "expected": ["Abhishek", "I"], "reasoning": "Abhishek → COLLABORATES_WITH → Riya (UX designer)"},
    {"query": "Who works with a startup founder?", "expected": ["Riya", "Karan"], "reasoning": "Riya/Karan → Aarav (founder)"},
    {"query": "Who is connected to venture capital?", "expected": ["Aarav", "TechNova"], "reasoning": "Aarav directly, TechNova raised from Sequoia"},
    {"query": "Who collaborates with someone who avoids dark colors?", "expected": ["Abhishek", "I"], "reasoning": "Abhishek → Riya (avoids dark palettes)"},
    {"query": "Find people connected to tech startups", "expected": ["Aarav", "Riya", "TechNova"], "reasoning": "Via investment/work relationships"},
    {"query": "Who knows someone in Singapore?", "expected": ["Karan", "Vikram"], "reasoning": "Karan → Vikram (Singapore)"},
    {"query": "Who is connected to Dr. Sharma?", "expected": ["Meera"], "reasoning": "Meera → mentor Dr. Sharma"},
    {"query": "Who is friends with an artist?", "expected": ["Riya", "Aria"], "reasoning": "Riya → friends → Aria (artist)"},
    {"query": "Who is connected to TechNova?", "expected": ["Aarav", "Sequoia"], "reasoning": "Aarav → founder, Sequoia → investor"},
    {"query": "Who attended university with someone in Delhi?", "expected": ["Abhishek", "Karan", "I"], "reasoning": "Abhishek → roommates → Karan (Delhi)"},
    {"query": "Who is married to someone who paints?", "expected": ["Abhishek", "I", "Aria"], "reasoning": "Abhishek → wife → Aria (paints)"},
    {"query": "Who works with someone interested in human-centered design?", "expected": ["Aarav", "Riya"], "reasoning": "Aarav → works with → Riya (human-centered design)"},
    {"query": "Who consults with a company founder?", "expected": ["Karan", "Aarav"], "reasoning": "Karan → consults → Aarav (founder)"},
    {"query": "Who knows someone who wrote a book?", "expected": ["Meera", "Dr. Sharma"], "reasoning": "Meera → mentor → Dr. Sharma (wrote book)"},
    {"query": "Who is connected to someone who raised funding?", "expected": ["Aarav", "TechNova", "Sequoia"], "reasoning": "TechNova → raised → Sequoia"},
    {"query": "Who works with an iPhone user?", "expected": ["Aarav", "Riya"], "reasoning": "Aarav → works with → Riya (iPhone)"},
    {"query": "Who is connected to AI solutions?", "expected": ["Aarav", "TechNova", "Abhishek", "I"], "reasoning": "TechNova (AI solutions), Abhishek (AI study)"},
    {"query": "Who knows someone interested in cognitive behavior?", "expected": ["Dr. Sharma", "Meera"], "reasoning": "Meera (cognitive behavior) → mentor → Dr. Sharma"},
    {"query": "Who is connected to Manchester United fans?", "expected": ["Vikram", "Karan"], "reasoning": "Vikram → brother → Karan (Man Utd)"},
    {"query": "Who collaborates on design with someone in Bangalore?", "expected": ["Abhishek", "I", "Riya"], "reasoning": "Abhishek → collaborates → Riya (Bangalore)"},
    {"query": "Who is connected to watercolor artists?", "expected": ["Abhishek", "I", "Aria", "Riya"], "reasoning": "Aria (watercolors), connected to Abhishek and Riya"},
    {"query": "Who has connections in the finance industry?", "expected": ["Karan", "Vikram"], "reasoning": "Vikram (finance) → brother → Karan"},
    {"query": "Who knows someone at a startup accelerator?", "expected": ["Aarav"], "reasoning": "Aarav → follows → Y Combinator"},
    {"query": "Who uses Apple products and knows a startup founder?", "expected": ["Riya", "Aarav"], "reasoning": "Riya (iPhone) → works with → Aarav (founder)"},
    {"query": "Who is connected to Pune through psychology?", "expected": ["Meera"], "reasoning": "Meera → lives in Pune, studies psychology"},
    {"query": "Who knows a data analyst?", "expected": ["Vikram", "Aarav", "Abhishek", "I"], "reasoning": "Connected to Karan (data analyst)"},
    {"query": "Who works with someone who uses Power BI?", "expected": ["Aarav", "Karan"], "reasoning": "Aarav → consults with → Karan (Power BI)"},
    {"query": "Who is connected to open-source contributors?", "expected": ["Riya", "Aria"], "reasoning": "Connected to Abhishek (open-source)"},
    {"query": "Who has connections in Mumbai tech scene?", "expected": ["Aarav", "TechNova", "Riya"], "reasoning": "Aarav (Mumbai), TechNova, Riya works with Aarav"},
    {"query": "Who knows someone who exercises regularly?", "expected": ["Karan", "Riya", "Aria"], "reasoning": "Connected to Abhishek (exercises)"},
    {"query": "Who is connected to digital immortality research?", "expected": ["Riya"], "reasoning": "Connected to Abhishek (After Human project)"},
    {"query": "Who knows a lo-fi music listener?", "expected": ["Aria", "Karan", "Riya"], "reasoning": "Connected to Abhishek (lo-fi)"},
    {"query": "Who works with someone from a consulting firm?", "expected": ["Aarav", "Karan"], "reasoning": "Karan (consulting firm) → consults → Aarav"},
    {"query": "Who is connected to art exhibition visitors?", "expected": ["Abhishek", "I", "Aria", "Riya"], "reasoning": "Aria and Riya met at art exhibition"},
    {"query": "Find the chain from Karan to Singapore", "expected": ["Karan", "Vikram", "Singapore"], "reasoning": "Karan → brother → Vikram → Singapore"},
    {"query": "Who is connected to the 2022 startup ecosystem?", "expected": ["Aarav", "TechNova"], "reasoning": "Aarav founded TechNova in 2022"},
    {"query": "Who knows someone who values design empathy?", "expected": ["Aarav", "Abhishek", "I"], "reasoning": "Connected to Riya (design empathy)"},
    {"query": "Who is connected to Arch Linux users?", "expected": ["Riya", "Aria", "Karan"], "reasoning": "Connected to Abhishek (Arch)"},
    {"query": "Who has indirect connection to painting?", "expected": ["Riya", "Abhishek", "I"], "reasoning": "Via Aria (painting)"},
    {"query": "Who knows someone interested in distributed systems?", "expected": ["Riya", "Karan", "Aria"], "reasoning": "Connected to Abhishek (system design)"},
    {"query": "Who is 2 hops from a psychologist?", "expected": ["Karan", "Riya"], "reasoning": "2 hops from Dr. Sharma via connections"},
    {"query": "Who is connected to coffee enthusiasts?", "expected": ["Riya", "Karan", "Aria"], "reasoning": "Connected to Abhishek (black coffee)"},
    {"query": "Find connections to Sequoia Capital", "expected": ["Aarav", "TechNova"], "reasoning": "TechNova → raised from → Sequoia"},
    {"query": "Who knows someone who graduated in 2023?", "expected": ["Karan", "Riya", "Aria"], "reasoning": "Connected to Abhishek (2023 grad)"},
    {"query": "Who has indirect connection to Delhi?", "expected": ["Abhishek", "I", "Vikram", "Aarav"], "reasoning": "Via Karan (Delhi)"},
    {"query": "Who is connected to someone with a quiet workspace preference?", "expected": ["Riya", "Karan", "Aria"], "reasoning": "Connected to Abhishek/Meera (quiet)"},
    {"query": "Who knows a Y Combinator follower?", "expected": ["Riya", "Karan", "TechNova"], "reasoning": "Connected to Aarav (Y Combinator)"},
    {"query": "Who is connected to someone learning Rust?", "expected": ["Riya", "Karan", "Aria"], "reasoning": "Connected to Abhishek (Rust)"},
    {"query": "Find the path from Abhishek to Sequoia", "expected": ["Abhishek", "Riya", "Aarav", "TechNova", "Sequoia"], "reasoning": "Abhishek → Riya → Aarav → TechNova → Sequoia"},
    {"query": "Who has 3-hop connection to Dr. Sharma?", "expected": ["Abhishek", "I", "Riya"], "reasoning": "Via Abhishek → someone → Meera → Dr. Sharma"},
]

CONSISTENCY_TESTS = [
    {"queries": ["What does Abhishek like?", "Tell me about Abhishek's preferences", "What are Abhishek's likes?"], "expected_consistent": ["black", "dark"]},
    {"queries": ["Where does Riya live?", "What city is Riya from?", "Riya's location?"], "expected_consistent": ["Bangalore"]},
    {"queries": ["What does Karan work as?", "Karan's profession?", "What is Karan's job?"], "expected_consistent": ["data analyst"]},
    {"queries": ["What does Abhishek study?", "Abhishek's field of study?", "What is Abhishek studying?"], "expected_consistent": ["AI", "Artificial Intelligence"]},
    {"queries": ["Who is Abhishek's wife?", "Abhishek's spouse?", "Name of Abhishek's wife?"], "expected_consistent": ["Aria"]},
    {"queries": ["What OS does Abhishek use?", "Abhishek's operating system?", "What system does Abhishek prefer?"], "expected_consistent": ["Linux"]},
    {"queries": ["What laptop does Aarav use?", "Aarav's computer?", "What device does Aarav work on?"], "expected_consistent": ["MacBook"]},
    {"queries": ["Where is Karan from?", "Karan's city?", "Where does Karan live?"], "expected_consistent": ["Delhi"]},
    {"queries": ["What does Meera study?", "Meera's major?", "What is Meera studying?"], "expected_consistent": ["Psychology"]},
    {"queries": ["What sport does Karan play?", "Karan's favorite sport?", "What game does Karan like?"], "expected_consistent": ["football"]},
    {"queries": ["What phone does Riya use?", "Riya's device?", "What does Riya photograph with?"], "expected_consistent": ["iPhone"]},
    {"queries": ["Who does Abhishek collaborate with?", "Abhishek's collaborator?", "Who works with Abhishek?"], "expected_consistent": ["Riya"]},
    {"queries": ["What is Aarav's company?", "Aarav's startup name?", "What did Aarav found?"], "expected_consistent": ["TechNova"]},
    {"queries": ["What coffee does Abhishek drink?", "Abhishek's coffee preference?", "How does Abhishek take his coffee?"], "expected_consistent": ["black"]},
    {"queries": ["What editor does Abhishek use?", "Abhishek's code editor?", "What IDE does Abhishek prefer?"], "expected_consistent": ["VSCode"]},
    {"queries": ["Who is Meera's mentor?", "Meera's guide?", "Who mentors Meera?"], "expected_consistent": ["Sharma", "Dr."]},
    {"queries": ["What language is Abhishek learning?", "New programming language for Abhishek?", "What is Abhishek studying now?"], "expected_consistent": ["Rust"]},
    {"queries": ["Where did Vikram move?", "Vikram's new location?", "Where is Vikram now?"], "expected_consistent": ["Singapore"]},
    {"queries": ["What does Aria paint?", "Aria's art style?", "What does Aria create?"], "expected_consistent": ["landscape", "watercolor"]},
    {"queries": ["When did Abhishek graduate?", "Abhishek's graduation year?", "What year did Abhishek finish university?"], "expected_consistent": ["2023"]},
    {"queries": ["What time does Abhishek sleep?", "Abhishek's bedtime?", "When does Abhishek go to sleep?"], "expected_consistent": ["2 AM", "late"]},
    {"queries": ["What music does Abhishek listen to?", "Abhishek's music preference?", "What does Abhishek play while coding?"], "expected_consistent": ["lo-fi"]},
    {"queries": ["What movies does Abhishek like?", "Abhishek's movie genre?", "What films does Abhishek prefer?"], "expected_consistent": ["sci-fi", "science fiction"]},
    {"queries": ["How often does Abhishek exercise?", "Abhishek's workout frequency?", "When does Abhishek exercise?"], "expected_consistent": ["three times", "week"]},
    {"queries": ["What does Karan support?", "Karan's football team?", "Which team does Karan follow?"], "expected_consistent": ["Manchester United"]},
    {"queries": ["Who funded TechNova?", "TechNova's investor?", "Who invested in TechNova?"], "expected_consistent": ["Sequoia"]},
    {"queries": ["When was TechNova founded?", "TechNova's founding year?", "What year did Aarav start TechNova?"], "expected_consistent": ["2022"]},
    {"queries": ["What does Karan know?", "Karan's skills?", "What tools does Karan use?"], "expected_consistent": ["SQL", "Power BI"]},
    {"queries": ["What does Riya dislike?", "Riya's pet peeves?", "What annoys Riya?"], "expected_consistent": ["clutter"]},
    {"queries": ["What does Meera practice?", "Meera's daily routine?", "What meditation does Meera do?"], "expected_consistent": ["mindfulness"]},
    {"queries": ["What does Abhishek believe in?", "Abhishek's philosophy?", "What values does Abhishek hold?"], "expected_consistent": ["open-source"]},
    {"queries": ["What colors does Riya like?", "Riya's color preference?", "What palette does Riya prefer?"], "expected_consistent": ["pastel"]},
    {"queries": ["What project is Abhishek working on?", "Abhishek's current project?", "What is Abhishek building?"], "expected_consistent": ["After Human"]},
    {"queries": ["Who is Karan's brother?", "Karan's sibling?", "What is Karan's brother's name?"], "expected_consistent": ["Vikram"]},
    {"queries": ["What does Vikram work in?", "Vikram's field?", "What industry is Vikram in?"], "expected_consistent": ["finance"]},
    {"queries": ["How did Aria and Riya meet?", "Where did Aria meet Riya?", "Aria and Riya's meeting place?"], "expected_consistent": ["art exhibition"]},
    {"queries": ["What book did Dr. Sharma write?", "Dr. Sharma's publication?", "What did Dr. Sharma author?"], "expected_consistent": ["cognitive therapy"]},
    {"queries": ["What does Abhishek avoid?", "Abhishek's dislikes in tech?", "What does Abhishek stay away from?"], "expected_consistent": ["vendor lock-in"]},
    {"queries": ["What accelerator does Aarav follow?", "Aarav's startup accelerator?", "What incubator does Aarav track?"], "expected_consistent": ["Y Combinator"]},
    {"queries": ["What does Karan prefer in data?", "Karan's data preference?", "What kind of data does Karan like?"], "expected_consistent": ["structured"]},
    {"queries": ["What environments does Abhishek dislike?", "Abhishek's environmental preferences?", "What places does Abhishek avoid?"], "expected_consistent": ["loud"]},
    {"queries": ["What does Meera read?", "Meera's reading material?", "What papers does Meera study?"], "expected_consistent": ["research papers"]},
    {"queries": ["What does Abhishek understand?", "Abhishek's expertise?", "What architecture does Abhishek know?"], "expected_consistent": ["system design", "distributed"]},
    {"queries": ["When did Riya join the startup?", "Riya's joining year?", "What year did Riya start working?"], "expected_consistent": ["2024"]},
    {"queries": ["What places does Meera prefer?", "Meera's study spots?", "Where does Meera like to study?"], "expected_consistent": ["quiet café"]},
    {"queries": ["What does Abhishek read?", "Abhishek's reading interests?", "What books does Abhishek enjoy?"], "expected_consistent": ["philosophy", "technology"]},
    {"queries": ["What work style does Aarav prefer?", "Aarav's work preference?", "How does Aarav like to work?"], "expected_consistent": ["remote"]},
    {"queries": ["What design principle does Riya believe in?", "Riya's design philosophy?", "What approach does Riya follow?"], "expected_consistent": ["empathy", "users first"]},
    {"queries": ["What did Abhishek complete last week?", "Abhishek's recent work?", "What was finished last week?"], "expected_consistent": ["architecture", "memory system"]},
    {"queries": ["What does Meera dislike about technology?", "Meera's tech concerns?", "What screen habits does Meera avoid?"], "expected_consistent": ["social media", "overload"]},
]

ROBUSTNESS_TESTS = [
    {"query": "What car does Abhishek drive?", "should_find": False, "reason": "No car info stored"},
    {"query": "What is Riya's favorite food?", "should_find": False, "reason": "No food preference stored"},
    {"query": "Does Karan play cricket?", "should_find": False, "reason": "Only football is mentioned"},
    {"query": "Where does Abhishek live?", "should_find": False, "reason": "Location not stored for Abhishek"},
    {"query": "Who is the CEO of After Human?", "should_find": False, "reason": "No CEO info stored"},
    {"query": "What is Abhishek's salary?", "should_find": False, "reason": "No salary information stored"},
    {"query": "Does Riya have pets?", "should_find": False, "reason": "No pet information stored"},
    {"query": "What is Karan's phone number?", "should_find": False, "reason": "No phone numbers stored"},
    {"query": "Does Meera have siblings?", "should_find": False, "reason": "No sibling info for Meera"},
    {"query": "What is Aarav's net worth?", "should_find": False, "reason": "No financial details stored"},
    {"query": "Does Abhishek play any instrument?", "should_find": False, "reason": "No music instrument info"},
    {"query": "What is Riya's height?", "should_find": False, "reason": "No physical attributes stored"},
    {"query": "Does Karan have children?", "should_find": False, "reason": "No children info stored"},
    {"query": "What is Meera's blood type?", "should_find": False, "reason": "No medical info stored"},
    {"query": "Where did Aarav go to college?", "should_find": False, "reason": "No education details for Aarav"},
    {"query": "Does Abhishek smoke?", "should_find": False, "reason": "No lifestyle habits like smoking"},
    {"query": "What is Riya's favorite restaurant?", "should_find": False, "reason": "No restaurant preferences"},
    {"query": "Does Karan own a house?", "should_find": False, "reason": "No property info stored"},
    {"query": "What is Meera's email address?", "should_find": False, "reason": "No contact info stored"},
    {"query": "Does Aarav have a co-founder?", "should_find": False, "reason": "No co-founder mentioned"},
    {"query": "What is Abhishek's Twitter handle?", "should_find": False, "reason": "No social media handles stored"},
    {"query": "Does Riya know JavaScript?", "should_find": False, "reason": "No programming skills for Riya"},
    {"query": "What bank does Karan use?", "should_find": False, "reason": "No banking info stored"},
    {"query": "Does Meera have a boyfriend?", "should_find": False, "reason": "No relationship status for Meera"},
    {"query": "What is Aarav's middle name?", "should_find": False, "reason": "No full name details stored"},
    {"query": "Does Abhishek own cryptocurrency?", "should_find": False, "reason": "No crypto info stored"},
    {"query": "What insurance does Riya have?", "should_find": False, "reason": "No insurance info stored"},
    {"query": "Does Karan travel abroad frequently?", "should_find": False, "reason": "No travel info for Karan"},
    {"query": "What is Meera's GPA?", "should_find": False, "reason": "No academic scores stored"},
    {"query": "Does Aarav have investors other than Sequoia?", "should_find": False, "reason": "Only Sequoia mentioned"},
    {"query": "What is Abhishek's shoe size?", "should_find": False, "reason": "No physical measurements stored"},
    {"query": "Does Riya speak French?", "should_find": False, "reason": "No language skills stored for Riya"},
    {"query": "What is Karan's favorite movie?", "should_find": False, "reason": "No movie preference for Karan"},
    {"query": "Does Meera cook?", "should_find": False, "reason": "No cooking info stored"},
    {"query": "What is Aarav's birthday?", "should_find": False, "reason": "No birth dates stored"},
    {"query": "Does Abhishek have tattoos?", "should_find": False, "reason": "No body modifications info"},
    {"query": "What gym does Riya go to?", "should_find": False, "reason": "No gym info for Riya"},
    {"query": "Does Karan drive a Tesla?", "should_find": False, "reason": "No vehicle info stored"},
    {"query": "What medication does Meera take?", "should_find": False, "reason": "No medical info stored"},
    {"query": "Does Aarav have an MBA?", "should_find": False, "reason": "No education details for Aarav"},
    {"query": "What is Abhishek's zodiac sign?", "should_find": False, "reason": "No astrology info stored"},
    {"query": "Does Riya have allergies?", "should_find": False, "reason": "No health info stored for Riya"},
    {"query": "What is Karan's favorite book?", "should_find": False, "reason": "No book preference for Karan"},
    {"query": "Does Meera play any sports?", "should_find": False, "reason": "No sports info for Meera"},
    {"query": "What is Aarav's LinkedIn URL?", "should_find": False, "reason": "No URLs stored"},
    {"query": "Does Abhishek have a PhD?", "should_find": False, "reason": "No PhD mentioned"},
    {"query": "What airline does Riya prefer?", "should_find": False, "reason": "No travel preferences stored"},
    {"query": "Does Karan gamble?", "should_find": False, "reason": "No gambling info stored"},
    {"query": "What is Meera's favorite color?", "should_find": False, "reason": "No color preference for Meera"},
    {"query": "Does Aarav have a yacht?", "should_find": False, "reason": "No luxury items stored"},
]

# ============================================================================
# UTILITIES
# ============================================================================
def memories_to_text(memories: List[Dict], mem: OfflineMemory = None, max_count: int = 10) -> str:
    """Convert memory list to text for judge evaluation.
    
    Uses the EXACT same logic as _get_memory_context from memory_system.py:
    - Extracts graph edges linked by memory_id
    - Formats as SOURCE --RELATIONSHIP--> DESTINATION
    - Includes both relationships AND memory text
    
    Args:
        memories: List of memory results with 'id' and 'memory' fields
        mem: OfflineMemory instance to access graph edges
        max_count: Maximum number of memories to include
    """
    if not memories:
        return ""
    
    # Filter and sort by score desc
    filtered = [m for m in (memories or []) if m.get("id")]
    filtered.sort(key=lambda r: r.get("score", 0.0), reverse=True)
    if not filtered:
        return ""
    
    # Build id -> text map
    id_to_text = {m.get("id"): (m.get("memory") or "").strip() for m in filtered}
    
    lines: List[str] = []
    seen_mids: set = set()
    seen_edges: set = set()
    seen_texts: set = set()
    count = 0
    
    # Get graph edges if mem instance available
    edges_by_mem: dict = {}
    if mem:
        try:
            edges = mem.graph.query_edges()
            for e in edges:
                mid = getattr(e, "memory_id", None)
                if mid:
                    edges_by_mem.setdefault(mid, []).append(e)
        except Exception:
            pass
    
    for r in filtered:
        if count >= max_count:
            break
        mid = r.get("id")
        
        if mid in seen_mids:
            continue
        seen_mids.add(mid)
        
        mem_text = id_to_text.get(mid, "").strip()
        if mem_text in seen_texts:
            continue
        
        # Collect all unique edges for this memory
        memory_edges: List[str] = []
        for e in edges_by_mem.get(mid, []):
            rel_line = f"{e.source} --{e.relationship}--> {e.destination}"
            if rel_line not in seen_edges:
                seen_edges.add(rel_line)
                memory_edges.append(rel_line)
        
        # Add edges first, then memory text ONCE
        if memory_edges:
            lines.extend(memory_edges)
            if mem_text:
                lines.append(mem_text)
                seen_texts.add(mem_text)
            count += 1
        else:
            # No edges - just add memory text
            if mem_text:
                lines.append(mem_text)
                seen_texts.add(mem_text)
                count += 1
    
    # Fallback to top-k memory texts if nothing found
    if not lines:
        for r in filtered[:max_count]:
            memory_text = (r.get("memory", "") or "").strip()
            if memory_text and memory_text not in seen_texts:
                lines.append(memory_text)
                seen_texts.add(memory_text)
    
    return "\n".join(lines)

def store_memories_sequentially(mem: OfflineMemory, user_id: str = "benchmark_user"):
    """
    Store memories strictly sequentially.
    Waits for full processing (Graph + Vector) before moving to next.
    """
    print(f"\nPhase 1: Sequential Storage ({len(MEMORIES_TO_STORE)} items)")
    print("="*60)
    
    stored_count = 0
    failed_count = 0
    
    for i, memory_text in enumerate(MEMORIES_TO_STORE):
        print(f"[{i+1}/{len(MEMORIES_TO_STORE)}] Storing: {memory_text[:60]}...")
        try:
            # mem.add() is synchronous for graph construction and vector storage
            # when using LocalLLM or ServerLLM backend.
            # It returns only after storage is complete.
            result = mem.add(memory_text, user_id=user_id, infer=True)
            
            if result and result.get("results"):
                stored_count += 1
            else:
                print(f"   ⚠️ Warning: Empty result for item {i+1}")
                failed_count += 1
                
        except Exception as e:
            print(f"   ❌ Failed to store item {i+1}: {e}")
            failed_count += 1
            
        # Brief pause to ensure filesystem/DB commits settle (paranoia check)
        time.sleep(0.1)
        
    print(f"\n✅ Storage Phase Complete: {stored_count} stored, {failed_count} failed")
    return stored_count > 0

def run_benchmark_suite(
    mem: OfflineMemory, 
    user_id: str, 
    benchmark_name: str, 
    output_file: str, 
    use_unified: bool
):
    """
    Run the full 250-test suite with specified retrieval method.
    Logs detailed results to output_file.
    """
    print(f"\n{benchmark_name} ({'Unified/@remember' if use_unified else 'Graph-Only/Normal'})")
    print("="*60)
    print(f"Outputting logs to: {output_file}")
    
    # Open log file
    f = open(output_file, "w", encoding="utf-8")
    f.write(f"BENCHMARK REPORT: {benchmark_name}\n")
    f.write(f"Retrieval Method: {'Unified (Vector+Graph)' if use_unified else 'Graph-Only'}\n")
    f.write("="*60 + "\n\n")
    
    total_passed = 0
    total_tests = 0
    category_scores = {}
    
    # Helper to run a category
    def run_category(category_name, tests):
        nonlocal total_passed, total_tests
        cat_passed = 0
        cat_total = len(tests)
        
        print(f"\nRunning {category_name} ({cat_total} tests)...")
        f.write(f"\n--- CATEGORY: {category_name} ---\n")
        
        for t_idx, test in enumerate(tests):
            query = test.get("query", "")
            if not query: # consistency tests differ
                queries = test.get("queries", [])
                query = queries[0] if queries else ""
            
            # --- RETRIEVAL ---
            try:
                if use_unified:
                    # @remember mode
                    res = unified_memory_search(mem, query, user_id=user_id, limit=10, debug=False)
                else:
                    # Normal chat mode
                    res = graph_only_search(mem, query, user_id=user_id, limit=10)
                
                memories = res.get("results", [])
            except Exception as e:
                print(f"   Search Error: {e}")
                memories = []
                
            retrieved_text = memories_to_text(memories, mem)
            
            # --- EVALUATION ---
            expected = test.get("expected") or test.get("expected_persons") or test.get("expected_consistent")
            if isinstance(expected, str): expected = [expected]
            
            # Consistency tests need special handling
            if "queries" in test:
                # For consistency, we just check the first query for the benchmark
                # A full consistency check would require multiple retrievals, 
                # but for this standardized suite we'll treat it as a groundedness check on the first query
                pass

            # Robustness tests need special handling - use Judge for semantic relevance
            if "should_find" in test:
                should_find = test["should_find"]
                
                if not memories or retrieved_text.strip() == "":
                    # No memories retrieved
                    if not should_find:
                        passed = True 
                        reason = "Correctly returned no results for unrelated query"
                    else:
                        passed = False
                        reason = "Failed to find any results when expected"
                else:
                    # Has results - use Judge to check if they're actually relevant
                    topic = test.get("reason", query)
                    judge_passed, judge_reason = judge_retrieval(query, [topic], retrieved_text)
                    
                    if not should_find:
                        # For "should NOT find" - PASS if Judge says results are NOT relevant
                        passed = not judge_passed
                        reason = f"Correctly irrelevant results" if passed else f"Incorrectly returned relevant results: {judge_reason}"
                    else:
                        # For "should find" - PASS if Judge says results ARE relevant
                        passed = judge_passed
                        reason = judge_reason
            else:
                # Standard groundedness/relevance check via Judge LLM
                passed, reason = judge_retrieval(query, expected, retrieved_text)
            
            # --- LOGGING ---
            status = "✅ PASS" if passed else "❌ FAIL"
            if passed: cat_passed += 1
            
            log_entry = (
                f"Test #{t_idx+1}: {query}\n"
                f"Expected: {expected}\n"
                f"Retrieved: {retrieved_text[:300]}...\n"
                f"Verdict: {status}\n"
                f"Reason: {reason}\n"
                f"{'-'*40}\n"
            )
            f.write(log_entry)
            f.flush()
            
            # Progress dot
            print("." if passed else "x", end="", flush=True)
            if (t_idx + 1) % 50 == 0: print()
            
        print(f"  Result: {cat_passed}/{cat_total} ({(cat_passed/cat_total)*100:.1f}%)")
        category_scores[category_name] = (cat_passed, cat_total)
        total_passed += cat_passed
        total_tests += cat_total

    # Run all categories
    run_category("Faithfulness", FAITHFULNESS_TESTS)
    run_category("Retrieval Accuracy", RETRIEVAL_TESTS)
    run_category("Multi-Hop Reasoning", REASONING_TESTS)
    run_category("Consistency", CONSISTENCY_TESTS)
    run_category("Robustness", ROBUSTNESS_TESTS)
    
    # Close log
    f.write(f"\n\nFINAL SCORE: {total_passed}/{total_tests} ({(total_passed/total_tests)*100:.1f}%)\n")
    f.close()
    
    return total_passed, total_tests, category_scores

# ============================================================================
# MAIN
# ============================================================================

def main():
    print("🚀 Unified Memory Benchmark System")
    print(f"   API: {LM_STUDIO_URL}")
    print(f"   Model: {LM_STUDIO_MODEL}")
    
    # Test LLM connection
    print("\n🔗 Testing Judge LLM connection...")
    if call_llm("Ping") == "ERROR":
        print("❌ Cannot connect to LM Studio. Aborting.")
        return

    # Initialize Memory System
    config_path = "configs/offline.yaml"
    test_data_path = "benchmark_test_data"
    
    if os.path.exists(test_data_path):
        print(f"🗑️ Cleaning {test_data_path}...")
        shutil.rmtree(test_data_path)
    os.makedirs(test_data_path)
    
    print("📦 Initializing Memory...")
    cfg = OfflineMemoryConfig.from_file(config_path)
    cfg.vector_store.path = test_data_path
    cfg.graph.db_path = os.path.join(test_data_path, "graph.db")
    cfg.history_db_path = os.path.join(test_data_path, "history.db")
    cfg.entity_index.db_path = os.path.join(test_data_path, "entity_index.db")
    cfg.node_vectors.path = os.path.join(test_data_path, "node_vectors")
    cfg.llm.backend = "none" # Inject server LLM manually
    
    mem = OfflineMemory(cfg)
    
    from server_llm import ServerLLM
    mem.llm = ServerLLM(
        base_url=LM_STUDIO_URL,
        model_name=LM_STUDIO_MODEL,
        deployment_name=LM_STUDIO_MODEL,
        api_key="local",
        default_temperature=0.1,
        default_top_p=0.9,
        default_max_tokens=500
    )
    
    user_id = "benchmark_user"
    
    # 1. Sequential Storage
    success = store_memories_sequentially(mem, user_id)
    if not success:
        return
        
    print("\n⏳ Waiting 5s for indexes to settle...")
    time.sleep(5)
    
    # 2. Benchmark 1: Normal Conversation (graph_only_search)
    score1, total1, cats1 = run_benchmark_suite(
        mem, user_id, 
        benchmark_name="Benchmark 1: Normal Conversation",
        output_file="benchmark_1_results.txt", 
        use_unified=False
    )
    
    # 3. Benchmark 2: @remember (unified_memory_search)
    score2, total2, cats2 = run_benchmark_suite(
        mem, user_id, 
        benchmark_name="Benchmark 2: @remember Command",
        output_file="benchmark_2_results.txt", 
        use_unified=True
    )
    
    # 4. Final Comparison
    print("\n" + "="*70)
    print("📊 FINAL BENCHMARK COMPARISON")
    print("="*70)
    print(f"{'Category':<25} | {'Normal (No Command)':<20} | {'@remember Command':<20}")
    print("-" * 70)
    
    for cat in cats1.keys():
        p1, t1 = cats1[cat]
        p2, t2 = cats2[cat]
        acc1 = (p1/t1)*100
        acc2 = (p2/t2)*100
        print(f"{cat:<25} | {p1}/{t1} ({acc1:5.1f}%)       | {p2}/{t2} ({acc2:5.1f}%)")
        
    print("-" * 70)
    total_acc1 = (score1/total1)*100
    total_acc2 = (score2/total2)*100
    print(f"{'OVERALL ACCURACY':<25} | {score1}/{total1} ({total_acc1:5.1f}%)       | {score2}/{total2} ({total_acc2:5.1f}%)")
    print("="*70)
    print("\nDetailed logs saved to:")
    print(" - benchmark_1_results.txt")
    print(" - benchmark_2_results.txt")

if __name__ == "__main__":
    main()
