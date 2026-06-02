from __future__ import annotations

import os
import sys
from pathlib import Path


def _setup_imports() -> None:
    here = Path(__file__).resolve().parent
    root = here.parent
    llm_style = root / "LLM_Style"
    for p in (root, llm_style):
        ps = str(p)
        if ps not in sys.path:
            sys.path.append(ps)


_setup_imports()

from linguistic_analyzer import LinguisticAnalyzer  # type: ignore
from profile_index import StyleEmotionProfile  # type: ignore
from style_system import build_system_prompt  # type: ignore


def main() -> None:
    print("[run] style learning smoke test (no pytest)\n")

    # Prepare temporary store under system_outputs
    root = Path(__file__).resolve().parent.parent
    store_dir = root / "local_data" / "style_emotion" / "style_test_store"
    store_dir.mkdir(parents=True, exist_ok=True)

    # Create profile and verify strict mode requires init
    profile = StyleEmotionProfile()
    try:
        profile.update_phrase_counts({"[filler] umm": 1})
        print("[error] update_phrase_counts did not raise before init")
    except RuntimeError:
        print("[ok] strict mode enforced before initialize_deduplicator()")

    # Initialize deduplicator strictly
    embed_model_path = root / "models" / "bge-m3"
    if not embed_model_path.exists():
        raise FileNotFoundError(f"Embedding model not found at {embed_model_path}")
    profile.initialize_deduplicator(str(embed_model_path), str(store_dir))
    print("[ok] deduplicator initialized")

    # Feed messages and learn
    analyzer = LinguisticAnalyzer()
    texts = [
        "lol nahhh fam that ain't it 🤣",
        "brooooo this pizza hittin' diff rn 🍕🔥",
        "sheeeesh that was clean ✨",

        # Weird spacing and casing
        "w h a t  i s  h a p p e n i n g ???",
        "ThIs MiXeD cAsE iS wEiRd",
        "   extra    spaces   everywhere    ",

        # Short responses
        "ok",
        "K.",
        "yea.",
        "nahhh",

        # Sarcastic / mocking
        "Wow, amazing, so original 👏👏👏",
        "Sure, because THAT'S gonna help 🙄",

        # With numbers and symbols
        "g2g in 5 min, cya!!",
        "Level 9000 powerup 💪⚡",
        "2+2=5 (jk) 🤓",

        # Hashtags / social
        "#blessed #mood",
        "can't stop laughing #LMAO 😂😂😂",
        "@user did you even read the doc?",

        # Foreign words inline
        "C'est la vie, bro... 😅",
        "mañana we'll finish it, promise!",
        "Arigatooo ✌️",

        # Emoticons + emojis combo
        ":) :P 😜",
        "XD lololol 😂😂😂",
        ";_; whyyyyyy",

        # Commands / code snippets
        "pls run `docker-compose up` before testing 🙏",
        "error 404: brain not found 💀",
        "sudo rm -rf / (jk jk don’t try it)",

        # Reactions only
        "😂😂😂",
        "💀",
        "👌",
        "🔥🔥🔥",

        # Long messy run-on
        "sooooo like I went there and then she was like no way and I was like yes way and then we all laughed soooo much it was crazyyyyy 🤯🤯🤯",

        # Formal vs informal back-to-back
        "Kindly confirm receipt of the attached invoice.",
        "yo did u get the thingy?? 😅",

        # Edge-case oddities
        "",
        "....",
        "!!!???!!!",
        "🤖🤖🤖🤖🤖🤖🤖🤖"
    ]

    for t in texts:
        rep = analyzer.analyze(t)
        profile.update_from_features(rep)
        profile.update_phrase_counts(analyzer.extract_phrase_counts(t))
        safe = t[:60].encode('cp1252', 'replace').decode('cp1252')
        print(f"[learned] {safe}...")

    # Check persistence outputs
    mem_path = store_dir / "pattern_memory.json"
    meta_path = store_dir / "pattern_store.meta.json"
    print(f"[check] pattern_memory.json exists: {mem_path.exists()}")
    print(f"[check] pattern_store.meta.json exists: {meta_path.exists()}")

    # Show presence of learned signals
    keys = list(profile.phrase_counts.keys())
    has_ngrams = any(k.startswith("[ngram2] ") or k.startswith("[ngram3] ") for k in keys)
    has_emoji = any(k.startswith("[emoji] ") for k in keys)
    print(f"[signals] ngrams: {has_ngrams}, emoji: {has_emoji}")

    # Print dynamic detailed style block
    detailed = profile.format_for_llm_system_prompt_detailed()
    print("\n===== STYLE BLUEPRINT (detailed) =====")
    print(detailed)
    print("====================================\n")

    # Print combined system prompt
    system_prompt = build_system_prompt(profile)
    print("===== SYSTEM PROMPT (combined) =====")
    print(system_prompt[:1200] + ("..." if len(system_prompt) > 1200 else ""))
    print("====================================\n")


if __name__ == "__main__":
    main()


