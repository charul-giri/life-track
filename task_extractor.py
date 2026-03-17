"""
lifetrack_extraction.py

Reusable module to extract tasks and associated times from a paragraph.
Main entrypoint:
    extract_tasks_from_paragraph(paragraph: str) -> list[dict]

Each returned dict has the form:
    {
        "task": "<task text>",
        "time": "<YYYY-MM-DD HH:MM:SS or ''>"
    }
"""

from __future__ import annotations

from typing import List, Dict, Tuple
import spacy
import dateparser

# ---------------------------------------------------------------------
# Model & configuration
# ---------------------------------------------------------------------

# Load spaCy model once when the module is imported.
_nlp = spacy.load("en_core_web_sm")

TASK_VERBS = {
    "do", "go", "visit", "drink", "prepare", "complete", "attend", "wash",
    "help", "plan", "learn", "clean", "check", "collect", "start", "sleep",
    "practice", "finish", "submit", "call", "buy", "take", "revise", "water",
}

TIME_ENTITY_LABELS = {"TIME", "DATE"}

# Max token distance between a task verb and a time expression
MAX_TIME_DISTANCE = 25


def get_nlp():
    """Internal helper in case you ever want to switch models."""
    return _nlp


# ---------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------

def extract_time_string(text: str) -> str:
    """
    Parse a time or date expression into a normalized string.

    Args:
        text: e.g. 'tomorrow at 5 pm', 'next Monday morning', etc.

    Returns:
        Time string in 'YYYY-MM-DD HH:MM:SS' format, or '' if parsing fails.
    """
    parsed = dateparser.parse(text)
    if parsed:
        return parsed.strftime("%Y-%m-%d %H:%M:%S")
    return ""


# ---------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------

def _collect_time_entities(doc) -> Tuple[List[Dict], set]:
    """
    Collect TIME/DATE entities and a set of all token indices that belong
    to those entities.

    Returns:
        times: list of dicts with keys: start, end, text, ent
        time_token_idxs: set of token indices belonging to any TIME/DATE ent
    """
    times: List[Dict] = []
    time_token_idxs = set()

    for ent in doc.ents:
        if ent.label_ in TIME_ENTITY_LABELS:
            times.append(
                {
                    "start": ent.start,
                    "end": ent.end,  # exclusive
                    "text": ent.text,
                    "ent": ent,
                }
            )
            for tok in ent:
                time_token_idxs.add(tok.i)

    return times, time_token_idxs


def _find_best_time_for_token(token, times: List[Dict]) -> str:
    """
    For a given verb token, find the nearest TIME/DATE entity by token distance.
    If the closest one is too far away, return '' (no time).
    """
    if not times:
        return ""

    best = None
    best_dist = 10**9

    for t in times:
        # distance from token to the span [start, end-1]
        start = t["start"]
        end = t["end"] - 1
        dist = min(abs(token.i - start), abs(token.i - end))
        if dist < best_dist:
            best_dist = dist
            best = t

    if best is None or best_dist > MAX_TIME_DISTANCE:
        return ""

    return extract_time_string(best["text"])


def _build_task_phrase(token, time_token_idxs: set) -> str:
    """
    Build a clean task phrase centered around a verb token, excluding
    any TIME/DATE entity tokens from the phrase.

    Strategy:
    - Include the verb.
    - Include direct objects, xcomps, attrs, and their useful modifiers.
    - Include prepositional phrases that are NOT time-related
      (e.g., 'to the gym', but NOT 'at 7 pm').
    - Collect tokens and then sort by their index to preserve natural order.
    """
    phrase_tokens = set()

    # Always include the main verb if not itself part of a time expression
    if token.i not in time_token_idxs:
        phrase_tokens.add(token)

    # Direct objects, xcomp, attrs, etc.
    dobj_like = [
        child
        for child in token.children
        if child.dep_ in {"dobj", "pobj", "attr", "xcomp"}
    ]

    for obj in dobj_like:
        if obj.i in time_token_idxs:
            continue
        phrase_tokens.add(obj)

        # include modifiers of the object
        for sub in obj.children:
            if sub.dep_ in {"compound", "amod", "pobj", "det"}:
                if sub.i in time_token_idxs:
                    continue
                phrase_tokens.add(sub)

    # Prepositions (e.g., "to the gym", "with friends")
    preps = [child for child in token.children if child.dep_ == "prep"]
    for p in preps:
        # If this prepositional phrase contains any TIME/DATE tokens,
        # skip the whole phrase (e.g., 'at 7 pm').
        has_time_descendant = any(
            desc.i in time_token_idxs for desc in p.subtree
        )
        if has_time_descendant:
            continue

        # Otherwise include the preposition phrase
        if p.i not in time_token_idxs:
            phrase_tokens.add(p)
        for obj in p.children:
            if obj.dep_ in {"pobj", "compound", "amod", "det"}:
                if obj.i in time_token_idxs:
                    continue
                phrase_tokens.add(obj)
                for sub in obj.children:
                    if sub.dep_ in {"compound", "amod", "det"}:
                        if sub.i in time_token_idxs:
                            continue
                        phrase_tokens.add(sub)

    # Sort tokens by original position for natural word order
    ordered = sorted(phrase_tokens, key=lambda t: t.i)
    phrase = " ".join(tok.text for tok in ordered).lower().strip()

    return phrase


# ---------------------------------------------------------------------
# Core task + time extraction logic
# ---------------------------------------------------------------------

def extract_tasks(paragraph: str) -> List[Dict[str, str]]:
    """
    Extract tasks and associated times from a paragraph.

    Steps:
    - Find TIME/DATE entities and remember their token positions.
    - For each 'task verb' (from TASK_VERBS), build a clean phrase
      WITHOUT time words.
    - For each task, select the closest TIME/DATE entity by token distance.
    - Filter out too-short tasks and duplicates.
    """
    nlp = get_nlp()
    doc = nlp(paragraph)
    tasks: List[Dict[str, str]] = []

    # Step 1: collect time entities and token indices
    times, time_token_idxs = _collect_time_entities(doc)

    # Step 2: extract tasks
    for token in doc:
        if token.pos_ == "VERB" and token.lemma_.lower() in TASK_VERBS:
            phrase = _build_task_phrase(token, time_token_idxs)

            # Step 3: find the best matching time for this verb
            assigned_time = _find_best_time_for_token(token, times)

            tasks.append({"task": phrase, "time": assigned_time})

    # Step 4: clean up tasks (remove garbage and duplicates)
    cleaned: List[Dict[str, str]] = []
    seen = set()

    for t in tasks:
        # Skip empty or very short tasks (often noise)
        if not t["task"] or len(t["task"].split()) < 2:
            continue

        key = (t["task"], t["time"])
        if key not in seen:
            cleaned.append(t)
            seen.add(key)

    return cleaned


def extract_tasks_from_paragraph(paragraph: str) -> List[Dict[str, str]]:
    """
    Main function to be used from other files.

    Thin wrapper around `extract_tasks` so you can
    import a clearly named function.
    """
    return extract_tasks(paragraph)


# ---------------------------------------------------------------------
# Simple manual test
# ---------------------------------------------------------------------

if __name__ == "__main__":
    example = (
        "Today at 8 am I have to prepare the breakfast and at 9 am go to the gym. "
        "In the evening around 7 pm call mom and start working on my website redesign. "
        "I must sleep early tonight."
    )
    results = extract_tasks_from_paragraph(example)
    for r in results:
        print(f"TASK: {r['task']!r}  TIME: {r['time']!r}")
