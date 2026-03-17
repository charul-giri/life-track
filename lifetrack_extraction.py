"""
lifetrack_extraction.py

Reusable module to extract tasks and associated times from a paragraph.
Converted from your flush_Version.ipynb notebook.

Main entrypoint:
    extract_tasks_from_paragraph(paragraph: str) -> list[dict]

Each returned dict has the form:
    {
        "task": "<task text>",
        "time": "<YYYY-MM-DD HH:MM:SS or ''>"
    }
"""

from __future__ import annotations

from typing import List, Dict
import spacy
import dateparser

# ---------------------------------------------------------------------
# Model & configuration
# ---------------------------------------------------------------------

# Load spaCy model once when the module is imported
# (so that repeated calls are fast).
_nlp = spacy.load("en_core_web_sm")

TASK_VERBS = [
    "do", "go", "visit", "drink", "prepare", "complete", "attend", "wash",
    "help", "plan", "learn", "clean", "check", "collect", "start", "sleep",
    "practice", "finish", "submit", "call", "buy", "take", "revise", "water",
]


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
# Core task + time extraction logic (from your notebook)
# ---------------------------------------------------------------------

def extract_tasks(paragraph: str) -> List[Dict[str, str]]:
    """
    Extract tasks and associated times from a paragraph.

    This is a direct modular version of your notebook's `extract_tasks`:
    - Finds DATE/TIME entities.
    - Finds verbs that look like tasks (from TASK_VERBS).
    - Builds a clean phrase for each task.
    - Assigns the nearest following time expression (if any).
    - Removes very short/garbage tasks and duplicates.

    Args:
        paragraph: Raw text / paragraph with multiple tasks and times.

    Returns:
        List of dictionaries:
        [
            {"task": "prepare the breakfast", "time": "2025-12-07 08:00:00"},
            {"task": "go to the gym", "time": "2025-12-07 09:00:00"},
            ...
        ]
    """
    nlp = get_nlp()
    doc = nlp(paragraph)
    tasks: List[Dict[str, str]] = []

    # Step 1: Extract time expressions
    times = []
    for ent in doc.ents:
        if ent.label_ in ["TIME", "DATE"]:
            parsed_time = dateparser.parse(ent.text)
            if parsed_time:
                times.append((ent.start, ent.end, parsed_time))

    # Step 2: Extract tasks cleanly
    for token in doc:
        if token.lemma_ in TASK_VERBS and token.pos_ == "VERB":

            # build clean task phrase
            phrase = token.text
            dobj = [
                child
                for child in token.children
                if child.dep_ in ["dobj", "pobj", "attr", "xcomp"]
            ]
            prep = [
                child
                for child in token.children
                if child.dep_ == "prep"
            ]

            # add objects
            for obj in dobj:
                phrase += " " + obj.text
                for sub in obj.children:
                    if sub.dep_ in ["compound", "amod", "pobj"]:
                        phrase += " " + sub.text

            # add prepositions like "to the gym"
            for p in prep:
                phrase += " " + p.text
                for obj in p.children:
                    if obj.dep_ in ["pobj", "compound"]:
                        phrase += " " + obj.text

            phrase = phrase.lower().strip()

            # Step 3: Find nearest upcoming time
            assigned_time = ""
            for start, end, parsed in times:
                if start > token.i:
                    assigned_time = parsed.strftime("%Y-%m-%d %H:%M:%S")
                    break

            tasks.append({"task": phrase, "time": assigned_time})

    # Step 4: Remove wrong tasks and duplicates
    cleaned: List[Dict[str, str]] = []
    seen = set()

    for t in tasks:
        # skip single-word garbage tasks
        if len(t["task"].split()) < 2:
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
# Simple manual test (optional)
# ---------------------------------------------------------------------

if __name__ == "__main__":
    example = (
        "today I have to prepare the breakfast, go to the gym, "
        "then in the evening around 7 pm call mom and start working "
        "on my website redesign. I must sleep early tonight."
    )
    results = extract_tasks_from_paragraph(example)
    for r in results:
        print(f"TASK: {r['task']!r}  TIME: {r['time']!r}")
