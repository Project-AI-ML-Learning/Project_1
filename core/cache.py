# cache.py
# Think of this like STICKY NOTES on your desk.
# Job: remember answers to questions already asked.
# Same question again? Read the sticky note.
# Don't call Gemini. Save money. Save time.

import json
import hashlib
import os

CACHE_FILE = "query_cache.json"

def load_cache():
    # Pick up all sticky notes from desk
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, "r") as f:
            return json.load(f)
    return {}


def save_cache(cache):
    # Put all sticky notes back on desk
    with open(CACHE_FILE, "w") as f:
        json.dump(cache, f, indent=2)


def make_key(query):
    # Turn the question into a short code
    # "What is AI?" → "a3f9c12b..."
    # Same question always = same code
    return hashlib.md5(query.strip().lower().encode()).hexdigest()


def get_cached_answer(query):
    # Look for a sticky note for this question
    cache = load_cache()
    key   = make_key(query)

    if key in cache:
        print("Sticky note found! Returning cached answer.")
        return cache[key]

    print("No sticky note. Need to ask Gemini.")
    return None


def store_answer(query, answer):
    # Write a new sticky note and put it on desk
    cache        = load_cache()
    key          = make_key(query)
    cache[key]   = answer
    save_cache(cache)
    print("Answer written to sticky note.")


def clear_all_cache():
    # Someone edited a document — ALL sticky notes
    # might be wrong now. Throw them all away.
    # Better to rebuild fresh than give wrong answers.
    if os.path.exists(CACHE_FILE):
        os.remove(CACHE_FILE)
        print("All sticky notes cleared.")

"""Teacher asks same question twice — first time you think hard (Gemini).
 Second time you just read your sticky note. But if the textbook changes — throw ALL sticky notes away. They might have wrong answers now."""