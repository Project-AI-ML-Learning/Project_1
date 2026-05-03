# registry.py
# Think of this like a NOTEBOOK.
# Job: remember what every file looked like
# last time we saw it.
# If the file changes — we know immediately.

import json
import hashlib
import os

REGISTRY_FILE = "file_hash_registry.json"

def load_registry():
    # Open the notebook
    # If no notebook exists yet — start fresh
    if os.path.exists(REGISTRY_FILE):
        with open(REGISTRY_FILE, "r") as f:
            return json.load(f)
    return {}


def save_registry(registry):
    # Write updated notes back to the notebook
    with open(REGISTRY_FILE, "w") as f:
        json.dump(registry, f, indent=2)


def get_file_hash(filepath):
    # Take a fingerprint of the file's content
    # Same content  = same fingerprint
    # Edit one word = completely different fingerprint
    with open(filepath, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()


def has_file_changed(filename, filepath):
    # Check notebook: what did this file look like before?
    registry     = load_registry()
    saved_hash   = registry.get(filename)
    current_hash = get_file_hash(filepath)

    if saved_hash is None:
        return "new"                    # never seen before

    if current_hash != saved_hash:
        return "edited"                 # fingerprint changed

    return "same"                       # nothing changed


def update_registry(filename, filepath):
    # Take a fresh fingerprint and write it to notebook
    registry           = load_registry()
    registry[filename] = get_file_hash(filepath)
    save_registry(registry)


def remove_from_registry(filename):
    # File deleted — tear that page out of the notebook
    registry = load_registry()
    if filename in registry:
        del registry[filename]
        save_registry(registry)

"""Notebook remembers every friend's face (fingerprint). New kid at school → write them down. Friend got a haircut (edited) → old photo doesn't match → update. Friend moved away (deleted) → cross them out."""