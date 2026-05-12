"""
memory.py — Per-user, per-collection chat history persistence
Stores history in .streamlit/memory/<username>/<collection>.json
"""
import os, json
from datetime import datetime
from langchain_core.messages import HumanMessage, AIMessage

MEMORY_DIR = os.path.join(".streamlit", "memory")
MAX_TURNS  = 20  # keep last 20 Q&A turns per file


def _path(username: str, collection: str) -> str:
    safe = collection.strip().replace(" ", "_").replace("/", "-")
    d = os.path.join(MEMORY_DIR, username)
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, f"{safe}.json")


def load_memory(username="global", collection="default") -> list:
    p = _path(username, collection)
    if os.path.exists(p):
        try:
            with open(p, encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return []
    return []


def save_memory(history: list, username="global", collection="default"):
    raw = to_raw_dicts(history)
    raw = raw[-(MAX_TURNS * 2):]          # trim
    try:
        p = _path(username, collection)
        # ── Save updated_at timestamp in a sidecar metadata file ─────────────
        meta_p = p.replace(".json", ".meta.json")
        # Get first human message as preview
        preview = ""
        for m in raw:
            if m.get("type") == "human":
                preview = m.get("content", "")[:80]
                break
        meta = {
            "collection":  collection,
            "updated_at":  datetime.now().isoformat(),
            "preview":     preview,
            "turn_count":  len([m for m in raw if m.get("type") == "human"]),
        }
        with open(meta_p, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
        with open(p, "w", encoding="utf-8") as f:
            json.dump(raw, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"[memory] save failed: {e}")


def clear_memory(username="global", collection="default"):
    p = _path(username, collection)
    if os.path.exists(p):
        os.remove(p)
    meta_p = p.replace(".json", ".meta.json")
    if os.path.exists(meta_p):
        os.remove(meta_p)


def list_recent_chats(username="global") -> list:
    """
    Return all saved conversations for a user, sorted newest first.
    Each entry: { collection, updated_at, preview, turn_count, history }
    """
    d = os.path.join(MEMORY_DIR, username)
    if not os.path.exists(d):
        return []

    results = []
    for fname in os.listdir(d):
        if not fname.endswith(".meta.json"):
            continue
        meta_p = os.path.join(d, fname)
        data_p = meta_p.replace(".meta.json", ".json")
        try:
            with open(meta_p, encoding="utf-8") as f:
                meta = json.load(f)
            history = []
            if os.path.exists(data_p):
                with open(data_p, encoding="utf-8") as f:
                    history = json.load(f)
            meta["history"] = history
            results.append(meta)
        except Exception:
            continue

    return sorted(results, key=lambda x: x.get("updated_at", ""), reverse=True)


def to_langchain_messages(raw: list) -> list:
    out = []
    for m in raw:
        if m.get("type") == "human":
            out.append(HumanMessage(content=m.get("content", "")))
        else:
            out.append(AIMessage(content=m.get("content", "")))
    return out


def to_raw_dicts(messages: list) -> list:
    out = []
    for m in messages:
        if isinstance(m, dict):
            out.append(m)
        elif isinstance(m, HumanMessage):
            out.append({"type": "human", "content": m.content})
        elif isinstance(m, AIMessage):
            out.append({"type": "ai",    "content": m.content})
    return out