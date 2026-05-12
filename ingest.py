"""
ingest.py — Named document collections
Collections live in vectordbs/<safe_name>/
Metadata stored in vectordbs/collections.json
"""
import os, json, shutil, tempfile
from datetime import datetime
from utils import process_pdfs, get_embedding_model
from langchain_community.vectorstores import FAISS

ROOT = "vectordbs"
META = os.path.join(ROOT, "collections.json")


def _safe(name: str) -> str:
    return name.strip().replace(" ", "_").replace("/", "-")


def _load_meta() -> dict:
    os.makedirs(ROOT, exist_ok=True)
    if os.path.exists(META):
        try:
            with open(META) as f: return json.load(f)
        except Exception: return {}
    return {}


def _save_meta(meta: dict):
    os.makedirs(ROOT, exist_ok=True)
    with open(META, "w") as f: json.dump(meta, f, indent=2)


def list_collections() -> list:
    meta = _load_meta()
    return sorted(meta.values(), key=lambda x: x.get("created_at",""), reverse=True)


def collection_path(name: str) -> str:
    return os.path.join(ROOT, _safe(name))


def collection_exists(name: str) -> bool:
    return os.path.exists(collection_path(name))


def delete_collection(name: str):
    p = collection_path(name)
    if os.path.exists(p):
        shutil.rmtree(p)
    meta = _load_meta()
    meta.pop(name, None)
    _save_meta(meta)


def ingest_uploaded_files(uploaded_files: list, collection_name: str,
                          progress_cb=None) -> tuple:
    """
    uploaded_files: list of Streamlit UploadedFile objects
    Returns (success: bool, message: str)
    """
    def log(msg):
        if progress_cb: progress_cb(msg)

    if not uploaded_files:
        return False, "No files provided."

    with tempfile.TemporaryDirectory() as tmp:
        for f in uploaded_files:
            dst = os.path.join(tmp, f.name)
            with open(dst, "wb") as fh:
                fh.write(f.read())
        log(f"📄 Processing {len(uploaded_files)} PDF(s)…")
        docs = process_pdfs(tmp)

    if not docs:
        return False, "No text could be extracted from the PDFs."

    log(f"✂️ Split into {len(docs)} chunks. Building index…")

    out = collection_path(collection_name)
    if os.path.exists(out): shutil.rmtree(out)
    os.makedirs(out, exist_ok=True)

    try:
        embeddings = get_embedding_model()
        vs = FAISS.from_documents(docs, embeddings)
        vs.save_local(out)
    except Exception as e:
        return False, f"Index build failed: {e}"

    meta = _load_meta()
    meta[collection_name] = {
        "name":        collection_name,
        "path":        out,
        "doc_count":   len(uploaded_files),
        "chunk_count": len(docs),
        "files":       [f.name for f in uploaded_files],
        "created_at":  datetime.now().isoformat(),
    }
    _save_meta(meta)
    log(f"✅ Done — {len(docs)} chunks indexed.")
    return True, f"✅ '{collection_name}' ready ({len(docs)} chunks)."


# ── Legacy CLI (unchanged) ────────────────────────────────────────────────────
def create_db():
    import shutil as _sh
    from utils import process_pdfs, get_embedding_model
    docs = process_pdfs("input")
    if not docs:
        print("No PDFs in input/"); return
    if os.path.exists("vectordb"): _sh.rmtree("vectordb")
    FAISS.from_documents(docs, get_embedding_model()).save_local("vectordb")
    print("✅ vectordb ready")

if __name__ == "__main__":
    create_db()