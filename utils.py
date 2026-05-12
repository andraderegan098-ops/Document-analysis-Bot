import os
import logging
import warnings
from dotenv import load_dotenv

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

try:
    from langchain.schema import Document
except ImportError:
    from langchain_core.documents import Document

import fitz  # pymupdf

load_dotenv()
os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore")
logging.getLogger("transformers").setLevel(logging.ERROR)


def get_embedding_model():
    """Returns a lightweight embedding model with normalized embeddings."""
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    return HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )


def process_pdfs(directory):
    """Reads PDFs from a directory using PyMuPDF and splits them into chunks."""
    if not os.path.exists(directory):
        os.makedirs(directory)
        return []

    all_docs = []
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)

    for file in os.listdir(directory):
        if file.endswith(".pdf"):
            file_path = os.path.join(directory, file)
            try:
                doc = fitz.open(file_path)
                pages_text = []
                for page in doc:
                    text = page.get_text().strip()
                    if text:
                        pages_text.append(text)
                doc.close()

                full_text = "\n\n".join(pages_text)

                if not full_text.strip():
                    print(f"⚠️ '{file}' yielded no text — skipping.")
                    continue

                chunks = text_splitter.split_text(full_text)
                for chunk in chunks:
                    all_docs.append(Document(
                        page_content=chunk,
                        metadata={"source": file}
                    ))
                print(f"✅ '{file}' → {len(chunks)} chunks")

            except Exception as e:
                print(f"⚠️ Error reading '{file}': {e}")

    return all_docs