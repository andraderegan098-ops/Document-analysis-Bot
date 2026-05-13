# app.py — DocuBot with Chart.js + PDF download + Image Vision + Safe Translation + Translate Button + Recent Chats

import os
import re
import json
import base64
import tempfile
from datetime import datetime
import streamlit as st
import streamlit.components.v1 as components
from audio_recorder_streamlit import audio_recorder
from PIL import Image

from retrieval import load_vectorstore, load_reranker, retrieve_with_rerank, build_context_string
from tools import ALL_TOOLS, generate_pdf_tool
from prompts import build_rag_system_message, build_no_context_system_message
from utils import get_embedding_model
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, AIMessage

try:
    from language_config import get_language_names, SUPPORTED_LANGUAGES
    from translation_utils import (
        detect_language, translate_text, translate_response,
        get_whisper_language_hint,
    )
    MULTILANG = True
except ImportError:
    MULTILANG = False

from auth import (
    is_logged_in, current_user, logout,
    render_login_page, render_user_management,
)
from memory import (
    load_memory, save_memory, clear_memory,
    to_langchain_messages, to_raw_dicts, list_recent_chats,
)
from ingest import (
    list_collections, collection_exists, collection_path,
    delete_collection, ingest_uploaded_files,
)
from langchain_community.vectorstores import FAISS

CHARTJS_PATTERN = re.compile(r"```chartjs\s*([\s\S]*?)```")
SUPPORTED_IMAGE_TYPES = ["jpg", "jpeg", "png", "gif", "webp"]

st.set_page_config(page_title="DocuBot - Intelligent Document Analysis", layout="wide")

# ── Styles ────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Pin chat input to the bottom, Claude-style ── */
[data-testid="stChatInput"] {
    position: fixed !important;
    bottom: 0 !important;
    left: 0 !important;
    right: 0 !important;
    z-index: 999 !important;
    background-color: #0e0e0e !important;
    border-top: 1px solid #2a2a2a !important;
    padding: 12px 24px 16px 24px !important;
}

/* Push chat messages up so the last one isn't hidden behind the input bar */
[data-testid="stChatMessageContainer"],
.main .block-container {
    padding-bottom: 100px !important;
}

/* Style the textarea itself */
.stChatInput textarea {
    background-color: #1e1e1e !important;
    color: #ffffff !important;
    border: 1px solid #3a3a3a !important;
    border-radius: 12px !important;
    font-size: 15px !important;
    padding: 12px 16px !important;
}
.stChatInput textarea:focus {
    border-color: #555 !important;
    box-shadow: 0 0 0 2px rgba(255,255,255,0.06) !important;
}

/* ── Translate panel ── */
div[data-testid="stHorizontalBlock"]:has(> div > div[data-testid="stSelectbox"]) {
    background: #181828;
    border: 1px solid #2d2d44;
    border-radius: 10px;
    padding: 8px 12px;
    margin-top: 4px;
}

/* ── Recent chats card ── */
.recent-chat-card {
    background: #1a1a2e;
    border: 1px solid #2d2d44;
    border-radius: 10px;
    padding: 14px 18px;
    margin-bottom: 10px;
}
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# Translation helpers (unchanged)
# ══════════════════════════════════════════════════════════════════════════════

_LANG_NAMES = {
    "en": "English", "hi": "Hindi", "es": "Spanish", "fr": "French",
    "de": "German", "zh": "Chinese", "ar": "Arabic", "pt": "Portuguese",
    "ru": "Russian", "ja": "Japanese", "ko": "Korean", "it": "Italian",
    "nl": "Dutch", "tr": "Turkish", "pl": "Polish", "sv": "Swedish",
    "da": "Danish", "fi": "Finnish", "no": "Norwegian", "cs": "Czech",
    "ro": "Romanian", "hu": "Hungarian", "uk": "Ukrainian",
    "bn": "Bengali", "ta": "Tamil", "te": "Telugu", "mr": "Marathi",
    "gu": "Gujarati", "kn": "Kannada", "ml": "Malayalam", "pa": "Punjabi",
    "ur": "Urdu", "fa": "Persian", "he": "Hebrew", "id": "Indonesian",
    "ms": "Malay", "th": "Thai", "vi": "Vietnamese", "el": "Greek",
    "bg": "Bulgarian", "hr": "Croatian", "sr": "Serbian",
}


def _translate_chunk_with_llm(text: str, target_lang: str, llm) -> str:
    lang_name = _LANG_NAMES.get(target_lang, target_lang)
    prompt = (
        f"Translate the following text into {lang_name}. "
        f"Preserve all markdown formatting (bullet points, bold, headers, etc). "
        f"Return ONLY the translated text, nothing else.\n\n"
        f"TEXT TO TRANSLATE:\n{text}"
    )
    try:
        from langchain_core.messages import HumanMessage as HM
        resp = llm.invoke([HM(content=prompt)])
        result = resp.content if isinstance(resp.content, str) else ""
        return result.strip() if result.strip() else text
    except Exception as e:
        st.warning(f"Translation error: {e}")
        return text


def safe_translate(text: str, target_lang: str, llm=None, source_lang: str = None) -> str:
    if not text:
        return text
    # "__detect__" means: always translate regardless of target language (e.g. user clicked Translate button)
    # Otherwise skip if source and target are both English
    force = (source_lang == "__detect__")
    if not force and target_lang == "en":
        return text

    if llm is not None:
        def _translate_fn(chunk):
            return _translate_chunk_with_llm(chunk, target_lang, llm)
    else:
        try:
            from deep_translator import GoogleTranslator
            def _translate_fn(chunk):
                return GoogleTranslator(source="auto", target=target_lang).translate(chunk)
        except Exception:
            if MULTILANG:
                def _translate_fn(chunk):
                    return translate_response(chunk, target_lang)
            else:
                return text

    parts = CHARTJS_PATTERN.split(text)
    translated_parts = []

    for i, part in enumerate(parts):
        if i % 2 == 0:
            if part.strip():
                try:
                    translated_parts.append(_translate_fn(part))
                except Exception:
                    translated_parts.append(part)
            else:
                translated_parts.append(part)
        else:
            translated_parts.append(f"```chartjs\n{part}\n```")

    return "".join(translated_parts)


# ══════════════════════════════════════════════════════════════════════════════
# Translate Button helper (unchanged)
# ══════════════════════════════════════════════════════════════════════════════

def get_all_language_options() -> list[tuple[str, str]]:
    if MULTILANG:
        lang_names = get_language_names()
        return sorted(lang_names.items(), key=lambda x: x[1])
    return [
        ("en", "English"), ("hi", "Hindi"), ("es", "Spanish"),
        ("fr", "French"), ("de", "German"), ("zh", "Chinese"),
        ("ar", "Arabic"), ("pt", "Portuguese"), ("ru", "Russian"),
        ("ja", "Japanese"), ("ko", "Korean"),
    ]


def render_translate_button(msg_index: int, original_text: str, llm=None):
    lang_options = get_all_language_options()
    codes  = [c for c, _ in lang_options]
    labels = [l for _, l in lang_options]

    open_key    = f"translate_open_{msg_index}"
    result_key  = f"translated_{msg_index}"
    lang_key    = f"translate_lang_{msg_index}"
    pending_key = f"translate_pending_{msg_index}"

    if open_key not in st.session_state:
        st.session_state[open_key] = False
    if lang_key not in st.session_state:
        st.session_state[lang_key] = codes[0]

    # ── Toggle button (small, muted) ─────────────────────────────────────────
    toggle_label = "🌐 Translate" if not st.session_state[open_key] else "✖ Close"
    if st.button(toggle_label, key=f"translate_toggle_{msg_index}",
                 type="secondary", use_container_width=False):
        st.session_state[open_key] = not st.session_state[open_key]
        st.rerun()

    # ── Language picker panel ─────────────────────────────────────────────────
    if st.session_state[open_key]:
        with st.container(border=True):
            col1, col2 = st.columns([4, 1])
            with col1:
                st.selectbox(
                    "Translate to",
                    options=codes,
                    format_func=lambda c: labels[codes.index(c)],
                    key=lang_key,
                    label_visibility="collapsed",
                )
            with col2:
                st.markdown("<div style='margin-top:4px'></div>", unsafe_allow_html=True)
                if st.button("✓ Go", key=f"translate_go_{msg_index}",
                             type="primary", use_container_width=True):
                    st.session_state[pending_key] = True
                    st.rerun()

    # ── Run translation ───────────────────────────────────────────────────────
    if st.session_state.get(pending_key):
        st.session_state[pending_key] = False
        target_code  = st.session_state[lang_key]
        target_label = labels[codes.index(target_code)]
        with st.spinner(f"🌐 Translating to {target_label}…"):
            translated = safe_translate(original_text, target_code, llm=llm, source_lang="__detect__")
        st.session_state[result_key] = {"text": translated, "lang": target_label}
        st.session_state[open_key]   = False
        st.rerun()

    # ── Show translated result ────────────────────────────────────────────────
    if result_key in st.session_state:
        result = st.session_state[result_key]
        with st.container(border=True):
            st.caption(f"🌐 Translated → **{result['lang']}**")
            render_message(result["text"])
            if st.button("✖ Clear", key=f"close_trans_{msg_index}", type="secondary"):
                del st.session_state[result_key]
                st.rerun()


# ── Image helpers ─────────────────────────────────────────────────────────────

def encode_image_base64(image_bytes: bytes, media_type: str) -> str:
    return base64.b64encode(image_bytes).decode("utf-8")


# ── Chart.js helpers ──────────────────────────────────────────────────────────

def _fix_json(raw: str) -> str:
    import re as _re
    raw = _re.sub(r'//[^\n]*', '', raw)
    raw = _re.sub(r',[ \t\n\r]*([}\]])', r'\1', raw)
    for ch, rep in [(u'\u201c', '"'), (u'\u201d', '"'), (u'\u2018', "'"), (u'\u2019', "'")]:
        raw = raw.replace(ch, rep)
    raw = _re.sub(r'": "([^"]*?)"', lambda m: '": "' + m.group(1).replace('\n', ' ') + '"', raw)
    return raw.strip()


def render_chartjs(chart_json_str: str):
    fixed = chart_json_str
    try:
        json.loads(chart_json_str)
    except json.JSONDecodeError:
        fixed = _fix_json(chart_json_str)
        try:
            json.loads(fixed)
        except json.JSONDecodeError as e:
            st.warning(f"⚠️ Could not render chart (JSON error: {e}). Showing raw data instead.")
            st.code(chart_json_str, language="json")
            return

    safe_json = json.dumps(json.loads(fixed))
    html = """
    <!DOCTYPE html><html>
    <head>
        <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
        <style>
            body { margin:0; background:transparent; display:flex;
                   justify-content:center; align-items:center; min-height:100vh; }
            canvas { max-width:100%; max-height:400px; }
        </style>
    </head>
    <body>
        <canvas id="chart"></canvas>
        <script>
            try {
                const config = """ + safe_json + """;
                if (!config.options) config.options = {};
                config.options.responsive = true;
                config.options.maintainAspectRatio = true;
                new Chart(document.getElementById('chart'), config);
            } catch(e) {
                document.body.innerHTML = '<p style="color:red;font-family:sans-serif">Chart error: ' + e.message + '</p>';
            }
        </script>
    </body></html>
    """
    components.html(html, height=450, scrolling=False)


def render_message(content: str):
    """Render markdown + live Chart.js charts."""
    parts = CHARTJS_PATTERN.split(content)
    for i, part in enumerate(parts):
        if i % 2 == 0:
            if part.strip():
                st.markdown(part)
        else:
            render_chartjs(part.strip())


# ── LLM init ──────────────────────────────────────────────────────────────────

def _initialize_llm():
    openai_key = os.getenv("OPENAI_API_KEY")
    gemini_key = os.getenv("GOOGLE_API_KEY")

    if openai_key and openai_key != "your_openai_api_key_here":
        try:
            return ChatOpenAI(model="gpt-4.1", temperature=0.1, api_key=openai_key)
        except Exception as e:
            st.warning(f"OpenAI failed: {e}")

    if gemini_key and gemini_key != "your_gemini_api_key_here":
        try:
            return ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.1, api_key=gemini_key)
        except Exception as e:
            st.error(f"Gemini failed: {e}")

    st.error("No valid API keys found.")
    return None


# ── Cached resources ──────────────────────────────────────────────────────────

@st.cache_resource
def _load_reranker_and_llm():
    reranker = load_reranker()
    llm      = _initialize_llm()
    if llm is None:
        return None, None
    return reranker, llm.bind_tools(ALL_TOOLS)


@st.cache_resource(show_spinner=False)
def _load_vectorstore_for_path(path: str):
    try:
        embeddings = get_embedding_model()
        return FAISS.load_local(path, embeddings, allow_dangerous_deserialization=True)
    except Exception as e:
        st.error(f"Could not load collection at '{path}': {e}")
        return None


def load_rag_resources():
    reranker, llm = _load_reranker_and_llm()
    if reranker is None:
        return None, None, None
    active_path = st.session_state.get("active_col_path")
    if not active_path:
        if os.path.exists("vectordb"):
            active_path = "vectordb"
        else:
            return None, reranker, llm
    vs = _load_vectorstore_for_path(active_path)
    return vs, reranker, llm


@st.cache_resource
def load_whisper_model():
    from faster_whisper import WhisperModel
    return WhisperModel("small", device="cpu", compute_type="int8")


# ── Audio ─────────────────────────────────────────────────────────────────────

def transcribe_audio(audio_bytes: bytes, lang_hint=None):
    model = load_whisper_model()
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        f.write(audio_bytes)
        path = f.name
    try:
        kwargs = {"language": lang_hint} if lang_hint else {}
        segments, info = model.transcribe(path, **kwargs)
        text = "".join(s.text for s in segments).strip()
        return text, info.language
    finally:
        os.unlink(path)


# ── RAG turn ──────────────────────────────────────────────────────────────────

def _run_rag_turn(user_input_en: str, vectorstore, reranker, llm):
    docs = retrieve_with_rerank(vectorstore, user_input_en, reranker)
    system_prompt = (
        build_no_context_system_message() if not docs
        else build_rag_system_message(build_context_string(docs), include_chart_instructions=True)
    )
    messages = [system_prompt] + st.session_state.get("memory", []) + [HumanMessage(content=user_input_en)]
    try:
        response = llm.invoke(messages)
        text = response.content if isinstance(response.content, str) else ""

        # ── FIX: inject the LLM's response as the PDF content if the tool
        #         was called without report_text (which is the common case when
        #         the LLM writes its answer into response.content and separately
        #         triggers the tool with an empty or missing argument).
        tool_calls = response.tool_calls or []
        for tc in tool_calls:
            if tc["name"] == "generate_pdf_tool" and text:
                if not tc["args"].get("report_text"):
                    tc["args"]["report_text"] = text

        tool_results = [_dispatch_tool(t) for t in tool_calls]
        return text, tool_results
    except Exception as e:
        return f"Error: {e}", []


# ── Vision turn ───────────────────────────────────────────────────────────────

def _run_vision_turn(user_input_en: str, image_b64: str, media_type: str,
                     vectorstore, reranker, llm, response_lang: str = "en"):
    docs = retrieve_with_rerank(vectorstore, user_input_en, reranker)

    lang_label = "English"
    if MULTILANG and response_lang in SUPPORTED_LANGUAGES:
        lang_label = SUPPORTED_LANGUAGES[response_lang]["name"]
    elif response_lang != "en":
        lang_label = response_lang

    if docs:
        context = build_context_string(docs)
        vision_instruction = (
            f"You are DocuBot, an intelligent document analysis assistant.\n\n"
            f"IMPORTANT: You MUST respond in {lang_label} language only.\n\n"
            f"The user has uploaded an image AND you have relevant document context below.\n"
            f"Answer the user's question by analysing the image carefully and, where relevant, "
            f"cross-referencing the document context.\n\n"
            f"DOCUMENT CONTEXT:\n{context}\n\n"
            f"Describe what you see in the image and answer the user's question thoroughly.\n"
            f"Remember: respond entirely in {lang_label}."
        )
    else:
        vision_instruction = (
            f"You are DocuBot, an intelligent document analysis assistant.\n\n"
            f"IMPORTANT: You MUST respond in {lang_label} language only.\n\n"
            f"The user has uploaded an image. Analyse it carefully and answer their question. "
            f"Describe charts, tables, text, diagrams, or any content visible in the image.\n"
            f"Remember: respond entirely in {lang_label}."
        )

    from langchain_core.messages import SystemMessage
    system_prompt = SystemMessage(content=vision_instruction)

    content = [
        {"type": "image_url", "image_url": {"url": f"data:{media_type};base64,{image_b64}"}},
        {"type": "text", "text": user_input_en},
    ]
    messages = [system_prompt] + st.session_state.get("memory", []) + [HumanMessage(content=content)]

    try:
        response = llm.invoke(messages)
        text = response.content if isinstance(response.content, str) else ""
        return text, []
    except Exception as e:
        return f"Vision error: {e}", []


# ── Tool dispatch ─────────────────────────────────────────────────────────────

def _dispatch_tool(tool_call) -> dict:
    if tool_call["name"] == "generate_pdf_tool":
        try:
            result = generate_pdf_tool.invoke(tool_call["args"])
            pdf_path = None
            match = re.search(r"outputs[/\\][^\s]+\.pdf", result)
            if match:
                pdf_path = match.group(0)
            return {"message": f"✅ {result}", "pdf_path": pdf_path}
        except Exception as e:
            return {"message": f"❌ Tool Error: {e}", "pdf_path": None}
    return {"message": "⚠️ Unknown tool", "pdf_path": None}


# ── PDF download ──────────────────────────────────────────────────────────────

def show_pdf_download(pdf_path: str):
    if pdf_path and os.path.exists(pdf_path):
        with open(pdf_path, "rb") as f:
            pdf_bytes = f.read()
        filename = os.path.basename(pdf_path)
        st.download_button(
            label="📥 Download PDF Report",
            data=pdf_bytes,
            file_name=filename,
            mime="application/pdf",
            key=f"dl_{filename}",
        )
    else:
        st.warning(f"PDF not found at: {pdf_path}")


# ══════════════════════════════════════════════════════════════════════════════
# Document Manager (sidebar)
# ══════════════════════════════════════════════════════════════════════════════

def _render_document_manager():
    st.sidebar.markdown("---")
    st.sidebar.subheader("📚 Document Collections")

    cols = list_collections()
    active_name = st.session_state.get("active_col_name", "")

    if active_name:
        st.sidebar.success(f"📖 **{active_name}**")
    else:
        st.sidebar.warning("⚠️ No collection loaded")

    if cols:
        names = [c["name"] for c in cols]
        try:   idx = names.index(active_name)
        except ValueError: idx = 0

        chosen = st.sidebar.selectbox(
            "Switch collection", names, index=idx,
            key="col_switcher", label_visibility="collapsed",
        )
        if chosen != active_name:
            _switch_collection(chosen)
            st.rerun()

        meta = next((c for c in cols if c["name"] == chosen), None)
        if meta:
            st.sidebar.caption(
                f"📄 {meta.get('doc_count','?')} file(s) · "
                f"{meta.get('chunk_count','?')} chunks"
            )

    with st.sidebar.expander("➕ Upload New Documents"):
        col_name_input = st.text_input(
            "Collection name", placeholder="e.g. TCS Annual 2024",
            key="new_col_name",
        )
        uploaded_pdfs = st.file_uploader(
            "Upload PDF(s)", type=["pdf"],
            accept_multiple_files=True, key="pdf_uploader",
        )
        if st.button("🔨 Build Collection", key="build_col_btn",
                     disabled=not (col_name_input and uploaded_pdfs)):
            log_box = st.empty()
            logs = []
            def cb(msg, _logs=logs, _box=log_box):
                _logs.append(msg)
                _box.info("\n".join(_logs))
            with st.spinner("Building vector index…"):
                ok, msg = ingest_uploaded_files(
                    uploaded_pdfs, col_name_input.strip(), progress_cb=cb
                )
            log_box.empty()
            if ok:
                st.success(msg)
                _switch_collection(col_name_input.strip())
                st.rerun()
            else:
                st.error(msg)

    if cols:
        with st.sidebar.expander("🗑️ Delete Collection"):
            del_choice = st.selectbox(
                "Select", [c["name"] for c in cols],
                key="del_col_sel", label_visibility="collapsed",
            )
            if st.button("Delete", key="del_col_btn", type="secondary"):
                delete_collection(del_choice)
                if st.session_state.get("active_col_name") == del_choice:
                    st.session_state.pop("active_col_name", None)
                    st.session_state.pop("active_col_path", None)
                    st.session_state["messages"] = []
                    st.session_state["memory"]   = []
                st.success(f"Deleted '{del_choice}'")
                st.rerun()

    st.sidebar.markdown("")
    if st.sidebar.button("🧹 Clear Chat History", use_container_width=True):
        user     = current_user()
        username = user["username"] if user else "global"
        col      = st.session_state.get("active_col_name", "default")
        clear_memory(username, col)
        st.session_state["messages"] = []
        st.session_state["memory"]   = []
        st.rerun()


def _switch_collection(name: str):
    path = collection_path(name)
    st.session_state["active_col_name"] = name
    st.session_state["active_col_path"] = path
    user     = current_user()
    username = user["username"] if user else "global"
    raw      = load_memory(username, name)
    st.session_state["messages"] = [
        {"role": "user" if m["type"] == "human" else "assistant",
         "content": m["content"]}
        for m in raw
    ]
    st.session_state["memory"] = to_langchain_messages(raw)


# ── Language sidebar ──────────────────────────────────────────────────────────

def _render_language_sidebar():
    if not MULTILANG:
        return
    st.sidebar.title("🌐 Language Settings")
    lang_names = get_language_names()
    auto_detect = st.sidebar.toggle("🔍 Auto-detect language",
                                    value=st.session_state.get("auto_detect", True))
    st.session_state["auto_detect"] = auto_detect

    sorted_langs = sorted(lang_names.items(), key=lambda x: x[1])
    codes  = [c for c, _ in sorted_langs]
    labels = [l for _, l in sorted_langs]
    current_code = st.session_state.get("selected_lang", "en")
    try:
        default_idx = codes.index(current_code)
    except ValueError:
        default_idx = codes.index("en")

    selected_idx = st.sidebar.selectbox(
        "Manual language override" if auto_detect else "Response language",
        options=range(len(codes)), format_func=lambda i: labels[i], index=default_idx,
    )
    st.session_state["selected_lang"] = codes[selected_idx]

    st.sidebar.markdown("---")
    st.sidebar.subheader("🎤 Voice Settings")
    voice_auto = st.sidebar.toggle("Auto-detect voice language",
                                   value=st.session_state.get("voice_auto", True))
    st.session_state["voice_auto"] = voice_auto
    if not voice_auto:
        voice_idx = st.sidebar.selectbox(
            "Spoken language", options=range(len(codes)),
            format_func=lambda i: labels[i], index=default_idx,
        )
        st.session_state["voice_lang"] = codes[voice_idx]
    else:
        st.session_state["voice_lang"] = None

    st.sidebar.markdown("---")
    st.sidebar.info(f"**{len(SUPPORTED_LANGUAGES)}+ languages supported**")
    last = st.session_state.get("detected_lang")
    if last and last in SUPPORTED_LANGUAGES:
        info = SUPPORTED_LANGUAGES[last]
        st.sidebar.success(f"🔎 Last detected: **{info['name']}** ({info['native']})")


# ══════════════════════════════════════════════════════════════════════════════
# ── NEW: Recent Chats Tab ─────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════

def _render_recent_chats_tab(user):
    st.subheader("🕘 Recent Chats")
    username = user["username"]
    recents  = list_recent_chats(username)

    if not recents:
        st.info("No recent chats yet. Start a conversation in the Chat tab and it will appear here.")
        return

    for i, chat in enumerate(recents):
        collection  = chat.get("collection", "Unknown")
        preview     = chat.get("preview", "No messages yet")
        turns       = chat.get("turn_count", 0)
        history     = chat.get("history", [])

        # Format timestamp
        try:
            dt       = datetime.fromisoformat(chat["updated_at"])
            time_str = dt.strftime("%b %d, %Y · %I:%M %p")
        except Exception:
            time_str = chat.get("updated_at", "Unknown time")

        # Card layout
        with st.container(border=True):
            col_info, col_btn = st.columns([5, 1])

            with col_info:
                st.markdown(f"#### 📁 {collection}")
                st.caption(f"🕒 {time_str}  ·  💬 {turns} question{'s' if turns != 1 else ''}")
                if preview:
                    trunc = preview[:120] + ("…" if len(preview) > 120 else "")
                    st.markdown(f"> {trunc}")

            with col_btn:
                st.markdown("<br>", unsafe_allow_html=True)   # vertical nudge
                if st.button("↩ Restore", key=f"restore_chat_{i}", use_container_width=True):
                    # Load messages into session state
                    st.session_state["messages"] = [
                        {
                            "role": "user" if m["type"] == "human" else "assistant",
                            "content": m["content"],
                        }
                        for m in history
                    ]
                    st.session_state["memory"] = to_langchain_messages(history)

                    # Switch to matching collection if it still exists
                    if collection_exists(collection):
                        _switch_collection(collection)
                    else:
                        st.warning(f"Collection '{collection}' no longer exists — conversation restored without switching.")

                    st.session_state["_active_tab"] = "chat"   # signal to jump to Chat tab
                    st.rerun()

        # Expandable preview of the full conversation
        if history:
            with st.expander(f"👁 Preview conversation ({len(history)} messages)"):
                for msg in history:
                    role  = "🧑 You" if msg["type"] == "human" else "🤖 DocuBot"
                    color = "#2d5a8e" if msg["type"] == "human" else "#2d6b4a"
                    st.markdown(
                        f'<div style="background:{color};border-radius:8px;'
                        f'padding:8px 12px;margin:4px 0;">'
                        f'<strong>{role}:</strong> {msg["content"][:300]}'
                        f'{"…" if len(msg["content"]) > 300 else ""}</div>',
                        unsafe_allow_html=True,
                    )


# ══════════════════════════════════════════════════════════════════════════════
# ── Main ──────────────────────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════

def main():
    # ── Auth gate ─────────────────────────────────────────────────────────────
    if not is_logged_in():
        result = render_login_page()
        if result:
            st.rerun()
        return

    # ── Sidebar: logout + user info ───────────────────────────────────────────
    user = current_user()
    st.sidebar.markdown(f"### 👤 {user['name']}")
    st.sidebar.caption(f"Role: **{user['role']}**")
    if st.sidebar.button("🚪 Logout", use_container_width=True):
        logout()
        st.rerun()
    st.sidebar.markdown("---")

    if user["role"] == "admin":
        with st.sidebar.expander("⚙️ Manage Users"):
            render_user_management()
        st.sidebar.markdown("---")

    _render_language_sidebar()

    st.title("🤖 DocuBot")
    st.caption("Intelligent Document Analysis")

    # ── Session defaults ──────────────────────────────────────────────────────
    for key, default in [
        ("messages",        []),
        ("memory",          []),
        ("last_audio_hash", ""),
        ("detected_lang",   "en"),
        ("selected_lang",   "en"),
        ("auto_detect",     True),
        ("voice_auto",      True),
        ("voice_lang",      None),
        ("pending_image",   None),
        ("active_col_name", ""),
        ("active_col_path", ""),
        ("_memory_loaded",  False),
        ("_active_tab",     "chat"),   # ← NEW: track which tab to show
    ]:
        if key not in st.session_state:
            st.session_state[key] = default

    # ── Auto-select first available collection on first load ──────────────────
    if not st.session_state["active_col_name"]:
        cols = list_collections()
        if cols:
            _switch_collection(cols[0]["name"])
        elif os.path.exists("vectordb"):
            st.session_state["active_col_path"] = "vectordb"
            st.session_state["active_col_name"]  = "Default"

    # ── Restore chat history from disk on first load ──────────────────────────
    if not st.session_state["_memory_loaded"]:
        if "messages" not in st.session_state or not st.session_state["messages"]:
            col_name = st.session_state.get("active_col_name", "default")
            username = user["username"]
            raw = load_memory(username, col_name)
            st.session_state["messages"] = [
                {"role": "user" if m["type"] == "human" else "assistant",
                 "content": m["content"]}
                for m in raw
            ]
            st.session_state["memory"] = to_langchain_messages(raw)
        st.session_state["_memory_loaded"] = True

    # ── Render document manager in sidebar ────────────────────────────────────
    _render_document_manager()

    # ══════════════════════════════════════════════════════════════════════════
    # TABS  ← only change from original main()
    # ══════════════════════════════════════════════════════════════════════════
    tab_chat, tab_recents = st.tabs(["💬 Chat", "🕘 Recent Chats"])

    # ── Recent Chats tab ──────────────────────────────────────────────────────
    with tab_recents:
        _render_recent_chats_tab(user)

    # ── Chat tab (everything from original main() after _render_document_manager) ──
    with tab_chat:

        vectorstore, reranker, llm = load_rag_resources()
        if vectorstore is None:
            st.info("📂 Upload documents using the sidebar to get started.")
            # Still allow recent chats to be browsed even with no collection loaded
            return

        # ── Display history ───────────────────────────────────────────────────
        for idx, msg in enumerate(st.session_state.messages):
            with st.chat_message(msg["role"]):
                if msg.get("image_b64"):
                    img_bytes = base64.b64decode(msg["image_b64"])
                    st.image(img_bytes, caption="📎 Attached image", width=300)
                render_message(msg["content"])

            if msg["role"] == "assistant":
                render_translate_button(idx, msg["content"], llm=llm)

            if msg.get("pdf_path"):
                show_pdf_download(msg["pdf_path"])

        # ── Image uploader (sidebar) ──────────────────────────────────────────
        st.sidebar.markdown("---")
        st.sidebar.subheader("🖼️ Image Analysis")
        uploaded_image = st.sidebar.file_uploader(
            "Upload an image to analyse",
            type=SUPPORTED_IMAGE_TYPES,
            help="Upload a chart, diagram, screenshot, or any image. Then ask a question about it below.",
        )

        if uploaded_image is not None:
            img_bytes  = uploaded_image.read()
            media_type = f"image/{uploaded_image.type.split('/')[-1]}"
            if media_type == "image/jpg":
                media_type = "image/jpeg"
            img_b64 = encode_image_base64(img_bytes, media_type)
            st.sidebar.image(img_bytes, caption=uploaded_image.name, use_container_width=True)
            st.session_state["pending_image"] = {
                "b64":        img_b64,
                "media_type": media_type,
                "name":       uploaded_image.name,
            }
            st.sidebar.success("✅ Image ready — ask your question below!")
        else:
            st.session_state["pending_image"] = None

        if st.session_state.get("pending_image"):
            col1, col2 = st.columns([6, 1])
            col1.info(f"🖼️ Image **{st.session_state['pending_image']['name']}** attached — ask your question below.")
            if col2.button("✖️", help="Remove image"):
                st.session_state["pending_image"] = None
                st.rerun()

        # ── Input ─────────────────────────────────────────────────────────────
        user_input = None
        input_lang = "en"

        audio_bytes = audio_recorder(text="🎤 Click to speak", key="voice")
        if audio_bytes:
            import hashlib as _hl
            audio_hash = _hl.md5(audio_bytes).hexdigest()
            last_hash  = st.session_state.get("last_audio_hash", "")

            if audio_hash != last_hash and len(audio_bytes) > 1000:
                st.session_state["last_audio_hash"] = audio_hash

                voice_lang_hint = None
                if MULTILANG and not st.session_state.get("voice_auto", True):
                    voice_code = st.session_state.get("voice_lang")
                    if voice_code:
                        voice_lang_hint = get_whisper_language_hint(voice_code)

                with st.spinner("🎤 Transcribing your voice…"):
                    try:
                        voice_text, detected_voice_lang = transcribe_audio(
                            audio_bytes, lang_hint=voice_lang_hint
                        )
                    except Exception as e:
                        st.error(f"Transcription error: {e}")
                        voice_text          = ""
                        detected_voice_lang = "en"

                if voice_text and voice_text.strip():
                    input_lang = detected_voice_lang
                    lang_label = ""
                    if MULTILANG and input_lang in SUPPORTED_LANGUAGES:
                        lang_label = f"[{SUPPORTED_LANGUAGES[input_lang]['name']}] "
                    st.info(f"🎤 Heard: {lang_label}**{voice_text}**")
                    user_input = voice_text.strip()
                else:
                    st.warning("🎤 Could not hear anything clearly. Please try again.")

        typed_input = st.chat_input("Ask something…")
        if typed_input:
            user_input = typed_input.strip()
            if MULTILANG:
                input_lang = (
                    detect_language(user_input)
                    if st.session_state.get("auto_detect", True)
                    else st.session_state.get("selected_lang", "en")
                )
            else:
                try:
                    from langdetect import detect
                    input_lang = detect(user_input)
                except Exception:
                    input_lang = "en"

        # ── Process ───────────────────────────────────────────────────────────
        if user_input:
            response_lang = (
                (input_lang if st.session_state.get("auto_detect", True)
                 else st.session_state.get("selected_lang", "en"))
                if MULTILANG else input_lang
            )
            st.session_state["detected_lang"] = input_lang

            # Translate input → English for the LLM
            if input_lang != "en":
                if MULTILANG:
                    with st.spinner("🌐 Translating input…"):
                        query_en = translate_text(user_input, input_lang, "en")
                else:
                    try:
                        from deep_translator import GoogleTranslator
                        query_en = GoogleTranslator(source="auto", target="en").translate(user_input)
                    except Exception:
                        query_en = user_input
            else:
                query_en = user_input

            pending_image = st.session_state.get("pending_image")

            user_msg = {"role": "user", "content": user_input}
            if pending_image:
                user_msg["image_b64"] = pending_image["b64"]
            st.session_state.messages.append(user_msg)
            st.session_state.memory.append(HumanMessage(content=user_input))

            with st.chat_message("user"):
                if pending_image:
                    img_bytes = base64.b64decode(pending_image["b64"])
                    st.image(img_bytes, caption=f"📎 {pending_image['name']}", width=300)
                st.markdown(user_input)

            with st.spinner("🤔 Thinking…"):
                if pending_image:
                    response_text_en, tool_results = _run_vision_turn(
                        query_en,
                        pending_image["b64"],
                        pending_image["media_type"],
                        vectorstore, reranker, llm,
                        response_lang=response_lang,
                    )
                    st.session_state["pending_image"] = None
                else:
                    response_text_en, tool_results = _run_rag_turn(query_en, vectorstore, reranker, llm)

            if response_lang != "en" and response_text_en:
                with st.spinner("🌐 Translating response…"):
                    response_text = safe_translate(response_text_en, response_lang, llm=llm)
            else:
                response_text = response_text_en

            if response_text:
                st.session_state.messages.append({"role": "assistant", "content": response_text})
                st.session_state.memory.append(AIMessage(content=response_text_en))

                # Persist to disk
                _username = user["username"]
                _col      = st.session_state.get("active_col_name", "default")
                save_memory(st.session_state["memory"], _username, _col)

                with st.chat_message("assistant"):
                    render_message(response_text)

                new_msg_index = len(st.session_state.messages) - 1
                render_translate_button(new_msg_index, response_text, llm=llm)

            if tool_results:
                for r in tool_results:
                    st.info(r["message"])
                    if r.get("pdf_path"):
                        if st.session_state.messages and st.session_state.messages[-1]["role"] == "assistant":
                            st.session_state.messages[-1]["pdf_path"] = r["pdf_path"]
                        show_pdf_download(r["pdf_path"])

            st.rerun()


if __name__ == "__main__":
    main()