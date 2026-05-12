"""
translation_utils.py
--------------------
Robust translation layer for DocuBot.
- Detects language with langdetect (+ fallback heuristics)
- Translates via deep-translator (GoogleTranslator)
- Caches recent translations to avoid redundant API calls
- Handles RTL, long texts (chunked), and unsupported pairs
"""
 
import re
import hashlib
import logging
from functools import lru_cache
 
from langdetect import detect, LangDetectException
from deep_translator import GoogleTranslator
 
from language_config import normalize_lang_code, SUPPORTED_LANGUAGES
 
logger = logging.getLogger(__name__)
 
# ── Cache: (text_hash, src, tgt) → translated string ────────────────────────
_translation_cache: dict[tuple, str] = {}
MAX_CACHE_SIZE = 256          # evict oldest when full
CHUNK_SIZE = 4500             # GoogleTranslator limit is ~5000 chars
 
 
# ── RTL languages ────────────────────────────────────────────────────────────
RTL_LANGUAGES = {"ar", "fa", "he", "ur"}
 
 
def is_rtl(lang_code: str) -> bool:
    return lang_code in RTL_LANGUAGES
 
 
# ── Language detection ───────────────────────────────────────────────────────
 
def detect_language(text: str) -> str:
    """
    Detect the language of *text* and return a normalized registry code.
    Falls back to 'en' on any error or very short inputs.
    """
    if not text or len(text.strip()) < 5:
        return "en"
    try:
        raw = detect(text)
        return normalize_lang_code(raw)
    except LangDetectException:
        return "en"
    except Exception as exc:
        logger.warning("Language detection error: %s", exc)
        return "en"
 
 
# ── Core translation ─────────────────────────────────────────────────────────
 
def _translate_chunk(text: str, source: str, target: str) -> str:
    """Translate a single chunk (≤ CHUNK_SIZE chars)."""
    # Map zh-cn/zh-tw to what Google Translator expects
    _src = "zh-CN" if source == "zh-cn" else ("zh-TW" if source == "zh-tw" else source)
    _tgt = "zh-CN" if target == "zh-cn" else ("zh-TW" if target == "zh-tw" else target)
    try:
        return GoogleTranslator(source=_src, target=_tgt).translate(text)
    except Exception as exc:
        logger.warning("Translation chunk failed (%s→%s): %s", source, target, exc)
        return text  # return original on failure
 
 
def translate_text(text: str, source_lang: str, target_lang: str) -> str:
    """
    Translate *text* from *source_lang* to *target_lang*.
 
    - Returns original text if source == target or translation not needed.
    - Splits long texts into chunks automatically.
    - Uses in-memory cache keyed by (text_hash, src, tgt).
    """
    if not text or source_lang == target_lang:
        return text
 
    # Validate target is supported
    if target_lang not in SUPPORTED_LANGUAGES:
        logger.warning("Unsupported target language '%s', skipping translation", target_lang)
        return text
 
    # Cache key
    text_hash = hashlib.md5(text.encode()).hexdigest()
    cache_key = (text_hash, source_lang, target_lang)
 
    if cache_key in _translation_cache:
        return _translation_cache[cache_key]
 
    # Chunk long texts
    if len(text) <= CHUNK_SIZE:
        result = _translate_chunk(text, source_lang, target_lang)
    else:
        # Split on paragraph boundaries first, then by size
        paragraphs = text.split("\n\n")
        chunks, current = [], ""
        for para in paragraphs:
            if len(current) + len(para) + 2 <= CHUNK_SIZE:
                current = (current + "\n\n" + para).lstrip()
            else:
                if current:
                    chunks.append(current)
                current = para
        if current:
            chunks.append(current)
 
        translated_chunks = [_translate_chunk(c, source_lang, target_lang) for c in chunks]
        result = "\n\n".join(translated_chunks)
 
    # Manage cache size
    if len(_translation_cache) >= MAX_CACHE_SIZE:
        oldest = next(iter(_translation_cache))
        del _translation_cache[oldest]
 
    _translation_cache[cache_key] = result
    return result
 
 
# ── Whisper hint ─────────────────────────────────────────────────────────────
 
def get_whisper_language_hint(lang_code: str) -> str | None:
    """
    Return the Whisper language code for *lang_code*, or None (→ auto-detect).
    Auto-detect is used when language is unknown or set to 'auto'.
    """
    if lang_code in ("auto", "en"):
        return None   # Let Whisper decide for English / auto mode
    info = SUPPORTED_LANGUAGES.get(lang_code)
    return info["whisper"] if info else None
 
 
# ── Convenience: detect + translate in one call ───────────────────────────────
 
def auto_translate_to_english(text: str) -> tuple[str, str]:
    """
    Detect language of *text* and translate to English if needed.
 
    Returns:
        (translated_text, detected_lang_code)
    """
    lang = detect_language(text)
    if lang == "en":
        return text, "en"
    translated = translate_text(text, lang, "en")
    return translated, lang
 
 
def translate_response(response: str, target_lang: str) -> str:
    """
    Translate an English *response* to *target_lang*.
    Skips chart code blocks to avoid corrupting JSON.
    """
    if target_lang == "en" or not response:
        return response
 
    # Preserve ```chartjs ... ``` blocks verbatim
    CHART_PATTERN = re.compile(r"(```chartjs[\s\S]*?```)", re.MULTILINE)
    parts = CHART_PATTERN.split(response)
 
    translated_parts = []
    for part in parts:
        if CHART_PATTERN.match(part):
            translated_parts.append(part)  # keep chart block as-is
        else:
            translated_parts.append(translate_text(part, "en", target_lang))
 
    return "".join(translated_parts)
