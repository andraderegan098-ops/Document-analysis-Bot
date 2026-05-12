"""
language_config.py
------------------
Central registry of 50+ supported languages for DocuBot.
Each entry maps a BCP-47/ISO-639-1 code to display name,
Whisper language code, and deep-translator code.
"""
 
# ── Full language table ──────────────────────────────────────────────────────
# Keys  : ISO-639-1 / BCP-47 codes used by langdetect + deep-translator
# whisper_code : code passed to faster-whisper (None = auto-detect)
# native_name  : shown in the UI language selector
 
SUPPORTED_LANGUAGES = {
    # ── European ─────────────────────────────────────────────────────────────
    "en": {"name": "English",              "native": "English",            "whisper": "en"},
    "fr": {"name": "French",               "native": "Français",           "whisper": "fr"},
    "de": {"name": "German",               "native": "Deutsch",            "whisper": "de"},
    "es": {"name": "Spanish",              "native": "Español",            "whisper": "es"},
    "it": {"name": "Italian",              "native": "Italiano",           "whisper": "it"},
    "pt": {"name": "Portuguese",           "native": "Português",          "whisper": "pt"},
    "nl": {"name": "Dutch",                "native": "Nederlands",         "whisper": "nl"},
    "pl": {"name": "Polish",               "native": "Polski",             "whisper": "pl"},
    "ru": {"name": "Russian",              "native": "Русский",            "whisper": "ru"},
    "uk": {"name": "Ukrainian",            "native": "Українська",         "whisper": "uk"},
    "cs": {"name": "Czech",                "native": "Čeština",            "whisper": "cs"},
    "sk": {"name": "Slovak",               "native": "Slovenčina",         "whisper": "sk"},
    "hu": {"name": "Hungarian",            "native": "Magyar",             "whisper": "hu"},
    "ro": {"name": "Romanian",             "native": "Română",             "whisper": "ro"},
    "bg": {"name": "Bulgarian",            "native": "Български",          "whisper": "bg"},
    "hr": {"name": "Croatian",             "native": "Hrvatski",           "whisper": "hr"},
    "sr": {"name": "Serbian",              "native": "Српски",             "whisper": "sr"},
    "sv": {"name": "Swedish",              "native": "Svenska",            "whisper": "sv"},
    "da": {"name": "Danish",               "native": "Dansk",              "whisper": "da"},
    "fi": {"name": "Finnish",              "native": "Suomi",              "whisper": "fi"},
    "no": {"name": "Norwegian",            "native": "Norsk",              "whisper": "no"},
    "el": {"name": "Greek",                "native": "Ελληνικά",           "whisper": "el"},
    "tr": {"name": "Turkish",              "native": "Türkçe",             "whisper": "tr"},
    "ca": {"name": "Catalan",              "native": "Català",             "whisper": "ca"},
    "lv": {"name": "Latvian",              "native": "Latviešu",           "whisper": "lv"},
    "lt": {"name": "Lithuanian",           "native": "Lietuvių",           "whisper": "lt"},
    "et": {"name": "Estonian",             "native": "Eesti",              "whisper": "et"},
    "sl": {"name": "Slovenian",            "native": "Slovenščina",        "whisper": "sl"},
    "ga": {"name": "Irish",                "native": "Gaeilge",            "whisper": "ga"},
    "cy": {"name": "Welsh",                "native": "Cymraeg",            "whisper": "cy"},
 
    # ── South Asian ──────────────────────────────────────────────────────────
    "hi": {"name": "Hindi",                "native": "हिंदी",               "whisper": "hi"},
    "bn": {"name": "Bengali",              "native": "বাংলা",               "whisper": "bn"},
    "te": {"name": "Telugu",               "native": "తెలుగు",              "whisper": "te"},
    "mr": {"name": "Marathi",              "native": "मराठी",               "whisper": "mr"},
    "ta": {"name": "Tamil",                "native": "தமிழ்",               "whisper": "ta"},
    "kn": {"name": "Kannada",              "native": "ಕನ್ನಡ",               "whisper": "kn"},
    "ml": {"name": "Malayalam",            "native": "മലയാളം",              "whisper": "ml"},
    "gu": {"name": "Gujarati",             "native": "ગુજરાતી",             "whisper": "gu"},
    "pa": {"name": "Punjabi",              "native": "ਪੰਜਾਬੀ",              "whisper": "pa"},
    "ur": {"name": "Urdu",                 "native": "اردو",               "whisper": "ur"},
    "ne": {"name": "Nepali",               "native": "नेपाली",              "whisper": "ne"},
    "si": {"name": "Sinhala",              "native": "සිංහල",               "whisper": "si"},
 
    # ── East / Southeast Asian ───────────────────────────────────────────────
    "zh-cn": {"name": "Chinese (Simplified)",  "native": "中文(简体)",      "whisper": "zh"},
    "zh-tw": {"name": "Chinese (Traditional)", "native": "中文(繁體)",      "whisper": "zh"},
    "ja": {"name": "Japanese",             "native": "日本語",              "whisper": "ja"},
    "ko": {"name": "Korean",               "native": "한국어",              "whisper": "ko"},
    "vi": {"name": "Vietnamese",           "native": "Tiếng Việt",         "whisper": "vi"},
    "th": {"name": "Thai",                 "native": "ภาษาไทย",             "whisper": "th"},
    "id": {"name": "Indonesian",           "native": "Bahasa Indonesia",   "whisper": "id"},
    "ms": {"name": "Malay",                "native": "Bahasa Melayu",      "whisper": "ms"},
    "tl": {"name": "Filipino",             "native": "Filipino",           "whisper": "tl"},
    "my": {"name": "Burmese",              "native": "မြန်မာဘာသာ",          "whisper": "my"},
    "km": {"name": "Khmer",                "native": "ភាសាខ្មែរ",           "whisper": "km"},
 
    # ── Middle East / African ────────────────────────────────────────────────
    "ar": {"name": "Arabic",               "native": "العربية",            "whisper": "ar"},
    "fa": {"name": "Persian (Farsi)",      "native": "فارسی",              "whisper": "fa"},
    "he": {"name": "Hebrew",               "native": "עברית",              "whisper": "he"},
    "sw": {"name": "Swahili",              "native": "Kiswahili",          "whisper": "sw"},
    "am": {"name": "Amharic",              "native": "አማርኛ",               "whisper": "am"},
    "ha": {"name": "Hausa",                "native": "Hausa",              "whisper": "ha"},
    "yo": {"name": "Yoruba",               "native": "Yorùbá",             "whisper": "yo"},
    "ig": {"name": "Igbo",                 "native": "Igbo",               "whisper": "ig"},
    "zu": {"name": "Zulu",                 "native": "isiZulu",            "whisper": "zu"},
    "af": {"name": "Afrikaans",            "native": "Afrikaans",          "whisper": "af"},
}
 
# ── Helper utilities ─────────────────────────────────────────────────────────
 
def get_language_names() -> dict[str, str]:
    """Return {code: 'English Name (Native Name)'} for UI dropdowns."""
    return {
        code: f"{info['name']} ({info['native']})"
        for code, info in SUPPORTED_LANGUAGES.items()
    }
 
 
def get_whisper_code(lang_code: str) -> str | None:
    """Return the Whisper language code for a given ISO code, or None for auto."""
    info = SUPPORTED_LANGUAGES.get(lang_code)
    return info["whisper"] if info else None
 
 
def normalize_lang_code(raw_code: str) -> str:
    """
    Normalize langdetect/deep-translator output to our registry keys.
    Handles edge cases like 'zh-CN' → 'zh-cn', 'zh' → 'zh-cn'.
    """
    code = raw_code.lower().strip()
 
    # Chinese disambiguation
    if code in ("zh-cn", "zh", "zh_cn"):
        return "zh-cn"
    if code in ("zh-tw", "zh_tw"):
        return "zh-tw"
 
    # Direct hit
    if code in SUPPORTED_LANGUAGES:
        return code
 
    # Try prefix match (e.g. 'pt-br' → 'pt')
    prefix = code.split("-")[0]
    if prefix in SUPPORTED_LANGUAGES:
        return prefix
 
    # Unknown → fall back to English
    return "en"
 
 
# Convenience set of all supported translator target codes
ALL_TARGET_CODES = set(SUPPORTED_LANGUAGES.keys())