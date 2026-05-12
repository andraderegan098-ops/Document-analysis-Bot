# 🤖 DocuBot — Intelligent Multimodal RAG Assistant

DocuBot is an advanced **multimodal AI-powered document assistant** built using **Streamlit**, **LangChain**, **OpenAI/Gemini**, and **Retrieval-Augmented Generation (RAG)**.

It combines:

* 📄 Document intelligence
* 🖼️ Image understanding
* 🎤 Voice interaction
* 🌍 Multilingual communication
* 📊 Interactive chart rendering
* 📑 PDF report generation
* 🔧 Tool calling
* 💬 Conversational AI memory

into one unified AI assistant.

---

# 📖 Detailed Project Overview

DocuBot is designed as a next-generation AI assistant capable of understanding:

* Text documents
* Images
* Voice input
* Multiple languages
* Analytical data
* User conversations

Unlike traditional chatbots that only generate text, DocuBot combines:

* Retrieval-Augmented Generation (RAG)
* Multimodal reasoning
* Tool execution
* Interactive visualizations
* Conversational memory

This creates a complete enterprise-style AI assistant capable of intelligent document analysis and contextual reasoning.

---

# 🚀 Features

## 📄 Retrieval-Augmented Generation (RAG)

### What It Does

The system:

1. Converts documents into embeddings
2. Stores embeddings inside a vector database
3. Retrieves relevant chunks for user queries
4. Injects retrieved context into the LLM
5. Generates grounded answers

### Benefits

* Reduces hallucinations
* Improves factual accuracy
* Enables document-aware AI responses
* Supports enterprise knowledge systems

### Workflow

```text
User Question
↓
Embedding Search
↓
Top Relevant Chunks
↓
Reranking
↓
Context Injection
↓
LLM Response
```

---

## 🖼️ Vision + Image Understanding

### Supported Inputs

* Charts
* Graphs
* Screenshots
* Tables
* Technical diagrams
* Reports
* Infographics

### Vision Workflow

```text
Image Upload
↓
Image Encoding
↓
Vision LLM Processing
↓
Optional RAG Context Injection
↓
AI Analysis
```

### Why It Matters

Most AI assistants only support text.

DocuBot supports:

* Text
* Images
* Voice
* Documents
* Charts

making it a fully multimodal AI platform.

---

## 🎤 Voice Input

### Voice Pipeline

```text
User Speech
↓
Audio Recording
↓
Whisper Transcription
↓
Language Detection
↓
LLM Processing
↓
AI Response
```

### Technologies Used

* Faster Whisper
* Streamlit Audio Recorder

### Features

* Real-time speech recognition
* Multilingual transcription
* Voice-based querying

---

## 🌍 Multilingual AI

### Supported Languages

DocuBot supports 50+ languages including:

* English
* Hindi
* Kannada
* Tamil
* Telugu
* Malayalam
* Spanish
* French
* German
* Chinese
* Japanese
* Arabic

### Translation Workflow

```text
Input Language
↓
Language Detection
↓
Translate to English
↓
RAG + LLM Processing
↓
Translate Response Back
↓
Display Final Response
```

### Why English Internally?

Most embedding models and LLMs perform best in English.

The translation pipeline improves:

* Retrieval quality
* Embedding accuracy
* LLM reasoning performance

---

## 📊 Interactive Chart Rendering

### Chart Types Supported

* Bar charts
* Pie charts
* Line charts
* Doughnut charts
* Analytical dashboards

### Workflow

```text
LLM Generates ChartJS JSON
↓
Regex Extraction
↓
JSON Validation + Repair
↓
Chart.js Rendering
↓
Interactive Visualization
```

### Example

````markdown
```chartjs
{
  "type": "bar",
  "data": {
    "labels": ["Q1", "Q2", "Q3"],
    "datasets": [{
      "label": "Revenue",
      "data": [120, 200, 150]
    }]
  }
}
```
````

### Why This Is Powerful

Instead of static text responses, the AI generates:

* interactive visualizations
* analytical dashboards
* business-style charts

---

## 📑 PDF Report Generation

### Features

* AI-generated reports
* Downloadable PDFs
* Structured summaries
* Business-style analytics reports

### Workflow

```text
User Request
↓
LLM Decides Tool Usage
↓
PDF Tool Invocation
↓
PDF Generation
↓
Download Button
```

### Importance

This transforms the system from:

```text
Simple chatbot
```

into:

```text
AI productivity platform
```

---

## 🔧 Tool Calling

### What Is Tool Calling?

Modern LLMs can:

* invoke functions
* execute tools
* perform actions automatically

### Current Tools

* PDF generation
* Dynamic workflows

### Future Expandability

The tool architecture supports future integrations like:

* Web search
* Database queries
* APIs
* OCR systems
* External automation

---

## 💬 Conversational Memory

### Memory System

DocuBot stores:

* user messages
* AI responses
* conversation context

using:

* Streamlit session state
* LangChain message objects

### Benefits

* Context-aware conversations
* Multi-turn reasoning
* Follow-up question handling
* Natural interaction flow

---

# 🧠 System Architecture

```text
                    ┌────────────────────┐
                    │      User Input     │
                    └─────────┬──────────┘
                              │
             ┌────────────────┼────────────────┐
             │                                 │
        Text Input                        Voice Input
             │                                 │
             └──────────────┬──────────────────┘
                            │
                    Language Detection
                            │
                       Translation
                            │
            ┌───────────────┴────────────────┐
            │                                │
       Text Query                      Image Upload
            │                                │
            │                        Vision Processing
            │                                │
            └──────────────┬─────────────────┘
                           │
                    Retrieval System
                           │
                 Vector DB + Reranker
                           │
                    Context Building
                           │
                     LLM Reasoning
                           │
        ┌──────────────────┼──────────────────┐
        │                  │                  │
    Charts             PDF Tools         Responses
        │
   Chart.js Rendering
```

---

# 🏗️ Tech Stack

## Frontend

* Streamlit

## AI Models

* OpenAI GPT-4.1
* Google Gemini 2.5 Flash

## AI Frameworks

* LangChain

## Vector Database

* FAISS
* Embedding-based retrieval

## Speech Recognition

* Faster Whisper

## Translation

* Deep Translator
* Custom translation pipeline

## Image Processing

* Pillow (PIL)

## Visualization

* Chart.js

---

# 📂 Project Structure

```bash
DocuBot/
│
├── app.py
├── retrieval.py
├── prompts.py
├── tools.py
├── utils.py
├── ingest.py
├── language_config.py
├── translation_utils.py
│
├── data/
│   └── documents/
│
├── outputs/
│
├── vectorstore/
│
├── requirements.txt
├── README.md
└── .env
```

---

# ⚙️ Installation

## 1. Clone Repository

```bash
git clone https://github.com/yourusername/docubot.git
cd docubot
```

---

## 2. Create Virtual Environment

### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

### Linux / Mac

```bash
python3 -m venv venv
source venv/bin/activate
```

---

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

# 🔑 Environment Variables

Create a `.env` file:

```env
OPENAI_API_KEY=your_openai_api_key
GOOGLE_API_KEY=your_google_api_key
```

---

# 📄 Document Ingestion

Place documents inside:

```bash
data/documents/
```

Run ingestion:

```bash
python ingest.py
```

### Ingestion Process

1. Documents loaded
2. Content chunked
3. Embeddings generated
4. Vectors stored
5. Retrieval index created

---

# ▶️ Running the Application

```bash
streamlit run app.py
```

---

# 🖼️ Vision Workflow

## Steps

1. Upload image from sidebar
2. Ask questions about the image
3. AI processes image + context
4. Response generated

### Example Queries

```text
Explain this graph
Analyze this screenshot
Describe this chart
Summarize this diagram
```

---

# 🎤 Voice Workflow

## Steps

1. Click microphone button
2. Speak naturally
3. Whisper transcribes speech
4. AI processes query
5. Response displayed

---

# 📊 Chart Rendering System

## Internal Workflow

```text
AI Generates Chart JSON
↓
Regex Extraction
↓
JSON Repair
↓
HTML Injection
↓
Chart.js Rendering
↓
Interactive Chart Display
```

### Error Handling

The system automatically repairs:

* malformed JSON
* trailing commas
* smart quotes
* invalid formatting

---

# 📑 PDF Generation Workflow

```text
User Request
↓
LLM Tool Decision
↓
Tool Invocation
↓
PDF File Creation
↓
Download Button Display
```

---

# 🔥 Advanced AI Features

## ✅ Multimodal RAG

Combines:

* vision understanding
* document retrieval
* contextual reasoning

This is one of the most advanced architectural features of the system.

---

## ✅ Tool-Augmented AI

The assistant can:

* reason
* retrieve information
* invoke tools
* generate files
* create visualizations

making it an agentic AI platform.

---

## ✅ Conversational AI

Supports:

* persistent memory
* contextual follow-ups
* multi-turn interactions

---

## ✅ Dynamic Visualization

Automatically generates and renders:

* charts
* dashboards
* visual analytics

---

# 🛡️ Error Handling

The system includes:

* JSON repair logic
* API fallback handling
* Translation fallback logic
* Safe file handling
* PDF validation
* Vision error handling
* Graceful chart failure recovery

---

# 🧪 Example Queries

```text
Summarize this document
Explain this chart
Analyze this screenshot
Generate a PDF report
Create a sales graph
Translate this response into Hindi
Describe the uploaded image
```

---

# 🚀 Future Improvements

Planned features:

* Hybrid search (BM25 + vector retrieval)
* OCR support
* Streaming responses
* Live web search
* Multi-file upload
* Source citations
* Knowledge graphs
* Agentic workflows
* User authentication
* Database-backed memory
* Cloud deployment
* Team collaboration
* Real-time dashboards

---

# 🌟 What Makes DocuBot Special

Most AI projects only support:

```text
Text → AI Response
```

DocuBot supports:

```text
+ Voice
+ Vision
+ Translation
+ Retrieval
+ Tool Calling
+ Charts
+ PDF Generation
+ Conversational Memory
```

This combination creates a highly advanced multimodal AI system.

---

## Steps

1. Fork repository
2. Create feature branch
3. Commit changes
4. Push branch
5. Open Pull Request

---

# 📜 License

MIT License

---

# 👨‍💻 Author

Regan Andrade

---

# ⭐ Acknowledgements

Built using:

* Streamlit
* LangChain
* OpenAI
* Google Gemini
* Faster Whisper
* Chart.js

Technologies Used

| Tech                                                                               | Purpose               |
| ---------------------------------------------------------------------------------- | --------------------- |
| [Streamlit](https://streamlit.io?utm_source=chatgpt.com)                           | Frontend              |
| [LangChain](https://www.langchain.com?utm_source=chatgpt.com)                      | LLM orchestration     |
| [OpenAI API](https://platform.openai.com?utm_source=chatgpt.com)                   | GPT models            |
| [Google Gemini API](https://ai.google.dev?utm_source=chatgpt.com)                  | Gemini models         |
| [Whisper](https://github.com/openai/whisper?utm_source=chatgpt.com)                | Speech recognition    |
| [Chart.js](https://www.chartjs.org?utm_source=chatgpt.com)                         | Chart rendering       |
| [FAISS](https://faiss.ai?utm_source=chatgpt.com)                                   | Vector database       |
| [Faster-Whisper](https://github.com/SYSTRAN/faster-whisper?utm_source=chatgpt.com) | Faster speech-to-text |

# 🚀 DocuBot

An advanced multimodal AI assistant for intelligent document analysis, visual reasoning, conversational analytics, and enterprise-style AI workflows.
