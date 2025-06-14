
# 🤖 Agentic RAG Chatbot

A conversational chatbot that intelligently retrieves and summarizes information from a collection of technical PDFs (e.g. LLM papers). It uses an **Agentic RAG (Retrieval-Augmented Generation)** workflow powered by **LangGraph**, **ChromaDB**, **Google Vertex AI**, and **Streamlit**.

## 🚀 Features

- ✅ Chat interface with natural language queries
- ✅ Tool-using agent that decides when to retrieve context or answer directly
- ✅ MMR-based similarity search for diverse relevant results
- ✅ Vertex AI embedding + Gemini 2.0 Flash for fast and accurate answers
- ✅ Contextual source display at query response
- ✅ Memory management via LangGraph thread checkpointing

---

## 🧰 Tech Stack

| Layer        | Tool/Service               |
|-------------|----------------------------|
| UI          | [Streamlit](https://streamlit.io) |
| Agent Graph | [LangGraph](https://github.com/langchain-ai/langgraph) |
| LLM + Embedding | [Vertex AI](https://cloud.google.com/vertex-ai) |
| Vector DB   | [ChromaDB](https://www.trychroma.com) |
| Checkpointing | `MemorySaver` (in-memory state) |

---


## 📦 Requirements

- Python 3.12 (recommended)
- Install dependencies:

```bash
pip install -r requirements.txt
```

---

## 🔐 Authentication (Vertex AI)

1. Go to Google Cloud Console and create a **service account key** with Vertex AI permissions.
2. Download the JSON key file
3. Place it in your root project directory.
4. The script sets it using:

```python
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "vertexai-client-api.json" # change to your key file name
```

---


## Create Vector Database

Run the script to load and embed the PDF into a local vector store:

```bash
python create_vectordb.py
```

## Launch the Chatbot

Start the chatbot interface using Streamlit:

```bash
streamlit run agentic_rag.py
