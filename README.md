# RAG with LangChain + Groq + Redis

This project demonstrates a simple Retrieval‐Augmented Generation (RAG) pipeline that:

1. **Extracts text** from a `.docx` document  
2. **Chunks** it for semantic similarity  
3. **Indexes** the chunks in Redis as a vector store  
4. **Stores** chat history in Redis  
5. **Queries** a Groq LLM (`deepseek-r1-distill-llama-70b`) via LangChain, combining retrieved chunks + conversation history  

---

## 🚀 Features

- **Document ingestion** from Microsoft Word (`.docx`)  
- **Semantic text splitting** (512‐char chunks with 50‐char overlap)  
- **Embeddings** via `sentence-transformers/all-MiniLM-L6-v2`  
- **Vector store** in Redis (no FAISS, fully persistent)  
- **Chat memory** in Redis  
- **RAG prompt** combining system instructions, history, and retrieved chunks  
- **Groq LLM** integration via `langchain_groq.ChatGroq`  

---

## 📦 Requirements

- Python 3.8+  
- Redis server (v6+) running locally or remotely  
- Twilio account **(not needed here; ignore)**  

Python dependencies (see `requirements.txt`):

```text
langchain>=0.0.200
langchain-groq
python-docx
redis
