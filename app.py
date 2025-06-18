import os
from docx import Document
import requests
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores.redis import Redis as RedisVectorStore
from langchain_groq import ChatGroq
from langchain.memory import RedisChatMessageHistory
from langchain.schema import SystemMessage, HumanMessage, AIMessage

# ——————————————————————————————————————————————
# 1. Load & extract text from DOCX
# ——————————————————————————————————————————————
def extract_text_from_docx(file_path: str) -> str:
    doc = Document(file_path)
    return "\n".join(p.text for p in doc.paragraphs if p.text.strip())

doc_path = r"C:\Users\InfoBay\Desktop\Redis\Thesis (3).docx"
document_content = extract_text_from_docx(doc_path)

# ——————————————————————————————————————————————
# 2. Chunk the document semantically
# ——————————————————————————————————————————————
text_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=50)
chunks = text_splitter.split_text(document_content)

# ——————————————————————————————————————————————
# 3. Initialize embeddings & Redis vector store
# ——————————————————————————————————————————————
hf_embedder = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

vector_store = RedisVectorStore.from_texts(
    texts=chunks,
    embedding=hf_embedder,
    redis_url="redis://127.0.0.1:6379",
    index_name="doc_thesis_index",     
    text_key="page_content"         
)

# ——————————————————————————————————————————————
# 4. Initialize Groq & Redis-backed chat memory
# ——————————————————————————————————————————————
chat_groq = ChatGroq(
    model="deepseek-r1-distill-llama-70b",
    groq_api_key=""
)

redis_chat_history = RedisChatMessageHistory(
    url="redis://127.0.0.1:6379",
    session_id="user_session_1"
)

# ——————————————————————————————————————————————
# 5. Run a RAG-style query
# ——————————————————————————————————————————————
user_query = "What is the model that is presented in this study?"

# 5a. Similarity search in Redis
relevant_chunks = vector_store.similarity_search(user_query, k=3)
retrieved_content = "\n".join(c.page_content for c in relevant_chunks)

# 5b. Load previous chat history
chat_history = redis_chat_history.messages  # List[BaseMessage]

# 5c. Build prompt
prompt = [
    SystemMessage(content="You are a helpful assistant answering queries based on the provided document chunks.")
]
prompt += chat_history
prompt.append(HumanMessage(content=f"Relevant document content: {retrieved_content}\n\nUser Question: {user_query}"))

# 5d. Get the response
response = chat_groq(prompt).content

# 5e. Save interaction back to Redis
redis_chat_history.add_message(HumanMessage(content=user_query))
redis_chat_history.add_message(AIMessage(content=response))

# ——————————————————————————————————————————————
# 6. Output
# ——————————————————————————————————————————————
print(response)
