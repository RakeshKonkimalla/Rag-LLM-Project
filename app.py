
import os
import io
import hashlib
import json
import time
import pickle
from typing import List, Dict, Tuple
from dotenv import load_dotenv
import streamlit as st
from PyPDF2 import PdfReader
from fastembed import TextEmbedding
import numpy as np
import faiss
import requests

load_dotenv()

# ----------------------------
# Configuration
# ----------------------------
SUBJECTS = ["C","C++","C#","Python","Java","Java Script","Ruby", "Frontend-Web development","Backend-Web development", "Mathematics", "Physics", "Chemistry"]
STORE_DIR = "store"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2" # more relaiable and lightweight
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
BATCH_EMBED = 256
TOP_K = 5

# Grok API config
GROK_API_KEY = os.getenv("GROK_AI_API_KEY")
GROK_API_URL = "https://api.groq.com/openai/v1"
GROK_MODEL = "openai/gpt-oss-20b"

os.makedirs(STORE_DIR, exist_ok=True)

# ----------------------------
# Utils
# ----------------------------
def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()

def extract_text_from_pdf_bytes(b: bytes) -> str:
    text_parts = []
    reader = PdfReader(io.BytesIO(b))
    for page in reader.pages:
        try:
            t = page.extract_text()
            if t:
                text_parts.append(t)
        except Exception:
            pass
    return "\n".join(text_parts)

def extract_text_from_txt_bytes(b: bytes) -> str:
    try:
        return b.decode("utf-8")
    except UnicodeDecodeError:
        return b.decode("latin-1")

def chunk_text(text: str, size: int, overlap: int) -> List[str]:
    chunks = []
    start = 0
    while start < len(text):
        end = start + size
        chunk = text[start:end]
        if len(chunk.strip()) > 20:
            chunks.append(chunk.strip())
        start = end - overlap
    return chunks

# ----------------------------
# Embedding model
# ----------------------------
@st.cache_resource(show_spinner=False)
def load_embedding_model():
    return TextEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")

embed_model = load_embedding_model()

def get_embeddings(texts: List[str]) -> np.ndarray:
    return np.array(list(embed_model.embed(texts)))

# ----------------------------
# FAISS + metadata management
# ----------------------------
def subject_dir(subject: str, uid: str) -> str:
    d = os.path.join(STORE_DIR, subject, uid)
    os.makedirs(d, exist_ok=True)
    return d

def meta_path(subject: str, uid: str) -> str:
    return os.path.join(subject_dir(subject, uid), "metadata.pkl")

def index_path(subject: str, uid: str) -> str:
    return os.path.join(subject_dir(subject, uid), "index.faiss") #vectordb to store vectors

def file_hashes_path(subject: str, uid: str) -> str:
    return os.path.join(subject_dir(subject, uid), "file_hashes.json")

def load_metadata(subject: str, uid: str) -> List[Dict]:
    p = meta_path(subject, uid)
    if os.path.exists(p):
        with open(p, "rb") as f:
            return pickle.load(f)
    return []

def save_metadata(subject: str, uid: str, md: List[Dict]):
    with open(meta_path(subject, uid), "wb") as f:
        pickle.dump(md, f)

def load_file_hashes(subject: str, uid: str) -> Dict:
    p = file_hashes_path(subject, uid)
    if os.path.exists(p):
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

def save_file_hashes(subject: str, uid: str, d: Dict):
    with open(file_hashes_path(subject, uid), "w", encoding="utf-8") as f:
        json.dump(d, f, indent=2)

def create_empty_faiss_index(dimension: int):
    return faiss.IndexFlatL2(dimension)

def load_or_create_index(subject: str, uid: str, dim: int):
    ip = index_path(subject, uid)
    if os.path.exists(ip):
        try:
            return faiss.read_index(ip)
        except Exception:
            return create_empty_faiss_index(dim)
    return create_empty_faiss_index(dim)

def save_index(subject: str, uid: str, index):
    faiss.write_index(index, index_path(subject, uid))

# ----------------------------
# Ingestion
# ----------------------------
def ingest_file(subject: str, uid: str, uploaded_file, update_progress=None) -> Tuple[bool, str]:
    try:
        raw = uploaded_file.read()
        file_hash = sha256_bytes(raw)
        file_name = uploaded_file.name

        fhashes = load_file_hashes(subject, uid)
        # Unique key now includes UID
        uid_key = f"{uid}_{file_hash}"
        if uid_key in fhashes:
            return False, f"This file already ingested for UID {uid} under subject '{subject}'."

        if file_name.lower().endswith(".pdf"):
            text = extract_text_from_pdf_bytes(raw)
        else:
            text = extract_text_from_txt_bytes(raw)

        chunks = chunk_text(text, CHUNK_SIZE, CHUNK_OVERLAP)
        if not chunks:
            return False, "No extractable text found."

        model_dim = 384
        index = load_or_create_index(subject, uid, model_dim)
        metadata = load_metadata(subject, uid)

        if update_progress:
            update_progress(0.1)

        # Embed all chunks in one call for speed
        emb = get_embeddings(chunks)

        if update_progress:
            update_progress(0.7)

        index.add(emb.astype("float32"))
        for chunk in chunks:
            metadata.append({
                "chunk_id": len(metadata),
                "file_name": file_name,
                "file_hash": file_hash,
                "uid": uid,
                "subject": subject,
                "text": chunk,
            })

        if update_progress:
            update_progress(0.9)

        save_index(subject, uid, index)
        save_metadata(subject, uid, metadata)
        fhashes[uid_key] = {"file_name": file_name, "uid": uid, "time": time.time()}
        save_file_hashes(subject, uid, fhashes)

        if update_progress:
            update_progress(1.0)

        return True, f"Ingested {len(chunks)} chunks for UID {uid} into subject '{subject}'."
    except Exception as e:
        return False, f"Error during Ingestion: {str(e)}"

# ----------------------------
# Retrieval (UID filtered)
# ----------------------------
def retrieve_top_k(subject: str, uid: str, query: str, k=TOP_K) -> List[Dict]:
    metadata = load_metadata(subject, uid)
    if not metadata:
        return []

    model_dim = 384
    index = load_or_create_index(subject, uid, model_dim)
    if index.ntotal == 0:
        return []

    q_emb = get_embeddings([query]).astype("float32")
    D, I = index.search(q_emb, min(k, index.ntotal))  
    results = []
    for idx in I[0]:
        if idx < len(metadata):
            results.append(metadata[idx])
    return results

# ----------------------------
# Grok API
# ----------------------------
def call_grok_api(question: str, context_chunks: List[str]) -> Tuple[bool, str]:
    if not GROK_API_KEY:
        return False, "Missing GROK_API_KEY."

    url = GROK_API_URL.rstrip("/") + "/chat/completions"
    headers = {"Authorization": f"Bearer {GROK_API_KEY}", "Content-Type": "application/json"}
    ctx = "\n\n---\n\n".join(context_chunks)
    if len(ctx) > 20000:
        ctx = ctx[-20000:]
    messages = [
        {"role": "system", "content": f"Use this context to answer the user's question. If the context is not relevant to the question, mention that content is not sufficient, so generating generic answer for this case:\n\n{ctx}"},
        {"role": "user", "content": question},
    ]
    payload = {"model": GROK_MODEL, "messages": messages, "max_tokens": 800, "temperature": 0}

    try:
        resp = requests.post(url, headers=headers, json=payload, timeout=60)
        resp.raise_for_status()
        data = resp.json()
        if "choices" in data:
            return True, data["choices"][0]["message"]["content"]
        return False, "Unexpected response."
    except Exception as e:
        return False, str(e)

# ----------------------------
# Streamlit UI
# ----------------------------
st.set_page_config(page_title="RAG Learning Assistant", layout="wide")

st.markdown(
    """
    <style>
    .stApp {background-color:#0f1115;color:#e6eef8;}
    .panel {background:rgba(255,255,255,0.02);padding:20px;border-radius:12px;box-shadow:0 4px 10px rgba(0,0,0,0.5);}
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("RAG Learning Assistant — UID Based")

if "success_msg" in st.session_state:
    st.success(st.session_state.success_msg)
    del st.session_state.success_msg

col1, col2 = st.columns([1, 2])

# ------------------ Ingestion Panel ------------------
with col1:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.header("Ingestion")
    subject = st.selectbox("Select subject", [""] + SUBJECTS)
    uid = st.text_input("Enter your UID (required)", "")
    file = st.file_uploader("Upload PDF or TXT", type=["pdf", "txt"])
    if st.button("Ingest file"):
        if not subject or not uid.strip() or not file:
            st.error("Please fill all fields (Subject, UID, File).")
        else:
            progress = st.progress(0.0)
            status = st.empty()
            def upd(p):
                progress.progress(p)
                status.text(f"Ingestion {int(p*100)}%")
            with st.spinner("Processing file..."):
                ok, msg = ingest_file(subject, uid.strip(), file, upd)
            if ok:
                st.session_state.success_msg = msg
                st.rerun()
            else:
                st.warning(msg)
    st.markdown('</div>', unsafe_allow_html=True)


# ------------------ Interaction Panel ------------------
with col2:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.header("Interaction")
    chat_subject = st.selectbox("Select subject for chat", [""] + SUBJECTS, key="chat_subj")
    chat_uid = st.text_input("Enter your UID (required)", "", key="chat_uid")

    enable_chat = False
    if chat_subject and chat_uid.strip():
        user_entries = load_metadata(chat_subject, chat_uid.strip())
        if not user_entries:
            st.error(f"No data found for UID {chat_uid} under {chat_subject}. Please ingest first.")
        else:
            enable_chat = True

    if enable_chat:
        question = st.text_area("Ask your question")
        if st.button("Ask"):
            with st.spinner("Retrieving top 5 chunks..."):
                chunks = retrieve_top_k(chat_subject, chat_uid.strip(), question, TOP_K)
                if not chunks:
                    st.warning("No matching chunks found.")
                else:
                    # Display the top 5 chunks on screen
                    st.markdown("**Top 5 retrieved chunks:**")
                    for i, chunk in enumerate(chunks):
                        st.markdown(f"**Chunk {i+1} — file: {chunk.get('file_name', '-')}, UID: {chunk.get('uid','-')}**")
                        st.write(chunk.get("text", ""))
                        st.write("---")

                    # Call Grok API
                    ctxs = [c["text"] for c in chunks]
                    ok, ans = call_grok_api(question, ctxs)
                    if ok:
                        st.subheader("Answer")
                        st.write(ans)
                    else:
                        st.error(ans)
    st.markdown('</div>', unsafe_allow_html=True)
