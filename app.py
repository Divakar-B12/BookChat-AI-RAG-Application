
import os
import hashlib
import tempfile
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate


# Load environment variables
load_dotenv()


# Streamlit configuration
st.set_page_config(
    page_title="BookChat AI",
    page_icon="📚",
    layout="wide"
)


# Configuration
DB_DIRECTORY = "chroma_db"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


# Initialize session state
if "retriever" not in st.session_state:
    st.session_state.retriever = None

if "messages" not in st.session_state:
    st.session_state.messages = []

if "book_hash" not in st.session_state:
    st.session_state.book_hash = None

if "book_name" not in st.session_state:
    st.session_state.book_name = None


# Initialize embedding model
@st.cache_resource
def get_embedding_model():
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL
    )


# Initialize Groq LLM
@st.cache_resource
def get_llm():
    return ChatGroq(
        model="qwen/qwen3.8-27b",
        temperature=0
    )


# Prompt template from your original main.py
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            You are a helpful AI assistant.
            Use ONLY the provided context to answer the question.
            If the answer is not present in the context,
            say: "I could not find the answer in the document."
            """
        ),
        (
            "human",
            """
            Context: {context}
            Question: {question}
            """
        )
    ]
)


# Create or load the vector database
def create_vector_database(pdf_bytes, book_hash):
    db_path = Path(DB_DIRECTORY) / book_hash

    embeddings = get_embedding_model()

    # Reuse the index if this PDF was already processed
    if db_path.exists() and any(db_path.iterdir()):
        vectorstore = Chroma(
            persist_directory=str(db_path),
            embedding_function=embeddings
        )
        return vectorstore

    db_path.mkdir(parents=True, exist_ok=True)

    temp_path = None

    try:
        # Save uploaded PDF temporarily
        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".pdf"
        ) as temp_file:
            temp_file.write(pdf_bytes)
            temp_path = temp_file.name

        # Load PDF
        loader = PyPDFLoader(temp_path)
        documents = loader.load()

        if not documents:
            raise ValueError(
                "No readable pages found in the PDF."
            )

        # Split document into chunks
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )

        chunks = splitter.split_documents(documents)

        if not chunks:
            raise ValueError(
                "No text chunks were extracted from the PDF."
            )

        # Create embeddings and store in ChromaDB
        vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory=str(db_path)
        )

        return vectorstore

    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)


# Create retriever using your original MMR settings
def get_retriever(vectorstore):
    return vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": 4,
            "fetch_k": 10,
            "lambda_mult": 0.5
        }
    )


# RAG answer generation from your original main.py
def get_answer(retriever, query):
    docs = retriever.invoke(query)

    context = "\n\n".join(
        doc.page_content for doc in docs
    )

    if not context.strip():
        return (
            "I could not find the answer in the document.",
            docs
        )

    final_prompt = prompt.invoke(
        {
            "context": context,
            "question": query
        }
    )

    response = get_llm().invoke(final_prompt)

    return response.content, docs


# --------------------------------------------------
# STREAMLIT UI
# --------------------------------------------------

st.title("📚 BookChat AI")
st.write(
    "Upload a PDF book and ask questions about its content."
)


# Sidebar: Upload book
with st.sidebar:
    st.header("📖 Upload your book")

    uploaded_file = st.file_uploader(
        "Choose a PDF book",
        type=["pdf"]
    )

    if uploaded_file is not None:
        st.write(f"Selected file: {uploaded_file.name}")

        if st.button(
            "Process Book",
            type="primary",
            use_container_width=True
        ):
            try:
                pdf_bytes = uploaded_file.getvalue()

                book_hash = hashlib.sha256(
                    pdf_bytes
                ).hexdigest()

                with st.spinner(
                    "Reading book and creating vector database..."
                ):
                    vectorstore = create_vector_database(
                        pdf_bytes,
                        book_hash
                    )

                    st.session_state.retriever = get_retriever(
                        vectorstore
                    )

                    # Clear previous conversation for a new book
                    if st.session_state.book_hash != book_hash:
                        st.session_state.messages = []

                    st.session_state.book_hash = book_hash
                    st.session_state.book_name = uploaded_file.name

                st.success("Book processed successfully!")
                st.rerun()

            except Exception as e:
                st.error(f"Error processing book: {e}")

    if st.session_state.book_name:
        st.divider()
        st.subheader("Current book")
        st.write(st.session_state.book_name)

    if st.button("Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()


# Main chat area
if st.session_state.retriever is None:
    st.info(
        "Upload a PDF book from the sidebar and click "
        "'Process Book' to start chatting."
    )


# Display conversation history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        if message.get("sources"):
            with st.expander("Retrieved pages"):
                for source in message["sources"]:
                    st.write(source)


# Chat input
query = st.chat_input(
    "Ask a question about your book...",
    disabled=st.session_state.retriever is None
)


if query:
    st.session_state.messages.append(
        {
            "role": "user",
            "content": query
        }
    )

    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        try:
            with st.spinner(
                "Searching the book and generating an answer..."
            ):
                answer, docs = get_answer(
                    st.session_state.retriever,
                    query
                )

                # Extract PDF page references
                sources = []

                for doc in docs:
                    page = doc.metadata.get("page")

                    if isinstance(page, int):
                        sources.append(
                            f"Page {page + 1}"
                        )
                    else:
                        sources.append(
                            "Uploaded document"
                        )

                st.markdown(answer)

                if sources:
                    with st.expander("Retrieved pages"):
                        for source in dict.fromkeys(sources):
                            st.write(source)

            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": answer,
                    "sources": list(dict.fromkeys(sources))
                }
            )

        except Exception as e:
            st.error(f"Error generating answer: {e}")
