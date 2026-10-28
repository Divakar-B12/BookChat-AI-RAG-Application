from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

documents = [
    Document(
        page_content="Java is an object-oriented programming language."
    ),
    Document(
        page_content="RAG retrieves relevant documents before generating answers."
    ),
    Document(
        page_content="ChromaDB stores vector embeddings and supports similarity search."
    ),
    Document(
            page_content="Java is Java because that is java."
        )
]

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vector_store = Chroma.from_documents(
    documents=documents,
    embedding=embeddings,
    persist_directory="chroma_db"
)

retrive = vector_store.similarity_search("What is Java", k=2)


for r in retrive:
    print(r)

