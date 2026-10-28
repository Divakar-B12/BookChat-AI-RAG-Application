from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

data = PyPDFLoader("document loaders/deep_learning.pdf")

docs = data.load()

splitter = RecursiveCharacterTextSplitter(
    chunk_size = 1000,
    chunk_overlap = 1
)

chunk = splitter.split_documents(docs)

print(len(chunk))


for i in chunk:
    print(i.page_content)
    print()
    print()
    print()