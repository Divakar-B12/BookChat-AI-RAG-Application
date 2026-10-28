from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter

data = TextLoader("document loaders/my_notes.txt")

docs = data.load()

splitter = CharacterTextSplitter(
    chunk_size = 10,
    chunk_overlap = 1
)

chunks = splitter.split_documents(documents= docs)

for i in chunks:
    print(i.page_content)