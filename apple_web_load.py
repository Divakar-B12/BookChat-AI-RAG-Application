from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import WebBaseLoader

load_dotenv()

url = "https://www.apple.com/in/macbook-pro/"

data = WebBaseLoader(url)
docs = data.load()

prompt = ChatPromptTemplate.from_messages(
    [("system", "What is the price os MacBook Pro"),
     ("human", "{data}")]
)

model = ChatGroq(model = "openai/gpt-oss-120b")

final_prompt = prompt.format_messages(data = docs[0].page_content)

response = model.invoke(final_prompt)


print(response.content)
