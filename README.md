# 📚 BookChat AI — RAG Application

BookChat AI is a Retrieval-Augmented Generation (RAG) application that allows users to upload PDF books and ask questions about their content. It retrieves relevant text from the uploaded document and uses a Large Language Model (LLM) to generate context-based answers.

## ✨ Features

- **PDF Upload:** Upload your own books in PDF format.
- **Document Processing:** Extract text from PDFs and split it into smaller chunks.
- **Text Embeddings:** Generate embeddings using Hugging Face Sentence Transformers.
- **Vector Database:** Store and retrieve document chunks using ChromaDB.
- **Semantic Retrieval:** Use Maximum Marginal Relevance (MMR) to retrieve relevant context.
- **AI-Powered Answers:** Generate responses using the Groq API.
- **Interactive Chat Interface:** Ask questions through a Streamlit web application.
- **Source References:** Display the PDF pages associated with retrieved content.
- **Multiple Books:** Maintain separate vector indexes for different uploaded PDFs.

## 🛠️ Tech Stack

- **Language:** Python
- **Frontend:** Streamlit
- **LLM Framework:** LangChain
- **LLM Provider:** Groq
- **Embedding Model:** `sentence-transformers/all-MiniLM-L6-v2`
- **Vector Database:** ChromaDB
- **PDF Processing:** PyPDF
- **Environment Management:** python-dotenv

## 🏗️ Application Workflow

1. The user uploads a PDF book.
2. The application extracts text from the PDF.
3. The extracted text is divided into overlapping chunks.
4. Hugging Face embeddings convert the chunks into vector representations.
5. ChromaDB stores the vectors and document text.
6. When a user asks a question, the retriever searches for relevant chunks using MMR.
7. The retrieved context and question are passed to the Groq LLM.
8. The generated answer is displayed in the Streamlit chat interface.

## 📂 Project Structure

```text
RAG-Project/
├── app.py
├── requirements.txt
├── .env
├── .gitignore
├── chroma_db/
├── document loaders/
└── vector store/
```

**Note:** `.env`, `.venv`, and `chroma_db/` are excluded from Git using `.gitignore`. The database directory is generated when documents are processed.

## ⚙️ Installation and Setup

### 1. Clone the repository

```bash
https://github.com/Divakar-B12/BookChat-AI-RAG-Application.git
cd <YOUR_PROJECT_DIRECTORY>
```

Replace the placeholders with your actual GitHub repository URL and project directory name.

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate it on Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment variables

Create a `.env` file in the project root:

```dotenv
GROQ_API_KEY=your_groq_api_key
```

Obtain your API key from the [Groq Console](https://console.groq.com/).

Never commit your actual API key to GitHub.

### 5. Run the application

```bash
streamlit run app.py
```

Open the local URL displayed in your terminal, usually `http://localhost:8501`.

## 🚀 How to Use

1. Start the application.
2. Upload a PDF book using the sidebar.
3. Click **Process Book** and wait for indexing to complete.
4. Enter a question in the chat input.
5. Review the answer generated from the retrieved document content.
6. Expand **Retrieved pages** to inspect the source page references.
7. Upload another PDF to ask questions about a different book.

## 🔍 Retrieval Configuration

The application uses Maximum Marginal Relevance (MMR) retrieval with the following configuration:

| Parameter | Value |
|---|---:|
| `k` | 4 |
| `fetch_k` | 10 |
| `lambda_mult` | 0.5 |
| `chunk_size` | 1000 |
| `chunk_overlap` | 200 |

These settings balance relevance and diversity in the retrieved document chunks.

## 🔐 Security and Limitations

- Keep API keys in environment variables and never expose them in public repositories.
- Uploaded PDF content and vector indexes require appropriate storage management when deployed.
- The application extracts embedded PDF text; scanned PDFs may require OCR.
- Answers depend on the quality of the extracted text and retrieved context.
- Groq API usage is subject to the provider's model availability, rate limits, and applicable usage policies.
- Local ChromaDB storage should not be assumed to persist across hosting sessions or deployments unless persistent storage is configured.

## 🌐 Deployment

The Streamlit application can be deployed using [Streamlit Community Cloud](https://share.streamlit.io/).

1. Push the project to GitHub.
2. Create a new app in Streamlit Community Cloud.
3. Select the repository and the `app.py` entry point.
4. Add `GROQ_API_KEY` under the app's Secrets settings.
5. Deploy and test PDF upload and question answering.

Ensure that `requirements.txt` contains all required packages before deployment.

## 📌 Future Improvements

- Support additional document formats.
- Add conversation export functionality.
- Improve document source citations.
- Add OCR support for scanned books.
- Implement persistent cloud storage for uploaded documents.
- Add user authentication and document management.

## 👨‍💻 Author

**Divakar B**

Built as a hands-on project to explore Retrieval-Augmented Generation, vector databases, semantic search, and Large Language Model integration using Python.

---

If you find this project useful, consider giving the repository a ⭐ on GitHub.
