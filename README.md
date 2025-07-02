# 📄 PDF Question-Answering System using LangChain, FAISS & LLaMA

This project builds an intelligent question-answering system that processes a PDF, splits it into meaningful chunks, generates embeddings, stores them in a FAISS vector store, and uses a LLaMA model to answer user queries with contextually relevant information.

---

## 🔧 Features

- 📄 Loads and processes any PDF document
- ✂️ Splits text into chunks with overlap using LangChain's text splitter
- 🤗 Generates sentence embeddings using Hugging Face models
- 🧠 Stores and retrieves document vectors using FAISS
- 🦙 Uses Meta LLaMA model for contextual answer generation
- 🎯 Uses cosine similarity to refine results for more accurate answers

---

## 🚀 Installation

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/pdf-qa-system.git
cd pdf-qa-system

**2. Create and Activate a Virtual Environment**
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate

** 3.Install Dependencies **
pip install -r requirements.txt

**4.Log in to Hugging Face**
from huggingface_hub import login
login(token="your_huggingface_token")

🛠️ Usage

1.Update the pdf_path in the script with your PDF file path.
2.Replace the Hugging Face login token with your own.
3.Run the script:
  python your_script.py

🧠 How It Works

PDF Loading: Extracts text from PDF using PyPDFLoader.
Chunking: Text is split into overlapping chunks for context retention.
Embedding Generation: Each chunk is embedded using all-MiniLM-L6-v2.
Storage & Retrieval: Chunks and embeddings are stored in FAISS. Top-k chunks are retrieved.
Filtering: Cosine similarity is calculated manually and low-score chunks are filtered out.
Answer Generation: The final context is sent to a LLaMA model to generate a human-like answer.

**🧪 Example Output:**

Ask a question (or type 'exit' to quit): What is the main objective of the document?
Selected Context:
--------------------------------------------------
[Chunk 1 content...]
--------------------------------------------------
[Chunk 2 content...]

AI Answer: The main objective of the document is to...

**Dependencies**
pip install langchain langchain_huggingface transformers huggingface_hub faiss-cpu scikit-learn numpy pypdf



---

Let me know if you’d like a `requirements.txt` generated from your script as well!

