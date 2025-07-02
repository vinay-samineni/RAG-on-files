from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from huggingface_hub import login
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from transformers import pipeline

# 1️⃣ Login to Hugging Face (Ensure your token is valid)
login(token="Enter your token here")

# 2️⃣ Load the PDF
pdf_path = ""  # Update with your file path
loader = PyPDFLoader(pdf_path)
documents = loader.load()

# 3️⃣ Split the text into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)  # Increased size
chunks = text_splitter.split_documents(documents)

# 4️⃣ Load Hugging Face embedding model
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# ✅ Test: Generate an embedding for the first chunk
first_chunk_embedding = embeddings.embed_query(chunks[0].page_content)
print("First Chunk Embedding (First 5 values):", first_chunk_embedding[:5])

# 5️⃣ Convert chunks into embeddings and store in FAISS
vector_store = FAISS.from_documents(chunks, embeddings)

# 6️⃣ Load the Llama model for text generation
pipe = pipeline("text-generation", model="meta-llama/Llama-3.2-1B")

# 7️⃣ Define similarity threshold
SIMILARITY_THRESHOLD = 0.7  # Adjust this value based on performance

# 8️⃣ Start user interaction loop
while True:
    # Get user input
    query = input("\nAsk a question (or type 'exit' to quit): ")
    if query.lower() == 'exit':
        break

    # Convert query into an embedding
    query_embedding = embeddings.embed_query(query)

    # Retrieve top-k chunks (fetch more to filter later)
    retrieved_chunks = vector_store.similarity_search(query, k=5)  
    for i, chunks in enumerate(retrieved_chunks):
        print(i)
        print(50*'-')
        print(chunks) 

    # Compute cosine similarity for each chunk manually
    filtered_chunks = []
    for chunk in retrieved_chunks:
        chunk_embedding = embeddings.embed_query(chunk.page_content)
        similarity = cosine_similarity([query_embedding], [chunk_embedding])[0][0]  # Compute similarity

        if similarity >= SIMILARITY_THRESHOLD:  # Apply threshold
            filtered_chunks.append((chunk, similarity))

    # Sort by similarity (highest first)
    filtered_chunks.sort(key=lambda x: x[1], reverse=True)

    # Take the top 3 chunks after filtering
    final_chunks = [chunk[0] for chunk in filtered_chunks[:3]]

    # Print selected chunks
    print("\nSelected Context:")
    for i, chunk in enumerate(final_chunks):
        print(i)
        print(50*'-')
        print(chunk)

    # Prepare context for Llama model
    context = "\n".join([chunk.page_content for chunk in final_chunks])
    prompt = f"Use the following context to answer the question:\n\n{context}\n\nQuestion: {query}\nAnswer:"

    # Generate answer using Llama
    response = pipe(prompt, max_new_tokens=100, num_return_sequences=1)
    
    # Print the AI-generated answer
    print("\nAI Answer:", response[0]['generated_text'])
