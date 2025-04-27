# app.py
import streamlit as st
from ingest import load_and_split_pdf
from retriever import create_vector_store, get_relevant_docs
import requests
import os
from dotenv import load_dotenv
load_dotenv()
# Load environment variables
HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY")

st.title("🧠 Medical Chatbot (COVID-19 Guidelines)")

# File uploader for custom PDF
uploaded_file = st.file_uploader("Upload a PDF file for reference:", type="pdf")

if uploaded_file:
    # Save the uploaded file temporarily
    with open("uploaded_file.pdf", "wb") as f:
        f.write(uploaded_file.read())
    
    # Process the uploaded PDF
    docs = load_and_split_pdf("uploaded_file.pdf")
    db = create_vector_store(docs)

    query = st.text_input("Ask a question about the uploaded document:")
    if query:
        context_docs = get_relevant_docs(db, query)
        context = "\n\n".join([doc.page_content for doc in context_docs])

        prompt = f"Answer the following question using ONLY the context below.\n\nContext:\n{context}\n\nQuestion: {query}"

        # Hugging Face API request
        headers = {"Authorization": f"Bearer {HUGGINGFACE_API_KEY}"}
        payload = {
            "inputs": prompt,
            "parameters": {"max_length": 500, "temperature": 0.7},
        }
        response = requests.post(
            "https://api-inference.huggingface.co/models/EleutherAI/gpt-neo-1.3B", 
            headers=headers, 
            json=payload
        )

        if response.status_code == 200:
            try:
                result = response.json()
                st.write(result[0]["generated_text"])
            except (KeyError, IndexError):
                st.write("Error: Unexpected response format:", response.text)
            except requests.exceptions.JSONDecodeError:
                st.write("Error: Unable to decode JSON response:", response.text)
        else:
            st.write(f"Error {response.status_code}: {response.text}")

