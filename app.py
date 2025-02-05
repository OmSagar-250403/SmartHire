import streamlit as st
import requests
from bs4 import BeautifulSoup
import re
import os
import numpy as np
from gensim.models import Word2Vec
from groq import Groq
from dotenv import load_dotenv
from fpdf import FPDF
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OpenAIEmbeddings

# Load environment variables
load_dotenv()
GROQ_API_KEY = os.getenv("API")

if not GROQ_API_KEY:
    st.error("Groq API key not found. Please add it in the .env file.")
    st.stop()

client = Groq(api_key=GROQ_API_KEY)

# Text Cleaning Function
def clean_text(text):
    return re.sub(r'[^\x00-\x7F]+', '', text)

# Scraping Function
def scrape_url(url):
    try:
        response = requests.get(url)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        page_content = soup.get_text(separator='\n', strip=True)
        return clean_text(page_content), url
    except requests.exceptions.RequestException as e:
        st.error(f"Error scraping the URL: {e}")
        return None, None

# Save content as PDF
def save_as_pdf(content, filename="scraped_content.pdf"):
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    
    for line in content.split('\n'):
        pdf.cell(200, 10, txt=line, ln=True)
    
    pdf.output(filename)
    return filename

# Word2Vec Model Training
def train_word2vec(sentences, vector_size=100, window=5, min_count=1):
    model = Word2Vec(sentences, vector_size=vector_size, window=window, min_count=min_count, workers=4)
    return model

# Convert text into Word2Vec embeddings
def get_word2vec_embeddings(model, text_chunks):
    embeddings = []
    for chunk in text_chunks:
        words = chunk.split()
        word_vectors = [model.wv[word] for word in words if word in model.wv]
        if word_vectors:
            embedding = np.mean(word_vectors, axis=0)
        else:
            embedding = np.zeros(model.vector_size)
        embeddings.append(embedding)
    return embeddings

# Splitting text into smaller chunks for Word2Vec
def get_text_chunks(raw_text, chunk_size=100):
    words = raw_text.split()
    return [' '.join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]

def get_vectorstore(text_chunks, embeddings):
    try:
        # FAISS expects (text, vector) pairs
        text_embedding_pairs = list(zip(text_chunks, embeddings))  
        
        # FAISS requires a placeholder embedding function (even though we're using Word2Vec)
        vectorstore = FAISS.from_texts(
            [text for text, _ in text_embedding_pairs],  # Extract texts
            embeddings=np.array([vec for _, vec in text_embedding_pairs]),  # Extract embeddings
            embedding=OpenAIEmbeddings()  # Placeholder, FAISS requires an embedding function
        )
        
        vectorstore.save_local("faiss_index")
        st.success("FAISS index created and saved successfully!")
        return vectorstore
    except Exception as e:
        st.error(f"Error creating FAISS vectorstore: {e}")
        return None

# Query Groq API
def query_groq(question):
    try:
        response = client.chat.completions.create(
            model="llama3-70b",  # or "mixtral-8x7b"
            messages=[{"role": "user", "content": question}],
            temperature=0.6,
            max_tokens=4096,
            top_p=0.95,
        )
        return response.choices[0].message.content
    except Exception as e:
        st.error(f"Error querying Groq API: {e}")
        return "Sorry, an error occurred."

# Handle user query and retrieve relevant content
def handle_userinput(question, vectorstore):
    search_results = vectorstore.similarity_search(question, k=1)
    if search_results:
        context = search_results[0].page_content
        st.write(f"Using extracted content: {context}")
        answer = query_groq(context + "\n\n" + question)
        st.session_state.chat_history.append({"user": question, "bot": answer})
        st.write(answer)
    else:
        st.error("No relevant content found in the vector database.")

# Streamlit UI
def main():
    st.set_page_config(page_title="Chat with Web Content", page_icon=":globe_with_meridians:")
    st.header("Chat with Web Content :globe_with_meridians:")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    url = st.text_input("Enter the URL to scrape:")
    
    if st.button("Process"):
        if url:
            scraped_content, source_url = scrape_url(url)
            if scraped_content:
                text_chunks = get_text_chunks(scraped_content)
                sentences = [chunk.split() for chunk in text_chunks]  # Tokenizing for Word2Vec

                # Train Word2Vec Model
                word2vec_model = train_word2vec(sentences)

                # Generate Embeddings
                embeddings = get_word2vec_embeddings(word2vec_model, text_chunks)

                # Store in FAISS
                vectorstore = get_vectorstore(text_chunks, embeddings)

                # Save PDF
                pdf_filename = save_as_pdf(scraped_content)
                st.session_state["pdf_filename"] = pdf_filename
                st.session_state.source_url = source_url
                st.session_state.vectorstore = vectorstore
                st.success("Web content processed and stored successfully!")

            else:
                st.error("Failed to scrape content from the URL.")
        else:
            st.error("Please enter a valid URL.")

    # Provide option to download scraped content as PDF
    if "pdf_filename" in st.session_state:
        with open(st.session_state["pdf_filename"], "rb") as pdf_file:
            st.download_button(
                label="Download Scraped Content as PDF",
                data=pdf_file,
                file_name="scraped_content.pdf",
                mime="application/pdf"
            )

    # Chat Functionality
    user_question = st.text_input("Ask a question about the web content:")
    
    if user_question and "vectorstore" in st.session_state:
        handle_userinput(user_question, st.session_state.vectorstore)
        if "source_url" in st.session_state:
            st.write(f"Source: [Link]({st.session_state.source_url})")

if __name__ == '__main__':
    main()
