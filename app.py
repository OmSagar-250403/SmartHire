import streamlit as st
import requests
from bs4 import BeautifulSoup
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import re

# Function to clean text
def clean_text(text):
    # Remove non-ASCII characters
    return re.sub(r'[^\x00-\x7F]+', '', text)

# Function to scrape content from a URL
def scrape_url(url):
    try:
        response = requests.get(url)
        response.raise_for_status()  # Raise an exception for HTTP errors
        soup = BeautifulSoup(response.text, 'html.parser')
        page_content = soup.get_text(separator='\n', strip=True)
        return clean_text(page_content), url  # Return cleaned content and source URL
    except requests.exceptions.RequestException as e:
        st.error(f"Error scraping the URL: {e}")
        return None, None

# Function to split text into chunks
def get_text_chunks(raw_text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100
    )
    text_chunks = text_splitter.split_text(raw_text)
    return text_chunks

# Function to create embeddings and vector store
def get_vectorstore(text_chunks):
    try:
        embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
        vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
        vectorstore.save_local("faiss_index")  # Save the FAISS index locally
        st.success("FAISS index created and saved successfully!")
        return vectorstore
    except Exception as e:
        st.error(f"Error creating vectorstore: {e}")
        return None

# Function to handle user input
def handle_userinput(question):
    response = st.session_state.conversation({"question": question})
    st.session_state.chat_history.append(response['chat_history'])
    st.write(response['answer'])

# Function to create conversation chain
def get_conversation_chain(vectorstore):
    llm = ChatGoogleGenerativeAI(model='gemini-1.5-pro-latest')
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain

# Main function to run the app
def main():
    st.set_page_config(page_title="Chat with Web Content", page_icon=":globe_with_meridians:")
    st.header("Chat with Web Content :globe_with_meridians:")

    # Input field for URL
    url = st.text_input("Enter the URL to scrape:")

    if st.button("Process"):
        if url:
            # Scrape content from the URL
            scraped_content, source_url = scrape_url(url)
            if scraped_content:
                # Split the scraped content into chunks
                text_chunks = get_text_chunks(scraped_content)
                # Create vector store
                vectorstore = get_vectorstore(text_chunks)
                if vectorstore:
                    # Initialize conversation chain
                    st.session_state.conversation = get_conversation_chain(vectorstore)
                    st.session_state.chat_history = []  # Initialize chat history
                    st.session_state.source_url = source_url  # Store source URL for reference
                    st.success("Web content processed successfully!")
                else:
                    st.error("There was an error creating the vectorstore.")
            else:
                st.error("Failed to scrape content from the URL.")
        else:
            st.error("Please enter a valid URL.")

    # Chat interface
    user_question = st.text_input("Ask a question about the web content:")
    if user_question:
        if "conversation" in st.session_state:
            handle_userinput(user_question)
            # Optionally include the source URL in the response
            if "source_url" in st.session_state:
                st.write(f"Source: [Link]({st.session_state.source_url})")
        else:
            st.error("Please process a URL first.")

if __name__ == '__main__':
    main()