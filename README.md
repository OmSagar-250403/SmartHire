# 🧠 Chat with Web Content using FAISS & Groq 🚀

This project builds a **chatbot** that can answer questions based on scraped **webpage content**.  
It uses **Word2Vec for embeddings** (no API needed) and **FAISS** for vector storage.  
The chatbot generates responses using **Groq LLM (Llama3-70B or Mixtral-8x7B).**  

---

## 🌟 Features
✅ **Scrape Web Content** – Extracts text from any webpage.  
✅ **Local Word2Vec Embeddings** – No API required for embedding generation.  
✅ **FAISS Vector Search** – Efficient retrieval of relevant content.  
✅ **Groq AI for Responses** – Uses Llama3-70B/Mixtral for generating intelligent answers.  
✅ **Save Content as PDF** – Download scraped text as a PDF.  

---

## 🛠️ Installation & Setup

### **1️⃣ Clone the Repository**
```sh
git clone https://github.com/OmSagar-250403/SmartHire.git


**2️⃣ Create a Virtual Environment (Optional, Recommended)**
python -m venv venv
source venv/bin/activate  # On macOS/Linux
venv\Scripts\activate      # On Windows


3️⃣ Install Dependencies
pip install -r requirements.txt


4️⃣ Set Up API Key
API=your_groq_api_key_here


🚀 Running the Application
streamlit run app.py

