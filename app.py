import os
import streamlit as st
from langchain_groq import ChatGroq
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
import gspread 
import pandas as pd
from oauth2client.service_account import ServiceAccountCredentials
from google.oauth2.service_account import Credentials
import json

from dotenv import load_dotenv

load_dotenv()

SERVICE_ACCOUNT_FILE = "service_account_googlesheet.json"  # Ensure this file exists locally
scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
creds = ServiceAccountCredentials.from_json_keyfile_name(SERVICE_ACCOUNT_FILE, scope)
client = gspread.authorize(creds)

#load the GROQ and google api key from .env file
service_account_json = json.loads(os.getenv("GOOGLE_SERVICE_ACCOUNT_JSON"))
groq_api_key = os.getenv("GROQ_API_KEY")
os.environ['GOOGLE_API_KEY'] = os.getenv("GOOGLE_API_KEY")

st.title("PlaceMate Q&A")

def load_google_sheet_data():
    try:
        sheet = client.open("survey (Responses)").sheet1  # Change to your sheet name
        data = sheet.get_all_records()

        if not data:
            st.warning("Google Sheets is empty!")
            return pd.DataFrame()  # Return empty DataFrame to avoid breaking code
        return pd.DataFrame(data)  # Convert to Pandas DataFrame
    except Exception as e:
        st.error(f"Error loading Google Sheets: {e}")
        return pd.DataFrame()


llm = ChatGroq(groq_api_key=groq_api_key,model_name="gemma2-9b-it")


def vector_embedding():
    if "vectors" not in st.session_state:
        try:
            st.session_state.embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        except Exception as e:
            st.error(f"Failed to initialize embeddings: {e}")
            return
        
        data_df = load_google_sheet_data()  # Fetch data from Google Sheet
        excluded_columns = ["Timestamp","Name", "Student Enrollment Number"]  # Modify this based on actual column names
        documents = [
            ". ".join(f"{col}: {val}" for col, val in row.items() if col not in excluded_columns)
            for _, row in data_df.iterrows()
        ]
        
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        final_documents = text_splitter.create_documents(documents)

        if not final_documents:
            st.error("No data found in Google Sheets!")
            return

        st.session_state.vectors = FAISS.from_documents(final_documents, st.session_state.embeddings)

prompt1 = st.text_input("Enter your Question....")

vector_embedding()
st.write("Vector Store DB is ready")

if prompt1:
    document_chain = create_stuff_documents_chain(llm, ChatPromptTemplate.from_template("""
        Answer the questions based on the provided context if available.
        If no context is available, answer the question to the best of your knowledge.

        <context>
        {context}
        <context>
        Question: {input}
    """))

    import time

    if "vectors" not in st.session_state:
        st.error("Vector store is not initialized. Click 'Creating vector Store' first.")
        st.stop()

    retriever = st.session_state.vectors.as_retriever(search_kwargs={"k": 15})
    retrieval_chain= create_retrieval_chain(retriever,document_chain)
    
    start = time.process_time()
    retrieved_docs = retriever.invoke(prompt1)

    if retrieved_docs:  # If relevant docs are found, use them
        response = retrieval_chain.invoke({'input': prompt1})
        final_answer = response['answer']
    else:  # If no relevant docs, use LLM directly
        final_answer = llm.predict(prompt1)

    st.write(final_answer)


    #with a streamlit expander
    #with st.expander("Document similarity search"):
        #if retrieved_docs:
            #for i, doc in enumerate(response["context"]):
                #st.write(doc.page_content)
                #st.write("-------------------------")
        #else:
            #st.write("No relevant document found. Answering using general knowledge.")'''