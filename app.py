import os
from flask import Flask, request, jsonify, render_template
from PyPDF2 import PdfReader
from langchain_groq import ChatGroq
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from flask_cors import CORS
from dotenv import load_dotenv

load_dotenv()

# Load API keys
groq_api_key = os.getenv("GROQ_API_KEY")
os.environ['GOOGLE_API_KEY'] = os.getenv("GOOGLE_API_KEY")

# Initialize Flask app
app = Flask(__name__, static_folder="static", template_folder="templates")
CORS(app)

# PDF reader
def load_pdf_text(file_path):
    try:
        reader = PdfReader(file_path)
        text = ""
        for page in reader.pages:
            text += page.extract_text() or ""
        return text
    except Exception as e:
        return ""

# Vector store setup
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
llm = ChatGroq(groq_api_key=groq_api_key, model_name="gemma2-9b-it")

VECTOR_STORE_PATH = "vector_store"

if os.path.exists(VECTOR_STORE_PATH):
    vector_store = FAISS.load_local(VECTOR_STORE_PATH, embeddings, allow_dangerous_deserialization=True)
else:
    pdf_text = load_pdf_text("data.pdf")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=200)
    documents = text_splitter.create_documents([pdf_text])
    vector_store = FAISS.from_documents(documents, embeddings)
    vector_store.save_local(VECTOR_STORE_PATH)

retriever = vector_store.as_retriever(search_kwargs={"k": 15})
document_chain = create_stuff_documents_chain(
    llm,
    ChatPromptTemplate.from_template("""
    Answer the questions based on the provided context if available.
    If no context is available, answer the question to the best of your knowledge.

    <context>
    {context}
    <context>
    Question: {input}
    """)
)
retrieval_chain = create_retrieval_chain(retriever, document_chain)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/ask", methods=["POST"])
def ask():
    try:
        prompt = request.form.get("user_input") or request.json.get("query")
        retrieved_docs = retriever.invoke(prompt)
        if retrieved_docs:
            response = retrieval_chain.invoke({'input': prompt})
            return jsonify({"answer": response['answer']})
        else:
            answer = llm.invoke(prompt)
            return jsonify({"answer": response.content})
    except Exception as e:
        print("Error:", str(e))  # 🔍 Debug print
        return jsonify({"error": str(e)})

port = int(os.environ.get("PORT", 5000)) 

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=port)
