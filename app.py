import os
from dotenv import load_dotenv
load_dotenv()

from flask import Flask, render_template, request, jsonify
import werkzeug

from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
import weaviate
from weaviate.classes.init import Auth
from langchain_weaviate.vectorstores import WeaviateVectorStore

api_key = os.getenv("GEMINI_API_KEY")
weaviate_api_key = os.getenv("WEAVIATE_API_KEY")
weaviate_url = os.getenv("WEAVIATE_URL")

llm = ChatGoogleGenerativeAI(
    api_key=api_key,
    model="gemini-2.0-flash"
)

current_vector_db = None
current_rag_chain = None

def setup_vector_database(pdf_path):
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = text_splitter.split_documents(documents)

    embeddings = GoogleGenerativeAIEmbeddings(
        google_api_key=api_key, 
        model="models/embedding-001"
    )

    client = weaviate.connect_to_weaviate_cloud(
        cluster_url=weaviate_url,
        auth_credentials=Auth.api_key(weaviate_api_key),
        skip_init_checks=True
    )

    vector_db = WeaviateVectorStore.from_documents(docs, embeddings, client=client)
    return vector_db

def create_rag_chain(vector_db):
    template = """You are an assistant for question-answering tasks. 
    Use the following pieces of retrieved context to answer the question. 
    If you don't know the answer, just say that you don't know. 
    
    Question: {question} 
    Context: {context} 
    
    Answer:"""
    
    prompt = ChatPromptTemplate.from_template(template)
    
    retriever = vector_db.as_retriever(search_kwargs={"k": 5})
    
    rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    return rag_chain

app = Flask(__name__)
app.config['DATA'] = 'DATA'
os.makedirs(app.config['DATA'], exist_ok=True)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload_pdf', methods=['POST'])
def upload_pdf():
    global current_vector_db, current_rag_chain
    
    if 'pdf-file' not in request.files:
        return "No file uploaded", 400
    
    file = request.files['pdf-file']
    
    if file.filename == '':
        return "No selected file", 400
    
    filename = werkzeug.utils.secure_filename(file.filename)
    filepath = os.path.join(app.config['DATA'], filename)
    file.save(filepath)
    
    current_vector_db = setup_vector_database(filepath)
    current_rag_chain = create_rag_chain(current_vector_db)
    
    return jsonify({'message': 'PDF uploaded successfully'}), 200

@app.route('/ask', methods=['POST'])
def ask():
    global current_rag_chain
    
    if current_rag_chain is None:
        return jsonify({'answer': "Please upload a PDF first before asking questions."}), 400
    
    message = request.form.get('messageText', '')
    response = current_rag_chain.invoke(message)
    
    return jsonify({'answer': response})

if __name__ == '__main__':
    app.run(debug=True)
