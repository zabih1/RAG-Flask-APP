import os
from dotenv import load_dotenv
import weaviate
from weaviate.classes.init import Auth
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
from langchain_weaviate.vectorstores import WeaviateVectorStore

# Load environment variables
load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")
weaviate_api_key = os.getenv("WEAVIATE_API_KEY")
print(weaviate_api_key)
weaviate_url = os.getenv("WEAVIATE_URL")
print(weaviate_url)

llm = ChatGoogleGenerativeAI(
    api_key=api_key,
    model="gemini-2.0-flash"
)


#==================================================================================
# Processes a PDF, extracts text, generates embeddings, and stores them in Weaviate.
#==================================================================================

def setup_vector_database(pdf_path):
    """Processes a PDF, extracts text, generates embeddings, and stores them in Weaviate."""
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
    print("vector db is setup")
    return vector_db

#===================================================================================
# Builds a pipeline that leverages a Weaviate vector store to retrieve context and a language model to generate answers.
#==================================================================================

def create_rag_chain(vector_db):
    """Creates a RAG (Retrieval-Augmented Generation) chain using Weaviate as the retriever."""

    
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
