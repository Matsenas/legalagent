import pandas as pd
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
# from pinecone import Pinecone
import os
import torch
from logger import setup_logger
from dotenv import load_dotenv

load_dotenv()
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
INDEX_NAME = os.environ.get("INDEX_NAME")
# pc = Pinecone(api_key=PINECONE_API_KEY)
logger = setup_logger("legal_agent", log_level=10)

# Cached embedding function (loaded once, reused across requests)
_cached_embed_fn = None 


# def check_cuda():
#     print("CUDA available:", torch.cuda.is_available())
#     if torch.cuda.is_available():
#         logger.info("GPU name:", torch.cuda.get_device_name(0))
#         logger.info("GPU count:", torch.cuda.device_count())


def data_preprocessing(data):
    try:
        list_of_documents = []
        for title, question, answer,  in zip(data['title'], data['question'], data['answer']):
            list_of_documents.append(Document(
                page_content=f"**Název:** {title}\n\n **Právní problém:** \n {question}\n\n **Právní rada:** {answer}",
                metadata={"title": title}
            ))
        logger.info(f"Documents proceessed successfully, size of documents: {len(list_of_documents)}")
        return list_of_documents
    except Exception as e:
        logger.error(f"Error in data preprocessing: {e}")
    


def create_embeddings():
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model_name = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        model_kwargs = {"device": device}
        embeddings = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs=model_kwargs
        )
        logger.info("Embeddings created successfully")
        return embeddings
    except Exception as e:
        logger.error(f"Error in creating embeddings: {e}")

def get_embed_fn():
    global _cached_embed_fn
    if _cached_embed_fn is not None:
        logger.info("Using cached embed function")
        return _cached_embed_fn

    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model_name = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        model_kwargs = {"device": device}
        _cached_embed_fn = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs=model_kwargs
        )
        logger.info("Embed function created successfully")
        return _cached_embed_fn
    except Exception as e:
        logger.error(f"Error in creating embed function: {e}")


def insertion_to_vectorstore(list_of_documents, embeddings, index_name):
    try:
        
        vectorstore = PineconeVectorStore.from_documents(
        documents=list_of_documents,
        embedding=embeddings,
        index_name=index_name,
        pinecone_api_key=PINECONE_API_KEY
        )
        logger.info("Data inserted to vectorstore successfully")
        return vectorstore
    except Exception as e:
        logger.error(f"Error in inserting to vectorstore: {e}")

def load_pinecone(index_name):
    
    try:
        docsearch = PineconeVectorStore.from_existing_index(
        embedding=get_embed_fn(),
        index_name=index_name
        )
        return docsearch
    except Exception as e:
        raise {"error": e}

# query = "Co se stane, pokud dojde k porušení autorských práv v České republice?"
# search=load_pinecone(INDEX_NAME)
# results = search.similarity_search(query, k=5)



# Populate Pinecone index
# data=pd.read_csv('legal_advice_CZE.csv')
# list_of_documents=data_preprocessing(data)
# embeddings=create_embeddings()
# insertion_to_vectorstore(list_of_documents, embeddings, INDEX_NAME)
