import os
from dotenv import load_dotenv

from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
load_dotenv()

pdf_path = Path(__file__).parent /"Aman_HR_Round.pdf"

loader = PyPDFLoader(
    file_path = pdf_path
)

docs = loader.load()
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=100,
    chunk_overlap=40,

)
split_docs = text_splitter.split_documents(docs)

embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
    
)



