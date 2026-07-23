import os
import redis
from dotenv import load_dotenv
from rq import Queue
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from openai import OpenAI
from fastapi import FastAPI, Request, Query

load_dotenv()

r = redis.Redis.from_url(os.environ["REDIS_URL"])
queue = Queue(connection=r)
app = FastAPI()
# try:
#     r.ping()
#     print("✅ Redis connected")
# except redis.exceptions.ConnectionError as e:
#     print(f"❌ Redis connection failed: {e}")
#     raise
client = OpenAI(api_key=os.getenv("OPENAI_API"))
pdf_path = Path(__file__).parent /"Aman_HR_Round.pdf"

loader = PyPDFLoader(
    file_path = pdf_path
)

docs = loader.load()
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=200,
)
split_docs = text_splitter.split_documents(docs)

embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
    openai_api_key=os.getenv("OPENAI_API")
)

vectorstore = Chroma.from_documents(
    documents = split_docs,
    embedding=embeddings,
    collection_name="aman_hr_round",
)


def process_query(user_query):
    results = vectorstore.similarity_search(
        query=user_query,
        k=3
    )
    context = "\n\n".join(doc.page_content for doc in results)


    SYSTEM_PROMPT = f"""You are a helpful assistant that answers questions based on the context provided.
    If the answer is not contained within the context, you should respond with "I don't know."
    Context: {context}
    """

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_query}
        ]
    )
    print(f"Assistant: {response.choices[0].message.content}")

@app.get("/")
async def root():
    return {"message": "Hello, World!"}

@app.post("/chat")
async def chat(query: str = Query(..., description="The user's query")):
    job = queue.enqueue(process_query, query)
    return {"job_id": job.id, "status": "queued"}










