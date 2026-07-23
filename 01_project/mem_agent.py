import os
from dotenv import load_dotenv
from mem0 import Memory
import openai
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

load_dotenv()

key = os.getenv("OPENAI_API")

vector_store = Chroma(
    persist_directory="./chroma_db",
    collection_name="mem0",
    embedding_function=OpenAIEmbeddings(model="text-embedding-3-small", api_key=key),
)
config = {
    "embedder":{
        "provider": "openai",
        "config": {
            "model": "text-embedding-3-small",
            "api_key": key
        }
    },
    "llm":{
        "provider": "openai",
        "config": {
            "model": "gpt-4.1-mini",
            "api_key": key
        }

    },
    "vector_store": {
        "provider": "langchain",
        "config": {
            "client": vector_store
        }
    }
}

m = Memory.from_config(config)
messages = [
    {"role": "user", "content": "Hi, I'm Alex. I love basketball and gaming."},
    {"role": "assistant", "content": "Hey Alex! I'll remember your interests."}
]
m.add(messages, user_id="alex")
results = m.search("What do you know about me?", filters={"user_id": "alex"})

import json
print(json.dumps(results, indent=2, ensure_ascii=False))