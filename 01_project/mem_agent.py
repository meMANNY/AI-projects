import os
from dotenv import load_dotenv
from mem0 import Memory

load_dotenv()

key = os.getenv("OPENAI_API")
# mem0's OpenAI embedder/LLM also read OPENAI_API_KEY from the environment
os.environ.setdefault("OPENAI_API_KEY", key or "")

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
    "graph_store": {
        "provider": "neo4j",
        "config": {
            "url": os.getenv("NEO4J_URI"),          # mem0 expects "url", not "uri"
            "username": os.getenv("NEO4J_USERNAME"),
            "password": os.getenv("NEO4J_PASSWORD"),
            "database": os.getenv("NEO4J_DATABASE")  # Aura home db (e.g. bdfe7660), not "neo4j"
        }
    },
    "vector_store": {
        "provider": "chroma",
        "config": {
            "collection_name": "mem0",
            "path": "./chroma_db"
        }
    }
}

m = Memory.from_config(config)
messages = [
    {"role": "user", "content": "Hi, I'm Aman. I love basketball and gaming."},
    {"role": "assistant", "content": "Hey Aman! I'll remember your interests."}
]
m.add(messages, user_id="aman")
results = m.search("What do you know about me?", user_id="aman")

import json
print(json.dumps(results, indent=2, ensure_ascii=False))