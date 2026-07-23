
import os

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API"))

response = client.chat.completions.create(
    model="gpt-4.1-mini",
    messages=[
        {"role": "user", "content": [
            {"type": "text", "text": "Generate a caption for this image."},
            {"type": "image_url", "image_url": {"url": "https://images.pexels.com/photos/33529747/pexels-photo-33529747.jpeg"}}
        ]},
    ]
)

print(response)
        
        