from google import genai
import os
from dotenv import load_dotenv
load_dotenv()
client = genai.Client(api_key=os.getenv("MY_API_KEY"))

history = [{
    "type": "user_input",
    "content": [
        {
            "type": "text",
            "text":"I have 3 dogs in my house."
        }
    ]
}]

chat1 = client.interactions.create(
    model = "gemini-3.5-flash",
    store = False,
    input = history
)

print ("Response 1:" , chat1.steps[-1].content[0].text)

for step in chat1.steps:
    history.append(step.model_dump())

history.append({
    "type": "user_input",
    "content": [{
        "type": "text",
        "text":"How many paws in the house?"
    }]
})

chat2 = client.interactions.create(
    model = "gemini-3.5-flash",
    store = False,
    input = history
)

print ("Response 2:" , chat2.steps[-1].content[0].text)