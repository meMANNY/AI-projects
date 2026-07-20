from google import genai
from google.genai import types
import os
from dotenv import load_dotenv
import requests
load_dotenv()

client = genai.Client(api_key=os.getenv("MY_API_KEY"))


def get_weather_info(location: str):
    """Return the current weather summary for a city or place."""
    url = f"https://wttr.in/{location}?format=3"
    response = requests.get(url, timeout=10)

    if response.status_code == 200:
        return response.text

    return "Could not retrieve weather information at this time."


def main():
    print("Weather agent ready. Type 'exit' to quit.")

    while True:
        user_input = input("Input: ").strip()
        if user_input.lower() in {"exit", "quit"}:
            break
        if not user_input:
            continue

        response = client.models.generate_content(
            model="gemini-3.5-flash",
            contents=user_input,
            config=types.GenerateContentConfig(
                tools=[get_weather_info],
            ),
        )
        print(f"Agent: {response.text}")


if __name__ == "__main__":
    main()