from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()

client = OpenAI(
    api_key=os.getenv("GEMINI_API_KEY"),
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

for model in [
    "gemini-3.6-flash",
    "gemini-3.5-flash",
    "gemini-3.1-flash-lite",
    "gemini-2.0-flash",
]:
    try:
        r = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "Hi"}],
        )
        print(model, "✅ works")
        print(r.choices[0].message.content)
        
    except Exception as e:
        print(model, "❌", e)