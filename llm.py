from groq import Groq
import os
from dotenv import load_dotenv

load_dotenv();

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if GROQ_API_KEY is None:
    raise ValueError("Bruh put in ur groq api key as an environment variable on the .env")

client = Groq(api_key = GROQ_API_KEY)
completion = client.chat.completions.create(
    model="llama-3.1-8b-instant",
    messages=[
        {
            "role": "user",
            "content": "Explain how LLM's work, simply"
        }
    ]
)
print(completion.choices[0].message.content)