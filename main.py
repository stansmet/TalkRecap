from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()

# print(os.getenv("OPENAI_API_KEY"))
# exit()

client = OpenAI(
  api_key=os.getenv("OPENAI_API_KEY"),
)

completion = client.chat.completions.create(
  model="gpt-4o-mini",
  store=True,
  messages=[
    {"role": "user", "content": "write a haiku about ai"},
    # {"role": "system", "content": "Ты ассистент команды"},
  ]
)

print(completion.choices[0].message);