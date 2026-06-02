from openai import OpenAI
import os
from dotenv import load_dotenv
load_dotenv()

endpoint = "https://abhik-ma8bxst0-eastus2.cognitiveservices.azure.com/openai/v1/"
model_name = "gpt-4o"
deployment_name = "gpt-4o"

api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(
    base_url=f"{endpoint}",
    api_key=api_key
)

completion = client.chat.completions.create(
    model=deployment_name,
    messages=[
        {
            "role": "user",
            "content": "What is the capital of France?",
        }
    ],
)

print(completion.choices[0].message)
print(completion.choices[0].message.content)