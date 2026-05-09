import google.generativeai as genai
import os
from dotenv import load_dotenv

load_dotenv()

# Configure with your exact key
genai.configure(api_key=os.environ["GOOGLE_API_KEY"])

print("Your API Key has access to these Chat models:")
print("-" * 40)

# Sweep the API for allowed models
for m in genai.list_models():
    if 'generateContent' in m.supported_generation_methods:
        print(m.name)