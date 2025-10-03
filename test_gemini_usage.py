"""Test script to check Gemini SDK usage_metadata attribute names."""

import os
from google import genai
from google.genai import types
from dotenv import load_dotenv

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    print("ERROR: GEMINI_API_KEY not found in environment")
    exit(1)

# Create client
client = genai.Client(api_key=GEMINI_API_KEY)

# Make a simple request
response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents="Say hello in 5 words"
)

print("Response object type:", type(response))
print("\nResponse attributes:")
for attr in dir(response):
    if not attr.startswith('_'):
        print(f"  - {attr}")

print("\n=== USAGE METADATA ===")
if hasattr(response, 'usage_metadata'):
    um = response.usage_metadata
    print(f"usage_metadata type: {type(um)}")
    print(f"usage_metadata attributes:")
    for attr in dir(um):
        if not attr.startswith('_'):
            value = getattr(um, attr, None)
            if not callable(value):
                print(f"  - {attr}: {value}")
else:
    print("No usage_metadata attribute found!")

print("\n=== RESPONSE TEXT ===")
print(response.text)

