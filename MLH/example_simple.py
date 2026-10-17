"""
Simple standalone example script based on Google GenAI snippet.
"""

import sys
from dotenv import load_dotenv
from google import genai
from google.genai import types
from google.genai.errors import APIError

# Load environment variables from .env if present
load_dotenv()

try:
    # Initialize the client (reads GEMINI_API_KEY from environment or .env)
    client = genai.Client()

    # Generate content using Gemma 4 with advanced configurations
    response = client.models.generate_content(
        model="gemma-4-26b-a4b-it",
        contents="Explain how quantum entanglement works in simple terms.",
        config=types.GenerateContentConfig(
            thinking_config=types.ThinkingConfig(thinking_level="high"),
            system_instruction="You are a clear and concise physics educator. Keep responses under 4 sentences."
        ),
    )

    print("Response from Gemma 4:")
    print(response.text)

except APIError as e:
    print(f"\n[API Error]: {e}", file=sys.stderr)
    print("Troubleshooting: Please ensure GEMINI_API_KEY is valid and the model is supported in your region.", file=sys.stderr)
    sys.exit(1)
except Exception as e:
    print(f"\n[Error]: {e}", file=sys.stderr)
    sys.exit(1)
