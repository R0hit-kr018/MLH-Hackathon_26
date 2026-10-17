import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file if it exists
dotenv_path = Path(__file__).parent / ".env"
if dotenv_path.exists():
    load_dotenv(dotenv_path=dotenv_path)
else:
    load_dotenv()

DEFAULT_MODEL = os.getenv("GEMINI_MODEL", "gemma-4-26b-a4b-it")
DEFAULT_SYSTEM_INSTRUCTION = (
    "You are a clear and concise physics educator. Keep responses under 4 sentences."
)
DEFAULT_THINKING_LEVEL = "high"


def get_api_key() -> str | None:
    """Retrieve GEMINI_API_KEY from environment or .env."""
    return os.getenv("GEMINI_API_KEY")
