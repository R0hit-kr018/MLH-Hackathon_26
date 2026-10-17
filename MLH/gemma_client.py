"""
Gemma client module providing high-level interface to Google GenAI API
with system instructions, thinking configuration, and comprehensive error handling.
"""

from typing import Generator, Optional
import sys
from google import genai
from google.genai import types
from google.genai import errors

import config


class GemmaClientError(Exception):
    """Custom exception for Gemma Client operations with user-friendly diagnostics."""
    pass


class GemmaClient:
    """Wrapper around google-genai Client specifically configured for Gemma and Gemini models."""

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the Gemma client.
        
        Args:
            api_key: Optional Google GenAI API key. If not provided, it is read
                     from the environment or .env file (GEMINI_API_KEY).
        """
        resolved_key = api_key or config.get_api_key()
        if not resolved_key or resolved_key.strip() in ("", "your-api-key-here"):
            raise GemmaClientError(
                "Missing GEMINI_API_KEY!\n"
                "To fix this:\n"
                "  1. Obtain an API key from Google AI Studio: https://aistudio.google.com/apikey\n"
                "  2. Create a .env file in the project directory:\n"
                "       GEMINI_API_KEY=your_actual_key_here\n"
                "  3. Or set the environment variable in your terminal:\n"
                "       Windows PowerShell: $env:GEMINI_API_KEY='your_actual_key_here'\n"
                "       Linux/macOS:        export GEMINI_API_KEY='your_actual_key_here'"
            )
        
        try:
            self.client = genai.Client(api_key=resolved_key)
        except Exception as e:
            raise GemmaClientError(f"Failed to initialize GenAI client: {e}") from e

    def build_config(
        self,
        system_instruction: Optional[str] = None,
        thinking_level: str = "high",
        temperature: Optional[float] = None,
    ) -> types.GenerateContentConfig:
        """
        Build GenerateContentConfig with thinking and system instructions.

        Args:
            system_instruction: Guiding prompt defining persona/behavior.
            thinking_level: Level of thinking depth ('high', 'medium', 'low', etc.).
            temperature: Optional sampling temperature.
        """
        kwargs = {}

        if system_instruction:
            kwargs["system_instruction"] = system_instruction

        # Configure thinking if thinking_level is provided
        if thinking_level:
            kwargs["thinking_config"] = types.ThinkingConfig(thinking_level=thinking_level)

        if temperature is not None:
            kwargs["temperature"] = temperature

        return types.GenerateContentConfig(**kwargs)

    def generate(
        self,
        prompt: str,
        model: Optional[str] = None,
        system_instruction: Optional[str] = config.DEFAULT_SYSTEM_INSTRUCTION,
        thinking_level: str = config.DEFAULT_THINKING_LEVEL,
        temperature: Optional[float] = None,
    ) -> str:
        """
        Generate content using Gemma with advanced configurations and robust error handling.

        Args:
            prompt: User content/question.
            model: Model identifier (defaults to config.DEFAULT_MODEL).
            system_instruction: Guiding persona/rules for the model.
            thinking_level: Thinking depth (e.g. 'high').
            temperature: Sampling temperature.

        Returns:
            The generated text string.
        """
        target_model = model or config.DEFAULT_MODEL
        gen_config = self.build_config(
            system_instruction=system_instruction,
            thinking_level=thinking_level,
            temperature=temperature,
        )

        try:
            response = self.client.models.generate_content(
                model=target_model,
                contents=prompt,
                config=gen_config,
            )
            return response.text or ""

        except errors.APIError as api_err:
            error_msg = self._diagnose_api_error(api_err, target_model)
            raise GemmaClientError(error_msg) from api_err

        except Exception as exc:
            raise GemmaClientError(
                f"Unexpected error while communicating with model '{target_model}': {exc}"
            ) from exc

    def stream_generate(
        self,
        prompt: str,
        model: Optional[str] = None,
        system_instruction: Optional[str] = config.DEFAULT_SYSTEM_INSTRUCTION,
        thinking_level: str = config.DEFAULT_THINKING_LEVEL,
        temperature: Optional[float] = None,
    ) -> Generator[str, None, None]:
        """
        Stream response chunks as they arrive from the model.
        """
        target_model = model or config.DEFAULT_MODEL
        gen_config = self.build_config(
            system_instruction=system_instruction,
            thinking_level=thinking_level,
            temperature=temperature,
        )

        try:
            response_stream = self.client.models.generate_content_stream(
                model=target_model,
                contents=prompt,
                config=gen_config,
            )
            for chunk in response_stream:
                if chunk.text:
                    yield chunk.text

        except errors.APIError as api_err:
            error_msg = self._diagnose_api_error(api_err, target_model)
            raise GemmaClientError(error_msg) from api_err

        except Exception as exc:
            raise GemmaClientError(
                f"Unexpected error during streaming from model '{target_model}': {exc}"
            ) from exc

    def _diagnose_api_error(self, err: errors.APIError, model_name: str) -> str:
        """Provide actionable diagnosis for common API errors."""
        err_str = str(err)
        status_code = getattr(err, "code", None)

        diagnostics = [f"Google GenAI API Error (Model: {model_name}):"]

        if status_code in (401, 403) or "API_KEY_INVALID" in err_str:
            diagnostics.append(
                "-> [Authentication Error]: Your GEMINI_API_KEY appears invalid or lacks permissions.\n"
                "   Verify your key at https://aistudio.google.com/apikey"
            )
        elif status_code == 404 or "NOT_FOUND" in err_str:
            diagnostics.append(
                f"-> [Model Not Found]: Model '{model_name}' was not recognized or is unavailable in your region.\n"
                "   Check your model name in .env (e.g., GEMINI_MODEL=gemma-4-26b-a4b-it or gemini-2.5-flash)."
            )
        elif status_code == 429 or "RESOURCE_EXHAUSTED" in err_str:
            diagnostics.append(
                "-> [Quota Exceeded]: You have hit the rate limit or quota for this model.\n"
                "   Wait a moment before retrying or check your quota in Google AI Studio."
            )
        else:
            diagnostics.append(f"-> Details: {err_str}")

        return "\n".join(diagnostics)
