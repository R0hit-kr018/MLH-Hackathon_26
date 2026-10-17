"""
Main execution script for Gemma 4 in Google Antigravity.
Configures system instructions, thinking configuration, and comprehensive error handling.
"""

import argparse
import sys
from gemma_client import GemmaClient, GemmaClientError
import config


def print_banner():
    print("=" * 65)
    print("        Gemma 4 with Gemini API & Thinking Configuration        ")
    print("=" * 65, flush=True)


def run_single_prompt(client: GemmaClient, prompt: str, model: str, system_instruction: str, thinking_level: str, stream: bool):
    print(f"\n[Model]: {model}")
    print(f"[Thinking Level]: {thinking_level}")
    print(f"[System Instruction]: {system_instruction}")
    print(f"[Prompt]: {prompt}\n")
    print("-" * 65)
    print("Response:")

    if stream:
        for chunk in client.stream_generate(
            prompt=prompt,
            model=model,
            system_instruction=system_instruction,
            thinking_level=thinking_level,
        ):
            print(chunk, end="", flush=True)
        print()
    else:
        response_text = client.generate(
            prompt=prompt,
            model=model,
            system_instruction=system_instruction,
            thinking_level=thinking_level,
        )
        print(response_text)

    print("-" * 65)


def run_interactive_mode(client: GemmaClient, model: str, system_instruction: str, thinking_level: str):
    print(f"\nEntering Interactive Mode with model: {model}")
    print("Type your questions below. Type 'exit', 'quit', or Ctrl+C to stop.\n")

    while True:
        try:
            prompt = input("\nYou > ").strip()
            if not prompt:
                continue
            if prompt.lower() in ("exit", "quit"):
                print("Exiting interactive mode. Goodbye!")
                break

            print("\nGemma > ", end="", flush=True)
            for chunk in client.stream_generate(
                prompt=prompt,
                model=model,
                system_instruction=system_instruction,
                thinking_level=thinking_level,
            ):
                print(chunk, end="", flush=True)
            print()
        except KeyboardInterrupt:
            print("\nSession interrupted. Exiting.")
            break
        except GemmaClientError as err:
            print(f"\n\n[Error]:\n{err}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Run Gemma with Google GenAI SDK, thinking configuration, and system instructions."
    )
    parser.add_argument(
        "--prompt", "-p",
        type=str,
        default="Explain how quantum entanglement works in simple terms.",
        help="Prompt to send to the model.",
    )
    parser.add_argument(
        "--model", "-m",
        type=str,
        default=config.DEFAULT_MODEL,
        help=f"Gemma or Gemini model to use (default: {config.DEFAULT_MODEL}).",
    )
    parser.add_argument(
        "--system", "-s",
        type=str,
        default=config.DEFAULT_SYSTEM_INSTRUCTION,
        help="System instruction defining persona and constraints.",
    )
    parser.add_argument(
        "--thinking", "-t",
        type=str,
        default=config.DEFAULT_THINKING_LEVEL,
        help="Thinking level (e.g. 'high', 'medium', 'low').",
    )
    parser.add_argument(
        "--stream",
        action="store_true",
        help="Stream response chunks as they arrive.",
    )
    parser.add_argument(
        "--interactive", "-i",
        action="store_true",
        help="Start an interactive chat session in the terminal.",
    )

    args = parser.parse_args()

    print_banner()

    try:
        # Initialize client with validation & error handling
        client = GemmaClient()
    except GemmaClientError as err:
        print(f"\nConfiguration Error:\n{err}\n", file=sys.stderr)
        sys.exit(1)

    try:
        if args.interactive:
            run_interactive_mode(
                client=client,
                model=args.model,
                system_instruction=args.system,
                thinking_level=args.thinking,
            )
        else:
            run_single_prompt(
                client=client,
                prompt=args.prompt,
                model=args.model,
                system_instruction=args.system,
                thinking_level=args.thinking,
                stream=args.stream,
            )
    except GemmaClientError as err:
        print(f"\nExecution Error:\n{err}\n", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
