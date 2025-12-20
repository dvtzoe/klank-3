import asyncio
import os

from dotenv import load_dotenv
from openai import AsyncOpenAI
from openai.types.chat.chat_completion_message_param import ChatCompletionMessageParam

SYSTEM_PROMPT = """You are a helpful assistant."""
MODEL = "google/gemini-2.0-flash-001"


async def main():
    _ = load_dotenv()

    api_key = os.getenv("OPENROUTER_API_KEY")

    if not api_key:
        raise ValueError(
            "API key not found. Please set OPENROUTER_API_KEY in your environment variables."
        )

    openai = AsyncOpenAI(api_key=api_key, base_url="https://openrouter.ai/api/v1")

    chat_log: list[ChatCompletionMessageParam] = [
        {"role": "system", "content": SYSTEM_PROMPT}
    ]

    while True:
        user_input = input("You: ")
        if user_input.lower() in ["exit", "quit"]:
            print("Goodbye!")
            break

        user_message: ChatCompletionMessageParam = {
            "role": "user",
            "content": user_input,
        }

        chat_log.append(user_message)

        response = await openai.chat.completions.create(
            model=MODEL,
            messages=chat_log,
        )

        bot_reply = response.choices[0].message.content
        print(f"Bot: {bot_reply}")


if __name__ == "__main__":
    asyncio.run(main())
