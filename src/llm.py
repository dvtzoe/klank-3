import os

from dotenv import load_dotenv
from openai import AsyncOpenAI
from openai.types.chat.chat_completion_message_param import ChatCompletionMessageParam

SYSTEM_PROMPT = """You are a helpful assistant."""
MODEL = "google/gemini-2.0-flash-001"


class LLMClient:
    def __init__(self):
        load_dotenv()
        base_url = os.getenv("OPENAI_LLM_BASE_URL", "https://openrouter.ai/api/v1")
        api_key = os.getenv("OPENAI_LLM_API_KEY")

        if not api_key:
            raise ValueError(
                "API key not found. Please set OPENROUTER_API_KEY in your environment variables."
            )

        self.openrouter: AsyncOpenAI = AsyncOpenAI(api_key=api_key, base_url=base_url)
        self.chat_log: list[ChatCompletionMessageParam] = [
            {"role": "system", "content": SYSTEM_PROMPT}
        ]

    async def get_response(
        self, user_message: ChatCompletionMessageParam
    ) -> str | None:
        self.chat_log.append(user_message)

        response = await self.openrouter.chat.completions.create(
            model=MODEL,
            messages=self.chat_log,
        )

        bot_reply = response.choices[0].message.content
        return bot_reply
