import os

from dotenv import load_dotenv
from openai import AsyncOpenAI

MODEL = "Systran/faster-distil-whisper-large-v3"


class STTClient:
    def __init__(
        self,
    ):
        load_dotenv()
        base_url = os.getenv("OPENAI_STT_BASE_URL", "http://localhost:8000/v1")
        api_key = os.getenv("OPENAI_STT_API_KEY")
        self.openai: AsyncOpenAI = AsyncOpenAI(api_key=api_key, base_url=base_url)

    async def transcribe_audio(self, audio_data: bytes) -> str:
        response = await self.openai.audio.transcriptions.create(
            model=MODEL,
            file=audio_data,
            response_format="text",
        )
        return response
