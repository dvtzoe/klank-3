import io
import os

import sounddevice as sd
import soundfile as sf
from dotenv import load_dotenv
from openai import AsyncOpenAI

MODEL = "speaches-ai/Kokoro-82M-v1.0-ONNX"
VOICE = "af_heart"


class TTSClient:
    def __init__(self):
        load_dotenv()
        base_url = os.getenv("OPENAI_TTS_BASE_URL", "http://localhost:8000/v1")
        api_key = os.getenv("OPENAI_TTS_API_KEY")
        self.openai: AsyncOpenAI = AsyncOpenAI(api_key=api_key, base_url=base_url)

    async def create(self, text: str, voice: str = "default") -> bytes:
        response = await self.openai.audio.speech.create(
            model=MODEL,
            input=text,
            voice=voice,
            response_format="wav",
            speed=1.0,
        )
        with open("output.wav", "wb") as f:
            f.write(response.response.read())
        return response.response.read()

    async def create_and_read(self, text: str, voice: str = VOICE):
        audio_data = await self.create(text, voice)
        data, samplerate = sf.read(io.BytesIO(audio_data))
        sd.play(data, samplerate)
