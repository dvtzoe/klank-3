import io
import os

import pyaudio
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
        self.pyaudio = pyaudio.PyAudio()
    
    def __del__(self):
        if hasattr(self, 'pyaudio'):
            self.pyaudio.terminate()

    async def create(self, text: str, voice: str = "default") -> bytes:
        response = await self.openai.audio.speech.create(
            model=MODEL,
            input=text,
            voice=voice,
            response_format="wav",
            speed=1.0,
        )
        audio_data = response.response.read()
        with open("output.wav", "wb") as f:
            f.write(audio_data)
        return audio_data

    async def create_and_read(self, text: str, voice: str = VOICE):
        audio_data = await self.create(text, voice)
        data, samplerate = sf.read(io.BytesIO(audio_data))
        
        # Convert to the appropriate format for PyAudio
        if len(data.shape) == 1:
            # Mono audio
            audio_data_int16 = (data * 32768.0).astype('int16')
            channels = 1
        else:
            # Stereo audio
            audio_data_int16 = (data * 32768.0).astype('int16')
            channels = data.shape[1]
        
        # Open stream and play
        stream = self.pyaudio.open(
            format=pyaudio.paInt16,
            channels=channels,
            rate=samplerate,
            output=True,
        )
        
        stream.write(audio_data_int16.tobytes())
        stream.stop_stream()
        stream.close()
