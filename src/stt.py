import asyncio
import inspect
import io
import os
import wave
from typing import Any, Callable

import numpy as np
import pyaudio
import torch
from dotenv import load_dotenv
from openai import AsyncOpenAI

SAMPLE_RATE = 16000
FRAME_DURATION_MS = 30
FRAME_SIZE = int(SAMPLE_RATE * FRAME_DURATION_MS / 1000)
SILENCE_TIMEOUT_MS = 800
# Silero VAD requires at least 512 samples (32ms at 16kHz)
MIN_VAD_SAMPLES = 512

MODEL = "Systran/faster-distil-whisper-large-v3"


class STTClient:
    def __init__(
        self,
        shutdown_event: asyncio.Event,
        loop: asyncio.AbstractEventLoop,
        audio_queue: asyncio.Queue[bytes],
    ):
        load_dotenv()
        base_url = os.getenv("OPENAI_STT_BASE_URL", "http://localhost:8000/v1")
        api_key = os.getenv("OPENAI_STT_API_KEY")
        self.speaches: AsyncOpenAI = AsyncOpenAI(api_key=api_key, base_url=base_url)

        self.callback: Callable[[str], Any] | None = None
        self.buffer: list[bytes] = []
        self.speaking: bool = False
        self.silence_ms: int = 0
        
        # Buffer for accumulating audio chunks for VAD
        self.vad_buffer: list[np.ndarray] = []
        self.vad_buffer_size: int = 0
        
        # Load Silero VAD model
        self.vad_model, _ = torch.hub.load(
            repo_or_dir="snakers4/silero-vad",
            model="silero_vad",
            force_reload=False,
            onnx=False,
        )
        self.vad_model.eval()

        self.shutdown_event: asyncio.Event = shutdown_event
        self.loop: asyncio.AbstractEventLoop = loop
        self.audio_queue: asyncio.Queue[bytes] = audio_queue
        
        # Initialize PyAudio
        self.pyaudio = pyaudio.PyAudio()
        self.stream = None
        
        asyncio.create_task(self._audio_worker())

    async def start_listening(
        self,
        callback: Callable[[str], Any] | None = None,
    ):
        self.callback = callback

        # Open PyAudio stream
        self.stream = self.pyaudio.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=SAMPLE_RATE,
            input=True,
            frames_per_buffer=FRAME_SIZE,
            stream_callback=self._input_stream_callback,
        )
        
        self.stream.start_stream()
        
        try:
            await self.shutdown_event.wait()
        finally:
            if self.stream:
                self.stream.stop_stream()
                self.stream.close()
            self.pyaudio.terminate()

    def _input_stream_callback(self, in_data: bytes, frame_count: int, time_info: dict, status: int):
        # Convert bytes to int16 numpy array
        pcm = np.frombuffer(in_data, dtype=np.int16)
        
        # Normalize to float32 for Silero VAD (expected range: -1 to 1)
        audio_float32 = pcm.astype(np.float32) / 32768.0
        
        # Accumulate audio for VAD
        self.vad_buffer.append(audio_float32)
        self.vad_buffer_size += len(audio_float32)
        
        # Only run VAD when we have enough samples
        is_speech = False
        if self.vad_buffer_size >= MIN_VAD_SAMPLES:
            # Concatenate accumulated audio
            audio_chunk = np.concatenate(self.vad_buffer)
            audio_tensor = torch.from_numpy(audio_chunk)
            
            # Get speech probability from Silero VAD
            with torch.no_grad():
                speech_prob = self.vad_model(audio_tensor, SAMPLE_RATE).item()
            
            # Consider speech if probability > 0.5
            is_speech = speech_prob > 0.5
            
            # Reset VAD buffer
            self.vad_buffer = []
            self.vad_buffer_size = 0

        if is_speech:
            self.speaking = True
            self.silence_ms = 0
            self.buffer.append(in_data)
        elif self.speaking:
            self.silence_ms += FRAME_DURATION_MS
            self.buffer.append(in_data)

            if self.silence_ms >= SILENCE_TIMEOUT_MS:
                audio = b"".join(self.buffer)
                self.buffer = []
                self.speaking = False
                self.silence_ms = 0

                self.loop.call_soon_threadsafe(self.audio_queue.put_nowait, audio)
        
        return (None, pyaudio.paContinue)

    async def _process(self, audio_pcm: bytes):
        try:
            buffer = io.BytesIO()
            with wave.open(buffer, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(SAMPLE_RATE)
                wf.writeframes(audio_pcm)
            wav_bytes = buffer.getvalue()

            result = await self.speaches.audio.transcriptions.create(
                model=MODEL,
                file=wav_bytes,
            )
            if self.callback:
                if inspect.iscoroutinefunction(self.callback):
                    await self.callback(result.text)
                else:
                    self.callback(result.text)
            return result
        except Exception as e:
            print("STT failed:", e)

    async def _audio_worker(self):
        while not self.shutdown_event.is_set():
            audio = await self.audio_queue.get()
            try:
                await self._process(audio)
            except Exception as e:
                print("STT error:", e)
