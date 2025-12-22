import asyncio
import inspect
import io
import os
import wave
from typing import Any, Callable

import numpy as np
import sounddevice as sd
import webrtcvad
from dotenv import load_dotenv
from openai import AsyncOpenAI

from wake_word import WakeWordDetector

SAMPLE_RATE = 16000
FRAME_DURATION_MS = 30
FRAME_SIZE = int(SAMPLE_RATE * FRAME_DURATION_MS / 1000)
SILENCE_TIMEOUT_MS = 800

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
        self.vad: webrtcvad.Vad = webrtcvad.Vad(2)

        # Wake word detection
        self.wake_word_detector = WakeWordDetector()
        self.wake_word_detected: bool = False
        self.wake_word_buffer: list[bytes] = []

        self.shutdown_event: asyncio.Event = shutdown_event
        self.loop: asyncio.AbstractEventLoop = loop
        self.audio_queue: asyncio.Queue[bytes] = audio_queue
        asyncio.create_task(self._audio_worker())

    async def start_listening(
        self,
        callback: Callable[[str], Any] | None = None,
    ):
        self.callback = callback

        with sd.InputStream(
            channels=1,
            samplerate=SAMPLE_RATE,
            blocksize=FRAME_SIZE,
            callback=self._input_stream_callback,
        ):
            await self.shutdown_event.wait()

    def _input_stream_callback(self, indata: np.ndarray, _f: int, _t: Any, _s: Any):
        pcm = (indata[:, 0] * 32768).astype(np.int16).tobytes()

        # Always check for wake word if not already detected
        if not self.wake_word_detected:
            self.wake_word_buffer.append(pcm)
            # Keep only the last ~2 seconds for wake word detection
            # (2 seconds * 1000ms / 30ms per frame = ~67 frames)
            if len(self.wake_word_buffer) > 67:
                self.wake_word_buffer.pop(0)

            # Check for wake word with the current frame
            if self.wake_word_detector.detect(pcm):
                self.wake_word_detected = True
                self.wake_word_buffer = []
                print("Wake word detected! Now listening for command...")
                return

        # Only process speech after wake word is detected
        if self.wake_word_detected:
            is_speech: Any = self.vad.is_speech(pcm, SAMPLE_RATE)

            if is_speech:
                self.speaking = True
                self.silence_ms = 0
                self.buffer.append(pcm)
            elif self.speaking:
                self.silence_ms += FRAME_DURATION_MS
                self.buffer.append(pcm)

                if self.silence_ms >= SILENCE_TIMEOUT_MS:
                    audio = b"".join(self.buffer)
                    self.buffer = []
                    self.speaking = False
                    self.silence_ms = 0
                    self.wake_word_detected = False  # Reset for next wake word
                    self.wake_word_detector.reset()

                    self.loop.call_soon_threadsafe(self.audio_queue.put_nowait, audio)

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
