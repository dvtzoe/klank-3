import asyncio
from typing import Any, Callable

import numpy as np
import pyaudio
import torch
from openwakeword.model import Model as WakeWordModel
from typing_extensions import Coroutine

# Tunables
WAKEWORD_THRESHOLD = 0.5

# Audio settings
SAMPLE_RATE = 16000
CHUNK_SIZE = 512
FORMAT = pyaudio.paInt16
CHANNELS = 1


class AudioListener:
    def __init__(
        self,
    ):
        self.pyaudio: pyaudio.PyAudio = pyaudio.PyAudio()
        self.wakeword: WakeWordModel = WakeWordModel()

        self.model: Any
        self.utils: Any
        self.get_speech_timestamps: Any
        self.save_audio: Any
        self.read_audio: Any
        self.VADIterator: Any
        self.collect_chunks: Any

        self.model, self.utils = torch.hub.load(  # pyright: ignore[reportGeneralTypeIssues]
            repo_or_dir="snakers4/silero-vad", model="silero_vad", force_reload=False
        )
        (
            self.get_speech_timestamps,
            self.save_audio,
            self.read_audio,
            self.VADIterator,
            self.collect_chunks,
        ) = self.utils

        self.stream: pyaudio.Stream
        self.vad: Any = self.VADIterator(self.model)
        self.speech_buffer: Any = []
        self.is_listening: bool = False

    async def listen(
        self,
        callback: Callable[[Any], Coroutine[Any, Any, None]],
        shutdown_event: asyncio.Event,
    ) -> None:
        self.stream = self.pyaudio.open(
            format=FORMAT,
            channels=CHANNELS,
            rate=SAMPLE_RATE,
            input=True,
            frames_per_buffer=CHUNK_SIZE,
        )

        print(f"Available Wake Words: {self.wakeword.models.keys()}")

        try:
            print("Listening for wake word...")
            loop = asyncio.get_running_loop()
            while not shutdown_event.is_set():
                data = await loop.run_in_executor(
                    None,
                    lambda: self.stream.read(CHUNK_SIZE, exception_on_overflow=False),
                )

                # int16 -> float32 [-1, 1]
                audio = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0

                if self.is_listening:
                    audio_tensor = torch.from_numpy(audio)
                    speech_event = self.vad(audio_tensor, return_seconds=False)

                    if speech_event is not None:
                        if "start" in speech_event:
                            self.speech_buffer = []

                        if "end" in speech_event:
                            print(f"Captured {len(self.speech_buffer)} chunks")
                            self.speech_buffer = []

                            await callback(torch.cat(self.speech_buffer))

                    if self.vad.triggered:
                        self.speech_buffer.append(audio)
                else:
                    scores: dict[Any, Any] = self.wakeword.predict(audio)  # pyright: ignore[reportAssignmentType]

                    for name, score in scores.items():
                        print(score)
                        if score > WAKEWORD_THRESHOLD:
                            print(f"Wake word detected ({name})")
                            self.is_listening = True
                            self.vad.reset_states()
                            self.speech_buffer.clear()
                            break
            print("Shutting down audio listener...")

        except KeyboardInterrupt:
            print("\nStopped")

        finally:
            self.cleanup()

    def cleanup(self) -> None:
        if self.stream.is_active():
            self.stream.stop_stream()
            self.stream.close()
        self.pyaudio.terminate()
