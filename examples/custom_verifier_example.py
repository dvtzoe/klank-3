"""
Example: How to integrate a custom verifier model into AudioListener

This example shows how to modify the AudioListener class to use a custom
verifier model trained with the train_wakeword.py script.

After training your custom model with:
    python scripts/train_wakeword.py --output custom_models/my_wakeword.joblib

You can integrate it into the AudioListener as shown below.

NOTE: This example uses internal APIs of openwakeword (preprocessor.get_features
and model_inputs) which were tested with openwakeword>=0.4.0. If you encounter
errors with different versions, you may need to adjust the feature extraction code.
"""

import asyncio
from typing import Any, Callable

import joblib
import numpy as np
import pyaudio
import torch
from openwakeword.model import Model as WakeWordModel
from typing_extensions import Coroutine

# Tunables
WAKEWORD_THRESHOLD = 0.5
CUSTOM_VERIFIER_THRESHOLD = 0.5  # Threshold for custom verifier (0-1)

# Audio settings
SAMPLE_RATE = 16000
CHUNK_SIZE = 512
FORMAT = pyaudio.paInt16
CHANNELS = 1


class AudioListenerWithCustomModel:
    """
    Extended AudioListener that supports custom verifier models.

    This class loads both the base wakeword model and a custom verifier model
    to provide more personalized wakeword detection.
    """

    def __init__(self, custom_verifier_path: str | None = None):
        self.pyaudio: pyaudio.PyAudio = pyaudio.PyAudio()
        self.wakeword: WakeWordModel = WakeWordModel()

        # Load custom verifier if provided
        self.custom_verifier = None
        self.use_custom_verifier = False
        if custom_verifier_path:
            try:
                self.custom_verifier = joblib.load(custom_verifier_path)
                self.use_custom_verifier = True
                print(f"Loaded custom verifier model from: {custom_verifier_path}")
            except Exception as e:
                print(f"Warning: Could not load custom verifier: {e}")
                print("Falling back to base model only")

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

    def verify_with_custom_model(
        self, features: np.ndarray, model_name: str
    ) -> bool:
        """
        Verify detection using the custom verifier model.

        Args:
            features: Audio features from the wakeword model
            model_name: Name of the detected model

        Returns:
            True if the custom verifier confirms the detection, False otherwise
        """
        if not self.use_custom_verifier:
            return True  # Skip verification if no custom model

        try:
            # Get prediction from custom verifier
            # The verifier outputs a probability that this is the user's voice
            prediction = self.custom_verifier.predict_proba([features.flatten()])
            score = prediction[0][1]  # Probability of positive class

            print(f"Custom verifier score: {score:.3f}")
            return score >= CUSTOM_VERIFIER_THRESHOLD
        except Exception as e:
            print(f"Custom verifier error: {e}")
            return True  # Fall back to accepting the detection

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
        if self.use_custom_verifier:
            print("Custom verifier enabled for personalized detection")

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
                        if score > WAKEWORD_THRESHOLD:
                            # Get features for custom verification
                            if self.use_custom_verifier:
                                try:
                                    # Extract features from the model for verification
                                    # Note: This uses internal API of openwakeword which may change
                                    # This is the same approach used by train_custom_verifier()
                                    features = self.wakeword.preprocessor.get_features(  # pyright: ignore[reportAttributeAccessIssue]
                                        self.wakeword.model_inputs[name]  # pyright: ignore[reportAttributeAccessIssue]
                                    )

                                    # Verify with custom model
                                    if not self.verify_with_custom_model(features, name):
                                        print(
                                            f"Wake word detected ({name}) but rejected by custom verifier"
                                        )
                                        continue
                                except (AttributeError, KeyError) as e:
                                    print(
                                        f"Warning: Could not extract features for verification: {e}"
                                    )
                                    # Fall through to accept the detection

                            print(f"Wake word detected and verified ({name})")
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


# Example usage
async def main():
    """Example of using the custom verifier model."""
    # Initialize with custom verifier
    audio_listener = AudioListenerWithCustomModel(
        custom_verifier_path="custom_models/my_wakeword.joblib"
    )

    # Define callback
    async def on_wakeword_detected(audio_data: Any):
        print("Wakeword detected with custom verification!")
        # Process audio...

    # Start listening
    shutdown_event = asyncio.Event()
    await audio_listener.listen(on_wakeword_detected, shutdown_event)


if __name__ == "__main__":
    asyncio.run(main())
