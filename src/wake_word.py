import os
from typing import Any

import numpy as np
from dotenv import load_dotenv
from openwakeword.model import Model

SAMPLE_RATE = 16000


class WakeWordDetector:
    def __init__(self):
        load_dotenv()

        # Initialize the wake word detection model
        # By default, loads all pre-trained models (alexa, hey_jarvis, hey_mycroft, timer)
        # You can optionally specify wakeword_model_paths to load specific models
        self.model = Model()
        self.detection_threshold = float(os.getenv("WAKE_WORD_THRESHOLD", "0.5"))
        print(
            f"Initialized wake word detector with threshold: {self.detection_threshold}"
        )
        print(f"Loaded wake word models: {list(self.model.models.keys())}")

    def detect(self, audio_data: bytes) -> bool:
        """
        Detect wake word in audio data.

        Args:
            audio_data: PCM audio data as bytes (16-bit signed integer)

        Returns:
            True if wake word detected, False otherwise
        """
        # Convert bytes to numpy array
        audio_array = np.frombuffer(audio_data, dtype=np.int16)

        # Normalize to float32 in range [-1, 1]
        audio_float = audio_array.astype(np.float32) / 32768.0

        # Predict
        predictions: Any = self.model.predict(audio_float)

        # Check if any wake word was detected above threshold
        for wake_word, score in predictions.items():
            if score > self.detection_threshold:
                print(f"Wake word '{wake_word}' detected with score: {score}")
                return True

        return False

    def reset(self):
        """Reset the model state."""
        self.model.reset()
