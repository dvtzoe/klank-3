#!/usr/bin/env python3
"""
Script to record user voice samples for training a personalized wakeword model.

This script records short audio clips of the user saying the wakeword,
saving them as 16-bit, 16kHz mono WAV files required for training.
"""

import argparse
import wave
from pathlib import Path

import numpy as np
import pyaudio

# Audio settings (required by openwakeword)
SAMPLE_RATE = 16000
CHUNK_SIZE = 512
FORMAT = pyaudio.paInt16
CHANNELS = 1


def record_audio(duration_seconds: int, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    """Record audio from the microphone."""
    p = pyaudio.PyAudio()

    print(f"Recording for {duration_seconds} seconds...")
    print("Speak your wakeword now!")

    stream = p.open(
        format=FORMAT,
        channels=CHANNELS,
        rate=sample_rate,
        input=True,
        frames_per_buffer=CHUNK_SIZE,
    )

    frames = []
    for _ in range(0, int(sample_rate / CHUNK_SIZE * duration_seconds)):
        data = stream.read(CHUNK_SIZE, exception_on_overflow=False)
        frames.append(data)

    stream.stop_stream()
    stream.close()
    p.terminate()

    print("Recording complete!")

    # Convert to numpy array
    audio_data = np.frombuffer(b"".join(frames), dtype=np.int16)
    return audio_data


def save_wav(filename: str, audio_data: np.ndarray, sample_rate: int = SAMPLE_RATE):
    """Save audio data to a WAV file."""
    with wave.open(filename, "wb") as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(2)  # 16-bit = 2 bytes
        wf.setframerate(sample_rate)
        wf.writeframes(audio_data.tobytes())


def main():
    parser = argparse.ArgumentParser(
        description="Record voice samples for wakeword training"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="training_data/positive",
        help="Directory to save recordings (default: training_data/positive)",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=10,
        help="Number of samples to record (default: 10)",
    )
    parser.add_argument(
        "--duration",
        type=int,
        default=2,
        help="Duration of each recording in seconds (default: 2)",
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default="wakeword",
        help="Prefix for output filenames (default: wakeword)",
    )

    args = parser.parse_args()

    # Create output directory if it doesn't exist
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print("Wakeword Recording Script")
    print(f"{'='*60}")
    print(f"Output directory: {output_dir}")
    print(f"Number of samples: {args.num_samples}")
    print(f"Duration per sample: {args.duration} seconds")
    print(f"{'='*60}\n")

    for i in range(args.num_samples):
        print(f"\nSample {i + 1}/{args.num_samples}")
        input("Press Enter when ready to record...")

        audio_data = record_audio(args.duration)

        # Save to file
        filename = output_dir / f"{args.prefix}_{i + 1:03d}.wav"
        save_wav(str(filename), audio_data)
        print(f"Saved to: {filename}")

    print(f"\n{'='*60}")
    print(f"Recording complete! {args.num_samples} samples saved to {output_dir}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
