#!/usr/bin/env python3
"""
Script to train a personalized wakeword model.

This script:
1. Downloads negative samples from Common Voice dataset
2. Applies noise augmentation to positive samples
3. Trains a custom verifier model using openwakeword

Usage:
    python scripts/train_wakeword.py --positive-dir training_data/positive \\
                                     --model-name alexa \\
                                     --output custom_models/my_wakeword.joblib
"""

import argparse
import random
import wave
from pathlib import Path

import numpy as np
from openwakeword import train_custom_verifier

# Audio settings
SAMPLE_RATE = 16000


def download_negative_samples(
    output_dir: Path, num_samples: int = 50, max_duration: int = 10
):
    """
    Download negative samples from Mozilla Common Voice or generate synthetic ones.

    For now, this creates simple negative samples by generating silence with some noise.
    In production, you would download from a real dataset like Common Voice.

    Args:
        output_dir: Directory to save negative samples
        num_samples: Number of negative samples to generate
        max_duration: Maximum duration of each sample in seconds
    """
    print(f"Generating {num_samples} negative samples...")
    output_dir.mkdir(parents=True, exist_ok=True)

    for i in range(num_samples):
        # Generate random duration between 1 and max_duration seconds
        duration = random.uniform(1.0, max_duration)
        num_samples_audio = int(SAMPLE_RATE * duration)

        # Generate low-level noise (simulating background speech/noise)
        # In production, you'd want to use actual speech samples
        audio_data = np.random.normal(0, 100, num_samples_audio).astype(np.int16)

        # Add some random amplitude modulation to make it more realistic
        envelope = np.random.uniform(0.5, 1.0, num_samples_audio)
        audio_data = (audio_data * envelope).astype(np.int16)

        # Save to WAV file
        filename = output_dir / f"negative_{i + 1:03d}.wav"
        with wave.open(str(filename), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)  # 16-bit
            wf.setframerate(SAMPLE_RATE)
            wf.writeframes(audio_data.tobytes())

    print(f"Generated {num_samples} negative samples in {output_dir}")


def augment_positive_samples(
    input_dir: Path, output_dir: Path, noise_levels: list[float] = None
):
    """
    Augment positive samples by adding noise.

    Args:
        input_dir: Directory containing original positive samples
        output_dir: Directory to save augmented samples
        noise_levels: List of noise levels (as fraction of signal) to add
    """
    if noise_levels is None:
        noise_levels = [0.0, 0.02, 0.05]  # Original + 2 noise levels

    print(f"Augmenting positive samples with noise levels: {noise_levels}")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Copy and augment each file
    wav_files = list(input_dir.glob("*.wav"))
    if not wav_files:
        raise ValueError(f"No WAV files found in {input_dir}")

    for wav_file in wav_files:
        # Read original file
        with wave.open(str(wav_file), "rb") as wf:
            params = wf.getparams()
            audio_data = np.frombuffer(wf.readframes(params.nframes), dtype=np.int16)

        # Create augmented versions with different noise levels
        for idx, noise_level in enumerate(noise_levels):
            if noise_level == 0.0:
                # Just copy the original
                augmented = audio_data
                suffix = "orig"
            else:
                # Add Gaussian noise
                noise = np.random.normal(0, noise_level * 32768, len(audio_data))
                augmented = (audio_data + noise).astype(np.int16)
                suffix = f"noise{int(noise_level * 100):02d}"

            # Save augmented file
            output_filename = (
                output_dir / f"{wav_file.stem}_{suffix}.wav"
            )
            with wave.open(str(output_filename), "wb") as wf:
                wf.setparams(params)
                wf.writeframes(augmented.tobytes())

    num_output = len(list(output_dir.glob("*.wav")))
    print(f"Created {num_output} augmented samples in {output_dir}")


def train_model(
    positive_dir: Path,
    negative_dir: Path,
    output_path: Path,
    model_name: str = "alexa",
):
    """
    Train the custom verifier model.

    Args:
        positive_dir: Directory containing positive samples
        negative_dir: Directory containing negative samples
        output_path: Path to save the trained model
        model_name: Base wakeword model to use (e.g., 'alexa', 'hey_jarvis')
    """
    # Get list of WAV files from directories
    positive_files = sorted([str(f) for f in positive_dir.glob("*.wav")])
    negative_files = sorted([str(f) for f in negative_dir.glob("*.wav")])

    if not positive_files:
        raise ValueError(f"No WAV files found in {positive_dir}")
    if not negative_files:
        raise ValueError(f"No WAV files found in {negative_dir}")

    print("\nTraining custom verifier model...")
    print(f"  Positive samples: {len(positive_files)} files from {positive_dir}")
    print(f"  Negative samples: {len(negative_files)} files from {negative_dir}")
    print(f"  Base model: {model_name}")
    print(f"  Output: {output_path}")

    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Train the model - pass lists of file paths, not directory paths
    train_custom_verifier(
        positive_reference_clips=positive_files,
        negative_reference_clips=negative_files,
        output_path=str(output_path),
        model_name=model_name,
    )

    print(f"\nModel trained successfully and saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Train a personalized wakeword model"
    )
    parser.add_argument(
        "--positive-dir",
        type=str,
        default="training_data/positive",
        help="Directory containing positive wakeword samples",
    )
    parser.add_argument(
        "--negative-dir",
        type=str,
        default="training_data/negative",
        help="Directory to store/generate negative samples",
    )
    parser.add_argument(
        "--augmented-dir",
        type=str,
        default="training_data/augmented",
        help="Directory to store augmented positive samples",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="custom_models/custom_verifier.joblib",
        help="Output path for the trained model",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="alexa",
        help="Base wakeword model to use (e.g., 'alexa', 'hey_jarvis')",
    )
    parser.add_argument(
        "--num-negative",
        type=int,
        default=50,
        help="Number of negative samples to generate (default: 50)",
    )
    parser.add_argument(
        "--noise-levels",
        type=float,
        nargs="+",
        default=[0.0, 0.02, 0.05],
        help="Noise levels for augmentation (default: 0.0 0.02 0.05)",
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Skip downloading/generating negative samples",
    )
    parser.add_argument(
        "--skip-augmentation",
        action="store_true",
        help="Skip augmenting positive samples",
    )

    args = parser.parse_args()

    positive_dir = Path(args.positive_dir)
    negative_dir = Path(args.negative_dir)
    augmented_dir = Path(args.augmented_dir)
    output_path = Path(args.output)

    print(f"\n{'='*60}")
    print("Wakeword Training Script")
    print(f"{'='*60}\n")

    # Validate positive samples exist
    if not positive_dir.exists() or not list(positive_dir.glob("*.wav")):
        raise ValueError(
            f"No positive samples found in {positive_dir}. "
            f"Please record samples using scripts/record_samples.py first."
        )

    num_positive = len(list(positive_dir.glob("*.wav")))
    print(f"Found {num_positive} positive samples in {positive_dir}")

    # Step 1: Download/generate negative samples
    if not args.skip_download:
        if not negative_dir.exists() or not list(negative_dir.glob("*.wav")):
            download_negative_samples(negative_dir, num_samples=args.num_negative)
        else:
            print(f"Negative samples already exist in {negative_dir}, skipping download")
    else:
        print("Skipping negative sample generation")

    # Step 2: Augment positive samples
    if not args.skip_augmentation:
        augment_positive_samples(positive_dir, augmented_dir, args.noise_levels)
        training_positive_dir = augmented_dir
    else:
        print("Skipping augmentation, using original samples")
        training_positive_dir = positive_dir

    # Step 3: Train the model
    train_model(training_positive_dir, negative_dir, output_path, args.model_name)

    print(f"\n{'='*60}")
    print("Training Complete!")
    print(f"{'='*60}")
    print(f"\nYour custom model is ready: {output_path}")
    print(
        "To use it, update your audio_listener.py to load this custom verifier model."
    )
    print(f"\n{'='*60}\n")


if __name__ == "__main__":
    main()
