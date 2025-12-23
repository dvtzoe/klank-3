# Klank 3

> klank klank klank

## A chatbot that always listens

uses OpenAI-compatible api for llm and stt

## Installation

1. Install dependencies:
   ```bash
   pip install -e .
   ```

2. Install PortAudio (required for PyAudio):
   - **Ubuntu/Debian:** `sudo apt-get install portaudio19-dev`
   - **macOS:** `brew install portaudio`
   - **Windows:** Usually included with PyAudio wheel

## Training Personal Wakeword Models

Klank 3 supports training personalized wakeword models using your own voice recordings. This allows you to create a custom wake word that responds specifically to your voice.

### Quick Start

1. **Record your voice samples**

   ```bash
   python scripts/record_samples.py --num-samples 10 --duration 2
   ```

   This will record 10 samples of you saying your wakeword, each 2 seconds long. The samples will be saved to `training_data/positive/`.

2. **Train the model**

   ```bash
   python scripts/train_wakeword.py --model-name alexa --output custom_models/my_wakeword.joblib
   ```

   This will:
   - Generate negative samples (background speech/noise) automatically
   - Augment your positive samples with noise for robustness
   - Train a custom verifier model based on the specified base model (e.g., 'alexa')

### Advanced Options

#### Recording Samples

```bash
python scripts/record_samples.py \
    --output-dir training_data/positive \
    --num-samples 15 \
    --duration 2 \
    --prefix my_wakeword
```

Options:
- `--output-dir`: Directory to save recordings (default: `training_data/positive`)
- `--num-samples`: Number of samples to record (default: 10, recommended: 10-20)
- `--duration`: Duration of each recording in seconds (default: 2)
- `--prefix`: Prefix for output filenames (default: `wakeword`)

#### Training the Model

```bash
python scripts/train_wakeword.py \
    --positive-dir training_data/positive \
    --model-name alexa \
    --output custom_models/my_wakeword.joblib \
    --num-negative 100 \
    --noise-levels 0.0 0.02 0.05 0.1
```

Options:
- `--positive-dir`: Directory containing your wakeword recordings
- `--model-name`: Base wakeword model to use (e.g., `alexa`, `hey_jarvis`)
- `--output`: Path to save the trained model
- `--num-negative`: Number of negative samples to generate (default: 50)
- `--noise-levels`: Noise levels for data augmentation (default: 0.0 0.02 0.05)
- `--skip-download`: Skip generating negative samples
- `--skip-augmentation`: Skip augmenting positive samples

### How It Works

1. **Positive Samples**: You record yourself saying the wakeword multiple times
2. **Negative Samples**: The script automatically generates background noise samples
3. **Data Augmentation**: Your recordings are augmented with various noise levels to improve robustness
4. **Training**: A custom verifier model is trained using openwakeword's `train_custom_verifier` function

### Using Your Custom Model

After training, you'll need to integrate the custom model into the audio listener. The trained model will be saved as a `.joblib` file in the `custom_models/` directory.

To use it in your application, modify the `AudioListener` class to load the custom verifier model alongside the base wakeword model.
