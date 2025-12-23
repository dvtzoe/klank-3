# Wakeword Training Scripts

This directory contains scripts for training personalized wakeword models.

## Scripts

### `record_samples.py`
Records user voice samples for training a personalized wakeword model.

**Usage:**
```bash
python scripts/record_samples.py --num-samples 10 --duration 2
```

**Options:**
- `--output-dir`: Directory to save recordings (default: `training_data/positive`)
- `--num-samples`: Number of samples to record (default: 10)
- `--duration`: Duration of each recording in seconds (default: 2)
- `--prefix`: Prefix for output filenames (default: `wakeword`)

**Tips:**
- Record 10-20 samples for best results
- Speak clearly and naturally
- Vary your tone and speed slightly between samples
- Record in a quiet environment

### `train_wakeword.py`
Trains a custom verifier model using recorded samples.

**Usage:**
```bash
python scripts/train_wakeword.py --model-name alexa --output custom_models/my_wakeword.joblib
```

**Options:**
- `--positive-dir`: Directory containing your wakeword recordings (default: `training_data/positive`)
- `--negative-dir`: Directory to store/generate negative samples (default: `training_data/negative`)
- `--augmented-dir`: Directory to store augmented positive samples (default: `training_data/augmented`)
- `--output`: Output path for the trained model (default: `custom_models/custom_verifier.joblib`)
- `--model-name`: Base wakeword model to use (e.g., `alexa`, `hey_jarvis`)
- `--num-negative`: Number of negative samples to generate (default: 50)
- `--noise-levels`: Noise levels for augmentation (default: 0.0 0.02 0.05)
- `--skip-download`: Skip generating negative samples
- `--skip-augmentation`: Skip augmenting positive samples

**Available Base Models:**
Run `python -c "from openwakeword import get_pretrained_model_paths; print(get_pretrained_model_paths())"` to see all available models.

Common models include:
- `alexa`
- `hey_jarvis`
- `ok_google`
- And others...

## Workflow

1. **Record samples:**
   ```bash
   python scripts/record_samples.py --num-samples 15
   ```

2. **Train the model:**
   ```bash
   python scripts/train_wakeword.py --model-name alexa
   ```

3. **Integrate the model:**
   See `examples/custom_verifier_example.py` for integration examples.

## Technical Details

### Audio Format Requirements
- **Sample rate:** 16kHz
- **Bit depth:** 16-bit
- **Channels:** Mono (1 channel)
- **Format:** WAV

These requirements are enforced by the openwakeword library and handled automatically by the scripts.

### Training Process

1. **Positive Samples:** Your voice recordings saying the wakeword
2. **Data Augmentation:** Adds noise to positive samples to improve robustness
3. **Negative Samples:** Generated background noise to teach the model what NOT to detect
4. **Model Training:** Uses scikit-learn logistic regression with features from the base openwakeword model

### Custom Verifier vs Full Model Training

The scripts use openwakeword's **custom verifier** approach, which:
- Requires less data (10-20 samples vs thousands)
- Trains faster (seconds vs hours)
- Works with existing pretrained models
- Personalizes detection to your voice

For training completely new wakewords from scratch, refer to the openwakeword documentation on full model training.
