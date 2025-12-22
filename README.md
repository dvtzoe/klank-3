# Klank 3

> klank klank klank

## A chatbot that always listens

uses OpenAI-compatible api for llm and stt

### Wake Word Detection

Klank 3 now includes wake word detection using OpenWakeWord. The bot listens continuously but only responds after hearing one of the configured wake words.

**Available wake words:**
- `alexa`
- `hey_jarvis`
- `hey_mycroft`
- `timer`
- `weather`

All wake words are loaded by default. Say any of them to activate the bot.

**Configuration:**

Set the detection threshold in your `.env` file:
```
WAKE_WORD_THRESHOLD=0.5
```

The threshold (0.0-1.0) controls detection sensitivity. Lower values are more sensitive but may have false positives.
