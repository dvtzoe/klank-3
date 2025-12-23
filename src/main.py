import asyncio
import signal
from typing import Any

from audio_listener import AudioListener
from llm import LLMClient
from stt import STTClient
from tts import TTSClient

shutdown_event = asyncio.Event()


def setup_signal_handlers(loop: asyncio.AbstractEventLoop):
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, shutdown_event.set)


async def cleanup():
    pending = [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]

    for task in pending:
        task.cancel()

    await asyncio.gather(*pending, return_exceptions=True)


async def main():
    loop = asyncio.get_running_loop()
    setup_signal_handlers(loop)

    llm_client = LLMClient()
    stt_client = STTClient()
    tts_client = TTSClient()
    audio_listener = AudioListener()

    async def c(audio_data: Any):
        print("Received audio data, sending to STT...")
        transcript = await stt_client.transcribe_audio(audio_data)
        print("Transcription:", transcript)
        response = await llm_client.get_response(
            {"role": "user", "content": transcript}
        )
        print("LLM Response:", response)
        if response:
            await tts_client.create_and_read(response)

    try:
        await audio_listener.listen(c, shutdown_event)
    except KeyboardInterrupt:
        print("Exiting...")

    finally:
        shutdown_event.set()
        await cleanup()


if __name__ == "__main__":
    asyncio.run(main())
