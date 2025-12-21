import asyncio
import signal

from llm import LLMClient
from stt import STTClient

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
    audio_queue = asyncio.Queue()
    setup_signal_handlers(loop)

    llm_client = LLMClient()
    stt_client = STTClient(
        shutdown_event=shutdown_event,
        loop=loop,
        audio_queue=audio_queue,  # pyright: ignore[reportUnknownArgumentType]
    )

    async def c(transcript: str):
        print("Transcription:", transcript)
        response = await llm_client.get_response(
            {"role": "user", "content": transcript}
        )
        print("LLM Response:", response)

    try:
        await stt_client.start_listening(
            callback=c,
        )
    except KeyboardInterrupt:
        pass

    finally:
        shutdown_event.set()
        await cleanup()


if __name__ == "__main__":
    try:
        asyncio.run(main())

    except KeyboardInterrupt:
        print("Exiting...")
