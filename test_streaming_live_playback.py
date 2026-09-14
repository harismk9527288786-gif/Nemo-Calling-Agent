import sys
import asyncio
import time
import sounddevice as sd
from google import genai
from google.genai import types

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding='utf-8')

async def main():
    with open("gemini_api_key.txt", "r", encoding="utf-8") as f:
        api_key = f.read().strip()

    client = genai.Client(api_key=api_key)
    voice_name = "Puck"
    prompt = "अरे हाय सर! बिल्कुल, मैं आपकी तुरंत सहायता करूँगा। बताइए आप किस बारे में जानना चाहते हैं?"

    print("=== Testing Real-Time Audio Chunk Streaming (Zero WAV Buffering) ===")
    print(f"Prompt: {prompt}")

    config = types.LiveConnectConfig(
        response_modalities=['AUDIO'],
        speech_config=types.SpeechConfig(
            voice_config=types.VoiceConfig(
                prebuilt_voice_config=types.PrebuiltVoiceConfig(
                    voice_name=voice_name
                )
            )
        ),
        system_instruction="You are an energetic, warm, and highly motivated Indian calling assistant. Speak in natural conversational Hinglish."
    )

    # 24kHz, 1-channel, 16-bit signed PCM audio output stream
    out_stream = sd.RawOutputStream(samplerate=24000, channels=1, dtype='int16')
    out_stream.start()

    print("\nConnecting to Gemini Live...")
    t0 = time.perf_counter()
    first_chunk_played = False
    total_bytes = 0

    try:
        async with client.aio.live.connect(model="gemini-2.5-flash-native-audio-latest", config=config) as session:
            await session.send(input=prompt, end_of_turn=True)
            print("Streaming audio chunks directly to speakers...")
            
            async for response in session.receive():
                server_content = response.server_content
                if server_content is not None:
                    model_turn = server_content.model_turn
                    if model_turn is not None:
                        for part in model_turn.parts:
                            if part.inline_data and part.inline_data.data:
                                chunk = part.inline_data.data
                                total_bytes += len(chunk)
                                if not first_chunk_played:
                                    ttfa = (time.perf_counter() - t0) * 1000
                                    print(f"\n⚡ Time to First Audio (TTFA): {ttfa:.1f} ms (Instant Playback Started!)")
                                    first_chunk_played = True
                                
                                # Write raw PCM chunk directly to sound device
                                out_stream.write(chunk)
                    
                    if server_content.turn_complete:
                        break

    finally:
        # Wait a tiny bit for the last buffer to finish playing
        await asyncio.sleep(0.3)
        out_stream.stop()
        out_stream.close()
        print(f"\nStream completed! Played {total_bytes} bytes in real-time.")

if __name__ == "__main__":
    asyncio.run(main())
