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

    config = types.LiveConnectConfig(
        response_modalities=['AUDIO'],
        speech_config=types.SpeechConfig(
            voice_config=types.VoiceConfig(
                prebuilt_voice_config=types.PrebuiltVoiceConfig(
                    voice_name=voice_name
                )
            )
        ),
        system_instruction="You are an energetic, warm Indian phone calling assistant. Speak in natural conversational Hinglish. Keep it short (1-2 sentences)."
    )

    out_stream = sd.RawOutputStream(samplerate=24000, channels=1, dtype='int16')
    out_stream.start()

    turns = [
        "हेलो, मैं कार लोन के बारे में जानना चाहता हूँ।",
        "कितना इंटरेस्ट रेट लगेगा?",
        "धन्यवाद, मैं कल कॉल करूँगा।"
    ]

    print("=== Testing Single Persistent Live Session with Real-Time Chunk Streaming ===")
    print("Opening persistent connection once...")
    
    t_conn = time.perf_counter()
    async with client.aio.live.connect(model="gemini-2.5-flash-native-audio-latest", config=config) as session:
        conn_time = (time.perf_counter() - t_conn) * 1000
        print(f"✅ Persistent WebSocket connected in {conn_time:.1f} ms!\n")

        for idx, user_prompt in enumerate(turns, 1):
            print(f"--- Turn {idx} ---")
            print(f"User: {user_prompt}")
            
            t0 = time.perf_counter()
            first_chunk = False
            total_bytes = 0
            
            await session.send(input=user_prompt, end_of_turn=True)
            
            async for response in session.receive():
                server_content = response.server_content
                if server_content is not None:
                    model_turn = server_content.model_turn
                    if model_turn is not None:
                        for part in model_turn.parts:
                            if part.inline_data and part.inline_data.data:
                                chunk = part.inline_data.data
                                total_bytes += len(chunk)
                                if not first_chunk:
                                    ttfa = (time.perf_counter() - t0) * 1000
                                    print(f"⚡ Time to First Audio (TTFA): {ttfa:.1f} ms")
                                    first_chunk = True
                                out_stream.write(chunk)
                    if server_content.turn_complete:
                        break
            
            turn_total = (time.perf_counter() - t0) * 1000
            print(f"Turn {idx} completed ({total_bytes} bytes in {turn_total:.1f} ms)\n")
            await asyncio.sleep(1.0)  # Pause between turns

    out_stream.stop()
    out_stream.close()
    print("All turns completed successfully!")

if __name__ == "__main__":
    asyncio.run(main())
