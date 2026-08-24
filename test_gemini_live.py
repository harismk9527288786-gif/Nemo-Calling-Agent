import asyncio
import os
import io
import sounddevice as sd
import numpy as np
from google import genai
from google.genai import types

with open("gemini_api_key.txt", "r", encoding="utf-8") as f:
    api_key = f.read().strip()

print(f"Connecting to Gemini Live Multimodal Audio API with key (prefix: {api_key[:10]}...)...")

client = genai.Client(api_key=api_key)

config = types.LiveConnectConfig(
    response_modalities=['AUDIO'],
    enable_affective_dialog=True,
    speech_config=types.SpeechConfig(
        voice_config=types.VoiceConfig(
            prebuilt_voice_config=types.PrebuiltVoiceConfig(
                voice_name="Puck"  # Puck, Charon, Kore, Fenrir, Aoede
            )
        )
    ),
    system_instruction="You are an energetic, warm, and highly motivated Indian phone calling assistant. Speak in natural conversational Hinglish with genuine enthusiasm and emotion. Keep responses short (1-2 sentences)."
)

async def test_live():
    output_stream = sd.OutputStream(samplerate=24000, channels=1, dtype='int16')
    output_stream.start()

    model_id = "gemini-2.5-flash-native-audio-latest"
    print(f"Opening Live WebSocket session with model: {model_id}...")
    
    try:
        async with client.aio.live.connect(model=model_id, config=config) as session:
            print("Connected to Gemini Live Multimodal Audio WebSocket!")
            
            prompt = "Hi! Main Rahul baat kar raha hoon, kya aap meri help kar sakte hain?"
            print(f"\nSending User Prompt: '{prompt}'...")
            
            await session.send(input=prompt, end_of_turn=True)
            print("Receiving Native Live Audio stream from Gemini...")

            async for response in session.receive():
                server_content = response.server_content
                if server_content is not None:
                    model_turn = server_content.model_turn
                    if model_turn is not None:
                        for part in model_turn.parts:
                            if part.text:
                                print(f"[Gemini Text]: {part.text}")
                            if part.inline_data:
                                pcm_data = part.inline_data.data
                                audio_array = np.frombuffer(pcm_data, dtype=np.int16)
                                output_stream.write(audio_array)
                                
                    if server_content.turn_complete:
                        print("\n[Turn Complete!]")
                        break

    except Exception as e:
        print(f"\n[Live API Error]: {e}")
    finally:
        output_stream.stop()
        output_stream.close()

if __name__ == "__main__":
    asyncio.run(test_live())
