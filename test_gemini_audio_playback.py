import sys
import asyncio
import sounddevice as sd
import numpy as np
import wave
import pygame
from google import genai
from google.genai import types

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding='utf-8')

with open("gemini_api_key.txt", "r", encoding="utf-8") as f:
    api_key = f.read().strip()

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
    system_instruction="You are an energetic, friendly Indian phone calling assistant. Speak in natural conversational Hinglish with high enthusiasm and warmth. Keep it short."
)

async def test_live_save():
    audio_chunks = []
    
    print("Connecting to Gemini Live Native Audio...")
    async with client.aio.live.connect(model="gemini-2.5-flash-native-audio-latest", config=config) as session:
        prompt = "हेलो, क्या मुझे एसके फर्नीचर के बारे में बता सकते हैं?"
        print(f"Sending prompt: '{prompt}'...")
        await session.send(input=prompt, end_of_turn=True)
        
        async for response in session.receive():
            server_content = response.server_content
            if server_content is not None:
                model_turn = server_content.model_turn
                if model_turn is not None:
                    for part in model_turn.parts:
                        if part.inline_data and part.inline_data.data:
                            audio_chunks.append(part.inline_data.data)
                if server_content.turn_complete:
                    break

    full_pcm = b"".join(audio_chunks)
    print(f"Received {len(full_pcm)} bytes of 24kHz PCM audio!")
    
    out_file = "test_gemini_native_audio.wav"
    with wave.open(out_file, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(24000)
        wf.writeframes(full_pcm)
        
    print(f"Saved to {out_file}! Playing audio via pygame...")
    pygame.mixer.init(frequency=24000)
    pygame.mixer.music.load(out_file)
    pygame.mixer.music.play()
    while pygame.mixer.music.get_busy():
        pygame.time.Clock().tick(10)
    pygame.mixer.quit()
    print("Gemini Native Audio playback complete!")

if __name__ == "__main__":
    asyncio.run(test_live_save())
