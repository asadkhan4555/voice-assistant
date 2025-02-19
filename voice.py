import torch
import soundfile as sf
import numpy as np
import sounddevice as sd
import pyttsx3
import openai
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq
from dotenv import load_dotenv
import os

print("Loading...")

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

model_id = "openai/whisper-base"
device = "cuda" if torch.cuda.is_available() else "cpu"
processor = AutoProcessor.from_pretrained(model_id)
model = AutoModelForSpeechSeq2Seq.from_pretrained(model_id).to(device)

engine = pyttsx3.init()
engine.setProperty("rate", 150)
engine.setProperty("volume", 1.0)

def record_audio(file_path="input.wav", duration=5, samplerate=16000):
    print("Recording... Speak now!")
    audio = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=1, dtype=np.float32)
    sd.wait()
    sf.write(file_path, audio, samplerate)

def process_audio(file_path):
    audio_data, sample_rate = sf.read(file_path, dtype="float32")
    if len(audio_data.shape) > 1:
        audio_data = np.mean(audio_data, axis=1)
    return processor(audio_data, sampling_rate=sample_rate, return_tensors="pt")

def transcribe_audio(file_path):
    audio_inputs = process_audio(file_path)
    audio_inputs = {k: v.to(device) for k, v in audio_inputs.items()}
    with torch.no_grad():
        predicted_ids = model.generate(**audio_inputs, task="transcribe")
    return processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]

def text_to_speech(text):
    engine.say(text)
    engine.runAndWait()

def get_gpt_response_streaming(text):
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "user", "content": text}],
        max_tokens=150,
        temperature=0.7,
        stream=True,
    )
    full_response = ""
    for chunk in response:
        if chunk.get("choices"):
            token = chunk["choices"][0].get("delta", {}).get("content", "")
            if token:
                full_response += token
                print(token, end="", flush=True)
    print()
    return full_response

if __name__ == "__main__":
    while True:
        audio_file = "input.wav"
        record_audio(audio_file, duration=5)
        transcribed_text = transcribe_audio(audio_file)
        gpt_response = get_gpt_response_streaming(transcribed_text)
        text_to_speech(gpt_response)