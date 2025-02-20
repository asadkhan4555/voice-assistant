import os
import time
import torch
import soundfile as sf
import numpy as np
import sounddevice as sd
import pyttsx3
import openai
import logging
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq
from dotenv import load_dotenv

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

if not openai_api_key:
    logging.error("OpenAI API key not found! Please set it in your environment variables.")
    exit(1)

openai.api_key = openai_api_key
device = "cuda" if torch.cuda.is_available() else "cpu"
logging.info(f"Using device: {device}")
try:
    model_id = "openai/whisper-base"
    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForSpeechSeq2Seq.from_pretrained(model_id).to(device)
    logging.info("Whisper model loaded successfully.")
except Exception as e:
    logging.error(f"Error loading Whisper model: {e}")
    exit(1)
try:
    engine = pyttsx3.init()
    engine.setProperty("rate", 150)  
    engine.setProperty("volume", 1.0) 
    logging.info("Text-to-Speech engine initialized.")
except Exception as e:
    logging.error(f"Error initializing TTS engine: {e}")
    exit(1)

def record_audio(file_path="input.wav", duration=5, samplerate=16000):
    """Records user speech and saves it as a WAV file."""
    try:
        logging.info("Recording... Speak now!")
        audio = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=1, dtype=np.float32)
        sd.wait()
        sf.write(file_path, audio, samplerate)
        logging.info("Recording completed.")
    except Exception as e:
        logging.error(f"Error during recording: {e}")

def process_audio(file_path):
    """Processes audio for transcription by converting stereo to mono and preparing tensors."""
    try:
        audio_data, sample_rate = sf.read(file_path, dtype="float32")
        if len(audio_data.shape) > 1:
            audio_data = np.mean(audio_data, axis=1)
        return processor(audio_data, sampling_rate=sample_rate, return_tensors="pt")
    except Exception as e:
        logging.error(f"Error processing audio file: {e}")
        return None

def transcribe_audio(file_path):
    """Transcribes speech to text using Whisper."""
    try:
        audio_inputs = process_audio(file_path)
        if audio_inputs is None:
            return ""

        audio_inputs = {k: v.to(device) for k, v in audio_inputs.items()}
        with torch.no_grad():
            predicted_ids = model.generate(**audio_inputs, task="transcribe")
        transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
        logging.info(f"Transcription: {transcription}")
        return transcription
    except Exception as e:
        logging.error(f"Error transcribing audio: {e}")
        return ""

def text_to_speech_streaming(text):
    """Streams text to speech output using pyttsx3."""
    try:
        engine.say(text)
        engine.runAndWait()
    except Exception as e:
        logging.error(f"Error in TTS: {e}")

def get_gpt_response_streaming(user_input):
    """Streams GPT response and sends chunks to TTS in real-time."""
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[{"role": "user", "content": user_input}],
            max_tokens=150,
            temperature=0.7,
            stream=True,
        )

        buffer = ""
        for chunk in response:
            if chunk.get("choices"):
                token = chunk["choices"][0].get("delta", {}).get("content", "")
                if token:
                    print(token, end="", flush=True)
                    buffer += token

                    if any(punct in buffer for punct in [".", "?", "!"]):
                        text_to_speech_streaming(buffer.strip())
                        buffer = "" 
        if buffer:
            text_to_speech_streaming(buffer.strip())

    except Exception as e:
        logging.error(f"Error in GPT response streaming: {e}")

def main():
    """Main loop for recording, transcribing, generating GPT responses, and speaking."""
    try:
        while True:
            audio_file = "input.wav"
            record_audio(audio_file, duration=5)
            transcribed_text = transcribe_audio(audio_file)

            if transcribed_text:
                get_gpt_response_streaming(transcribed_text)
            else:
                logging.warning("No transcription detected. Skipping GPT request.")

    except KeyboardInterrupt:
        logging.info("User interrupted execution. Exiting gracefully...")
        exit(0)
    except Exception as e:
        logging.error(f"Unexpected error in main loop: {e}")
        exit(1)

if __name__ == "__main__":
    main()