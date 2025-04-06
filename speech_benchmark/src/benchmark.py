import os
import time
import wave
import json
import jiwer
import pandas as pd
import torch
import librosa

#API de transcription audio
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
from vosk import Model, KaldiRecognizer
import whisper

AUDIO_DIR = "../audio_converted/"
REF_DIR = "../references/"

VOSK_MODEL_PATH = "../models/vosk-model-small-fr-0.22"

WHISPER_MODEL = whisper.load_model("medium")

WAV2VEC_MODEL = Wav2Vec2ForCTC.from_pretrained("jonatasgrosman/wav2vec2-large-xlsr-53-french")
WAV2VEC_PROCESSOR = Wav2Vec2Processor.from_pretrained("jonatasgrosman/wav2vec2-large-xlsr-53-french")

results = []

def load_reference(file_name):
    path = os.path.join(REF_DIR, file_name.replace(".wav", ".txt"))
    with open(path, "r", encoding="utf-8") as f:
        return f.read().strip()

def whisper_transcribe(file_path):
    start = time.time()
    result = WHISPER_MODEL.transcribe(file_path, language="fr")
    return result["text"].strip(), time.time() - start

def vosk_transcribe(file_path):
    wf = wave.open(file_path, "rb")
    if wf.getnchannels() != 1 or wf.getsampwidth() != 2:
        raise ValueError("Le fichier doit être en WAV PCM 16-bit mono.")
    model = Model(VOSK_MODEL_PATH)
    rec = KaldiRecognizer(model, wf.getframerate())
    final_result = ""
    start = time.time()
    while True:
        data = wf.readframes(4000)
        if len(data) == 0:
            break
        if rec.AcceptWaveform(data):
            part = json.loads(rec.Result())
            final_result += part.get("text", "") + " "
    part = json.loads(rec.FinalResult())
    final_result += part.get("text", "")
    return final_result.strip(), time.time() - start

def wav2vec_transcribe(file_path):
    start = time.time()
    speech, rate = librosa.load(file_path, sr=16000)
    input_values = WAV2VEC_PROCESSOR(speech, return_tensors="pt", sampling_rate=16000).input_values
    with torch.no_grad():
        logits = WAV2VEC_MODEL(input_values).logits
    ids = torch.argmax(logits, dim=-1)
    transcription = WAV2VEC_PROCESSOR.batch_decode(ids)[0]
    return transcription.lower().strip(), time.time() - start

for file in sorted(os.listdir(AUDIO_DIR)):
    if not file.endswith(".wav"):
        continue

    print(f"Traitement de {file}...")
    path = os.path.join(AUDIO_DIR, file)
    reference = load_reference(file)

    print("  → Whisper")
    hypo_w, time_w = whisper_transcribe(path)
    results.append({
        "Fichier": file,
        "IA": "Whisper",
        "Texte_IA": hypo_w,
        "Référence": reference,
        "WER": round(jiwer.wer(reference, hypo_w), 3),
        "CER": round(jiwer.cer(reference, hypo_w), 3),
        "Temps": round(time_w, 2)
    })

    print("  → Vosk")
    hypo_v, time_v = vosk_transcribe(path)
    results.append({
        "Fichier": file,
        "IA": "Vosk",
        "Texte_IA": hypo_v,
        "Référence": reference,
        "WER": round(jiwer.wer(reference, hypo_v), 3),
        "CER": round(jiwer.cer(reference, hypo_v), 3),
        "Temps": round(time_v, 2)
    })

    print("  → Wav2Vec2")
    hypo_w2v, time_w2v = wav2vec_transcribe(path)
    results.append({
        "Fichier": file,
        "IA": "Wav2Vec2",
        "Texte_IA": hypo_w2v,
        "Référence": reference,
        "WER": round(jiwer.wer(reference, hypo_w2v), 3),
        "CER": round(jiwer.cer(reference, hypo_w2v), 3),
        "Temps": round(time_w2v, 2)
    })

df = pd.DataFrame(results)
df.to_csv("../output/benchmark_open_results.csv", index=False)
print("Benchmark terminé. Résultats enregistrés dans output/benchmark_open_results.csv")
