from pydub import AudioSegment
import os

INPUT_FOLDER = "../audio/"
OUTPUT_FOLDER = "../audio_converted/"

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

for filename in os.listdir(INPUT_FOLDER):
    if filename.endswith(".wav") or filename.endswith(".mp3"):
        print(f"Conversion de {filename}...")
        sound = AudioSegment.from_file(os.path.join(INPUT_FOLDER, filename))
        sound = sound.set_channels(1).set_frame_rate(16000)
        out_path = os.path.join(OUTPUT_FOLDER, filename.replace(".mp3", ".wav"))
        sound.export(out_path, format="wav")

print("Conversion terminée.")
