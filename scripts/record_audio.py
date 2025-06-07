import sounddevice as sd
from scipy.io.wavfile import write
import os
import time


def record_audio(confidence_level, i):
    fs = 44100  # Sample rate
    seconds = 5  # Duration of recording

    filename = confidence_level
    out_dir = f"data/raw/{filename}"
    os.makedirs(out_dir, exist_ok=True)

    print("Recording...")
    recording = sd.rec(int(seconds * fs), samplerate=fs, channels=1)
    sd.wait()  # Wait until recording is finished

    file_path = f"{out_dir}/{filename}_{i}.wav"
    write(file_path, fs, recording)  # Save as WAV file
    print("Saved: ", file_path)


if __name__ == "__main__":
    i = 0
    while True:
        record_audio('confident', i)
        time.sleep(0.25)
        i+=1

