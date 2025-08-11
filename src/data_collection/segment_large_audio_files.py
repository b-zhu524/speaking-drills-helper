from pydub import AudioSegment
import os
import torchaudio


def segment_audio(input_file, segment_length_ms, output_dir):    # Load the audio file
    audio = AudioSegment.from_file(input_file)
    audio_length_ms = len(audio)

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Segment the audio file
    for i in range(0, audio_length_ms, segment_length_ms):
        segment = audio[i:i + segment_length_ms]

        segment_number = file_count = len([name for name in os.listdir(output_dir) if os.path.isfile(os.path.join(output_dir, name))]) if os.path.exists(output_dir) else 0
        file_name = os.path.basename(input_file)
        segment_filename = os.path.join(output_dir, f"segment_{segment_number}-{file_name}.wav")

        segment.export(segment_filename, format="wav")
        print(f"Exported {segment_filename}")


def load_data(input_dir, output_dir):
    MIN_DURATION = 5    # 5 seconds

    for file in os.listdir(input_dir):
        file_path = os.path.join(input_dir, file)
        if os.path.isdir(file_path):
            continue

        if not file.endswith(".wav"):
            convert_m4a_to_wav(os.path.join(input_dir, file), os.path.join(input_dir, file.replace(".m4a", ".wav")))
                # Check duration before segmenting
        try:
            info = torchaudio.info(file_path)
            duration_sec = info.num_frames / info.sample_rate
            if duration_sec < MIN_DURATION:
                print(f"Skipping {file} — too short ({duration_sec:.2f} sec)")
                continue
        except Exception as e:
            print(f"Error reading {file}: {e}")
            continue


        segment_audio(file_path, 5000, output_dir)
 

def convert_m4a_to_wav(input_file, output_file):
    audio = AudioSegment.from_file(input_file, format="m4a")
    audio.export(output_file, format="wav")


def convert_all_files():
    input_dir_clear = "data/raw-long-files/training/clear"
    input_dir_unclear = "data/raw-long-files/training/unclear"

    for file in os.listdir(input_dir_clear):
        if file.endswith(".m4a"):
            convert_m4a_to_wav(os.path.join(input_dir_clear, file), os.path.join(input_dir_clear, file.replace(".m4a", ".wav")))
            os.remove(os.path.join(input_dir_clear, file))
    
    for file in os.listdir(input_dir_unclear):
        if file.endswith(".m4a"):
            convert_m4a_to_wav(os.path.join(input_dir_unclear, file), os.path.join(input_dir_unclear, file.replace(".m4a", ".wav")))
            os.remove(os.path.join(input_dir_unclear, file))


if __name__ == "__main__":
    load_data("data/raw-long-files/training/clear", "data/raw/clear-raw")
    load_data("data/raw-long-files/training/unclear", "data/raw/unclear-raw")

