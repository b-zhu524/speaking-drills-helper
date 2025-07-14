from pydub import AudioSegment
import os


def segment_audio(input_file, segment_length_ms, output_dir):    # Load the audio file
    audio = AudioSegment.from_file(input_file)
    audio_length_ms = len(audio)

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Segment the audio file
    for i in range(0, audio_length_ms, segment_length_ms):
        segment = audio[i:i + segment_length_ms]

        segment_number = file_count = len([name for name in os.listdir(output_dir) if os.path.isfile(os.path.join(output_dir, name))]) if os.path.exists(output_dir) else 0
        segment_filename = os.path.join(output_dir, f"segment_{segment_number}.wav")

        segment.export(segment_filename, format="wav")
        print(f"Exported {segment_filename}")


def load_data():
    input_dir_clear = "data/raw-long-files/clear"
    input_dir_unclear = "data/raw-long-files/unclear"

    for file in os.listdir(input_dir_clear):
        if file.endswith(".wav"):
            input_file = os.path.join(input_dir_clear, file)
            segment_audio(input_file, 5000, "data/raw/clear-raw")
    for file in os.listdir(input_dir_unclear):
        if file.endswith(".wav"):
            input_file = os.path.join(input_dir_unclear, file)
            segment_audio(input_file, 5000, "data/raw/unclear-raw")


def convert_m4a_to_wav(input_file, output_file):
    audio = AudioSegment.from_file(input_file, format="m4a")
    audio.export(output_file, format="wav")


def convert_all_files():
    input_dir_clear = "data/raw-long-files/clear"
    input_dir_unclear = "data/raw-long-files/unclear"

    for file in os.listdir(input_dir_clear):
        if file.endswith(".m4a"):
            convert_m4a_to_wav(os.path.join(input_dir_clear, file), os.path.join(input_dir_clear, file.replace(".m4a", ".wav")))
            os.remove(os.path.join(input_dir_clear, file))
    
    for file in os.listdir(input_dir_unclear):
        if file.endswith(".m4a"):
            convert_m4a_to_wav(os.path.join(input_dir_unclear, file), os.path.join(input_dir_unclear, file.replace(".m4a", ".wav")))
            os.remove(os.path.join(input_dir_unclear, file))


if __name__ == "__main__":
    convert_all_files()
    load_data()
