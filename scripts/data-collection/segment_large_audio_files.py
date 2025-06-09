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
    input_dir_confident = "data/raw-long-files/confident"
    input_dir_unconfident = "data/raw-long-files/not_confident"

    for file in os.listdir(input_dir_confident):
        if file.endswith(".wav"):
            input_file = os.path.join(input_dir_confident, file)
            segment_audio(input_file, 5000, "data/raw/confident")
    for file in os.listdir(input_dir_unconfident):
        if file.endswith(".wav"):
            input_file = os.path.join(input_dir_unconfident, file)
            segment_audio(input_file, 5000, "data/raw/not_confident")


def convert_m4a_to_wav(input_file, output_file):
    audio = AudioSegment.from_file(input_file, format="m4a")
    audio.export(output_file, format="wav")


def convert_all_files():
    input_dir_confident = "data/raw-long-files/confident"
    input_dir_unconfident = "data/raw-long-files/not_confident"

    for file in os.listdir(input_dir_confident):
        if file.endswith(".m4a"):
            convert_m4a_to_wav(os.path.join(input_dir_confident, file), os.path.join("data/raw/confident", file.replace(".m4a", ".wav")))
            os.remove(os.path.join(input_dir_confident, file))
    
    for file in os.listdir(input_dir_unconfident):
        if file.endswith(".m4a"):
            convert_m4a_to_wav(os.path.join(input_dir_unconfident, file), os.path.join("data/raw/not_confident", file.replace(".m4a", ".wav")))
            os.remove(os.path.join(input_dir_unconfident, file))


if __name__ == "__main__":
    convert_all_files()
    load_data()
