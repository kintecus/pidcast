import os
import subprocess
import sys
import datetime
import re

# Configuration
YT_DLP_PATH = "yt-dlp"  # Ensure yt-dlp is installed and in PATH
FFMPEG_PATH = "ffmpeg"  # Ensure ffmpeg is installed and in PATH
WHISPER_CPP_PATH = "/Users/ostaps/Code/whisper.cpp/build/bin/whisper-cli"  # Adjust to your whisper.cpp binary path
WHISPER_MODEL = "/Users/ostaps/Code/whisper.cpp/models/ggml-base.en.bin"

# Function to sanitize filenames
def sanitize_filename(filename):
    # Convert to lowercase
    filename = filename.lower()
    # Replace non-alphanumeric characters with a space
    filename = re.sub(r"[^\w\s-]", "", filename)
    # Replace spaces and underscores with hyphens
    filename = re.sub(r"[\s_]+", "-", filename)
    return filename.strip("-")

# Function to get current date as YYYY-MM-DD
def get_date():
    return datetime.datetime.now().strftime("%Y-%m-%d")

# Function to download and extract audio
def download_audio(url):
    print("[INFO] Downloading and extracting audio...")
    
    # yt-dlp command
    yt_dlp_cmd = [
        YT_DLP_PATH,
        "-f", "bestaudio",
        "--extract-audio",
        "--audio-format", "wav",
        "--postprocessor-args", "-ar 16000 -ac 1 -c:a pcm_s16le",
        "-o", "%(title)s.%(ext)s",
        url
    ]
    
    result = subprocess.run(yt_dlp_cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print("[ERROR] yt-dlp failed:", result.stderr)
        sys.exit(1)

    # Extract filename from output
    output = result.stdout + result.stderr
    match = re.search(r"Destination: (.+\.wav)", output)
    
    if match:
        filename = match.group(1)
        print(f"[INFO] Audio saved as {filename}")
        return filename
    else:
        print("[ERROR] Could not determine output filename.")
        sys.exit(1)

# Function to rename file with clean format
def rename_file(original_filename):
    base_name = os.path.splitext(original_filename)[0]
    sanitized_title = sanitize_filename(base_name)
    new_filename = f"{get_date()}-{sanitized_title}.wav"
    os.rename(original_filename, new_filename)
    print(f"[INFO] Renamed to {new_filename}")
    return new_filename

# Function to transcribe audio using whisper.cpp
def transcribe_audio(audio_file):
    print("[INFO] Transcribing audio with Whisper...")

    whisper_cmd = [
        "/Users/ostaps/Code/whisper.cpp/build/bin/whisper-cli",
        "-m", "/Users/ostaps/Code/whisper.cpp/models/ggml-base.en.bin",
        "-f", audio_file
    ]

    result = subprocess.run(whisper_cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print("[ERROR] Whisper transcription failed:", result.stderr)
        return None

    transcript_text = result.stdout  # Capture Whisper's printed output

    if not transcript_text.strip():
        print("[ERROR] Whisper did not generate any output.")
        return None

    # Save transcript to a Markdown file
    transcript_file = audio_file.replace(".wav", "-transcript.md")
    with open(transcript_file, "w", encoding="utf-8") as f:
        f.write(transcript_text)

    print(f"[INFO] Transcript saved as {transcript_file}")
    return transcript_file

# main function
def main():
    if len(sys.argv) < 2:
        print("Usage: python yt_transcribe.py <YouTube-URL>")
        sys.exit(1)

    url = sys.argv[1]
    audio_file = download_audio(url)
    renamed_audio = rename_file(audio_file)
    transcribe_audio(renamed_audio)

if __name__ == "__main__":
    main()
