import os
import subprocess
import sys
import datetime
import re
import time
import json

# Configuration
YT_DLP_PATH = "yt-dlp"  # Ensure yt-dlp is installed and in PATH
FFMPEG_PATH = "ffmpeg"  # Ensure ffmpeg is installed and in PATH
WHISPER_CPP_PATH = "/Users/ostaps/Code/whisper.cpp/build/bin/whisper-cli"  # Adjust to your whisper.cpp binary path
WHISPER_MODEL = "/Users/ostaps/Code/whisper.cpp/models/ggml-base.en.bin"
OBSIDIAN_PATH = "/Users/ostaps/Library/Mobile Documents/iCloud~md~obsidian/Documents/Obsidian Vault/03 - RESOURCES/Podcasts"

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
    
    # Create a temporary file for progress output
    progress_file = 'download_progress.txt'
    
    # yt-dlp command with progress output
    yt_dlp_cmd = [
        YT_DLP_PATH,
        "-f", "bestaudio",
        "--extract-audio",
        "--audio-format", "wav",
        "--postprocessor-args", "-ar 16000 -ac 1 -c:a pcm_s16le",
        "-o", "%(title)s.%(ext)s",
        "--newline",  # Ensure new lines in progress output
        "--progress-template", "%(progress._percent_str)s",
        url
    ]
    
    process = subprocess.Popen(
        yt_dlp_cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True
    )
    
    # Print progress while downloading
    while True:
        output = process.stdout.readline()
        if output == '' and process.poll() is not None:
            break
        if output:
            # Clear line and print progress
            sys.stdout.write('\r' + ' ' * 50 + '\r')  # Clear previous line
            sys.stdout.write(f"Downloading: {output.strip()}")
            sys.stdout.flush()
    
    # Get final output
    _, stderr = process.communicate()
    
    if process.returncode != 0:
        print("\n[ERROR] yt-dlp failed:", stderr)
        sys.exit(1)

    # Extract filename from stderr
    match = re.search(r"Destination: (.+\.wav)", stderr)
    
    if match:
        filename = match.group(1)
        print(f"\n[INFO] Audio saved as {filename}")
        return filename
    else:
        print("\n[ERROR] Could not determine output filename.")
        sys.exit(1)

# Function to rename file with clean format
def rename_file(original_filename):
    base_name = os.path.splitext(original_filename)[0]
    sanitized_title = sanitize_filename(base_name)
    new_filename = f"{get_date()}-{sanitized_title}.wav"
    os.rename(original_filename, new_filename)
    print(f"[INFO] Renamed to {new_filename}")
    return new_filename

def get_audio_duration(audio_file):
    """Get duration of audio file in seconds using ffprobe."""
    cmd = [
        'ffprobe',
        '-v', 'quiet',
        '-print_format', 'json',
        '-show_format',
        '-show_streams',
        audio_file
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print("[ERROR] Failed to get audio duration")
        return None
        
    data = json.loads(result.stdout)
    duration = float(data['format']['duration'])
    return duration

def count_words(text):
    """Count words in text."""
    return len(text.split())

def print_summary(audio_file, duration, transcription_time, word_count):
    """Print processing summary."""
    ratio = duration / transcription_time
    print("\n=== Processing Summary ===")
    print(f"Audio duration: {duration:.2f} seconds")
    print(f"Transcription time: {transcription_time:.2f} seconds")
    print(f"Processing ratio: {ratio:.2f}x realtime")
    print(f"Word count: {word_count}")
    print(f"Words per minute: {(word_count / (duration/60)):.1f}")
    print("=======================\n")

# Function to transcribe audio using whisper.cpp
def transcribe_audio(audio_file):
    print("[INFO] Transcribing audio with Whisper...")

    # Get audio duration before transcription
    audio_duration = get_audio_duration(audio_file)
    
    # Time the transcription process
    start_time = time.time()
    
    whisper_cmd = [
        WHISPER_CPP_PATH,
        "-m", WHISPER_MODEL,
        "-f", audio_file,
        "-t", "8"
    ]
    
    result = subprocess.run(whisper_cmd, capture_output=True, text=True)
    
    transcription_time = time.time() - start_time
    
    if result.returncode != 0:
        print("[ERROR] Whisper transcription failed:", result.stderr)
        return None

    transcript_text = result.stdout

    if not transcript_text.strip():
        print("[ERROR] Whisper did not generate any output.")
        return None

    # Save transcript to a Markdown file in OBSIDIAN_PATH
    base_name = os.path.splitext(audio_file)[0]
    transcript_file = os.path.join(OBSIDIAN_PATH, f"{base_name}-transcript.md")
    with open(transcript_file, "w", encoding="utf-8") as f:
        f.write(transcript_text)

    # Generate and print summary
    word_count = count_words(transcript_text)
    print_summary(audio_file, audio_duration, transcription_time, word_count)

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
