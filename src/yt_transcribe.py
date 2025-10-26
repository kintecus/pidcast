import argparse
import os
import json
import time
import datetime
import uuid
import re
import yt_dlp
import subprocess

# Configuration
YT_DLP_PATH = "yt-dlp"  # Ensure yt-dlp is installed and in PATH
FFMPEG_PATH = "ffmpeg"  # Ensure ffmpeg is installed and in PATH
WHISPER_CPP_PATH = "/Users/ostaps/Code/whisper.cpp/build/bin/whisper-cli"  # Adjust to your whisper.cpp binary path
WHISPER_MODEL = "/Users/ostaps/Code/whisper.cpp/models/ggml-base.en.bin"
OBSIDIAN_PATH = "/Users/ostaps/Library/Mobile Documents/iCloud~md~obsidian/Documents/Obsidian Vault/03 - RESOURCES/Podcasts"

def sanitize_filename(filename):
    """Sanitize filename to remove invalid characters."""
    return re.sub(r'[^\w\s-]', '', filename).strip()

def ensure_wav_file(input_file):
    """Ensure the audio file exists and is in WAV format."""
    if not os.path.exists(input_file):
        # Check if we have a webm file instead
        webm_file = input_file.replace('.wav', '.webm')
        if os.path.exists(webm_file):
            # Convert webm to wav
            command = [
                FFMPEG_PATH,
                '-i', webm_file,
                '-ar', '16000',
                '-ac', '1',
                '-c:a', 'pcm_s16le',
                input_file
            ]
            subprocess.run(command, check=True)
            os.remove(webm_file)  # Clean up the webm file
    return os.path.exists(input_file)

def run_whisper_transcription(audio_file, whisper_model, output_format, output_file):
    """Runs whisper transcription."""
    # Build base command
    command = [
        WHISPER_CPP_PATH,
        "-f", audio_file,
        "-m", whisper_model,
        "-t", "8"
    ]
    
    # Add output format flag
    if output_format == "txt" or output_format == "otxt":
        command.append("--output-txt")
    elif output_format == "vtt":
        command.append("--output-vtt")
    elif output_format == "srt":
        command.append("--output-srt")
    elif output_format == "json":
        command.append("--output-json")
    
    # Add output file path
    if output_file:
        command.extend(["-of", output_file])
    
    print("Running command:", " ".join(command))
    subprocess.run(command, check=True)

def main():
    parser = argparse.ArgumentParser(description="Automate YouTube video transcription with Whisper.")
    parser.add_argument("video_url", help="YouTube video URL")
    parser.add_argument("--output_dir", default=".", help="Output directory for Markdown files")
    parser.add_argument("--whisper_model", default=WHISPER_MODEL, help="Whisper model size (tiny, base, small, medium, large)")
    parser.add_argument("--output_format", default="otxt", help="Whisper output format(txt, vtt, srt, tsv, json, all)")
    parser.add_argument("--front_matter", default="{}", help="JSON string for Markdown front matter")
    parser.add_argument("--stats_file", default="transcription_stats.json", help="File to store statistics")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")

    args = parser.parse_args()

    run_uid = str(uuid.uuid4())
    run_timestamp = datetime.datetime.now().isoformat()
    start_time = time.time()

    if args.verbose:
        print(f"Starting transcription for: {args.video_url}")

    # Download video and extract audio using yt_dlp library
    try:
        ydl_opts = {
            'format': 'bestaudio/best',
            'outtmpl': 'temp_audio.%(ext)s',
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'wav',
                'preferredquality': '192',
            }],
            'postprocessor_args': [
                '-ar', '16000',
                '-ac', '1',
                '-c:a', 'pcm_s16le',
            ],
        }

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info_dict = ydl.extract_info(args.video_url, download=True)
            video_title = info_dict.get('title', None)
            sanitized_title = sanitize_filename(video_title)
            audio_file = "temp_audio.wav"

        if args.verbose:
            print("Video downloaded and audio extracted.")

        # Verify audio file exists
        if not ensure_wav_file(audio_file):
            raise FileNotFoundError(f"Could not create or find audio file: {audio_file}")

        # Run Whisper transcription
        whisper_output_file = os.path.join(args.output_dir, sanitized_title)
        output_format = args.output_format.replace("o", "")  # Convert "otxt" to "txt"
        run_whisper_transcription(audio_file, args.whisper_model, output_format, whisper_output_file)

        if args.verbose:
            print("Transcription completed.")

        # Create Markdown file
        markdown_file = os.path.join(args.output_dir, f"{sanitized_title}.md")
        transcript_file = f"{whisper_output_file}.txt"
        if os.path.exists(transcript_file):
            with open(transcript_file, "r") as f:
                transcript = f.read()

        front_matter = json.loads(args.front_matter)
        front_matter_str = "---\n" + json.dumps(front_matter, indent=2) + "\n---\n"

        with open(markdown_file, "w") as f:
            f.write(front_matter_str)
            f.write(transcript)

        if args.verbose:
            print(f"Markdown file created: {markdown_file}")

        # Store statistics
        end_time = time.time()
        duration = end_time - start_time
        stats = {
            "run_uid": run_uid,
            "run_timestamp": run_timestamp,
            "video_title": sanitized_title,
            "video_url": args.video_url,
            "run_duration": duration
        }

        try:
            with open(args.stats_file, "r") as f:
                existing_stats = json.load(f)
        except FileNotFoundError:
            existing_stats = []

        existing_stats.append(stats)

        with open(args.stats_file, "w") as f:
            json.dump(existing_stats, f, indent=2)

        if args.verbose:
            print("Statistics stored.")

    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        # Clean up temporary audio file
        if os.path.exists(audio_file):
            os.remove(audio_file)

if __name__ == "__main__":
    main()
