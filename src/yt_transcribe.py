import argparse
import os
import json
import time
import datetime
import uuid
import re
import yt_dlp
import subprocess
from typing import Optional, Dict, Any

# Configuration
YT_DLP_PATH = "yt-dlp"  # Ensure yt-dlp is installed and in PATH
FFMPEG_PATH = "ffmpeg"  # Ensure ffmpeg is installed and in PATH
WHISPER_CPP_PATH = "/Users/ostaps/Code/whisper.cpp/build/bin/whisper-cli"
WHISPER_MODEL = "/Users/ostaps/Code/whisper.cpp/models/ggml-base.en.bin"
OBSIDIAN_PATH = "/Users/ostaps/Library/Mobile Documents/iCloud~md~obsidian/Documents/Obsidian Vault/03 - RESOURCES/Podcasts"

# Default paths (relative to script directory)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_TRANSCRIPTS_DIR = os.path.join(SCRIPT_DIR, "transcripts")
DEFAULT_STATS_FILE = os.path.join(DEFAULT_TRANSCRIPTS_DIR, "transcription_stats.json")

# Retry configuration
MAX_DOWNLOAD_RETRIES = 3
RETRY_SLEEP_SECONDS = 10


def sanitize_filename(filename: str) -> str:
    """Sanitize filename to remove invalid characters."""
    return re.sub(r'[^\w\s-]', '', filename).strip()


def create_smart_filename(title: str, max_length: int = 60, include_date: bool = True) -> str:
    """Create a smart, shortened filename from video title."""
    filler_patterns = [
        r'^Episode\s+\d+[:\s-]*',
        r'^EP\.?\s*\d+[:\s-]*',
        r'^\d+[:\s-]+',
        r'\s*[-–—]\s*Keynote Speakers?\s*',
        r'\s*[-–—]\s*Interview\s*',
        r'\s*\|\s*',
        r'\s*[-–—]\s*Part\s+\d+',
    ]
    
    cleaned_title = title
    for pattern in filler_patterns:
        cleaned_title = re.sub(pattern, '', cleaned_title, flags=re.IGNORECASE)
    
    cleaned_title = re.sub(r'\s+', ' ', cleaned_title).strip()
    words = cleaned_title.split()
    
    important_words = []
    regular_words = []
    low_priority = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 
                    'of', 'with', 'how', 'what', 'why', 'when', 'where', 'this', 'that',
                    'these', 'those', 'is', 'are', 'was', 'were', 'be', 'been', 'being'}
    
    for word in words:
        word_lower = word.lower()
        if word[0].isupper() or word.isupper() or word_lower not in low_priority:
            important_words.append(word)
        else:
            regular_words.append(word)
    
    result_words = []
    current_length = 0
    
    for word in important_words:
        word_len = len(word) + 1
        if current_length + word_len <= max_length:
            result_words.append(word)
            current_length += word_len
        else:
            break
    
    for word in regular_words:
        word_len = len(word) + 1
        if current_length + word_len <= max_length:
            result_words.append(word)
            current_length += word_len
        else:
            break
    
    filename = '_'.join(result_words)
    filename = re.sub(r'[^\w\s-]', '', filename)
    filename = re.sub(r'[-\s]+', '_', filename)
    
    if include_date:
        date_prefix = datetime.datetime.now().strftime('%Y-%m-%d')
        filename = f"{date_prefix}_{filename}"
    
    return filename


def get_unique_filename(directory: str, base_filename: str, extension: str = '.md') -> str:
    """Get a unique filename by adding version suffix if file exists."""
    filepath = os.path.join(directory, f"{base_filename}{extension}")
    
    if not os.path.exists(filepath):
        return filepath
    
    version = 2
    while True:
        versioned_filename = f"{base_filename}_v{version}{extension}"
        filepath = os.path.join(directory, versioned_filename)
        if not os.path.exists(filepath):
            return filepath
        version += 1


def ensure_wav_file(input_file: str, verbose: bool = False) -> bool:
    """Ensure the audio file exists and is in WAV format."""
    if os.path.exists(input_file):
        return True
    
    # Check if we have a webm file instead
    webm_file = input_file.replace('.wav', '.webm')
    if os.path.exists(webm_file):
        if verbose:
            print(f"Converting {webm_file} to WAV format...")
        try:
            command = [
                FFMPEG_PATH,
                '-i', webm_file,
                '-ar', '16000',
                '-ac', '1',
                '-c:a', 'pcm_s16le',
                input_file
            ]
            subprocess.run(command, check=True, capture_output=True)
            os.remove(webm_file)  # Clean up the webm file
            if verbose:
                print("Conversion successful.")
            return True
        except subprocess.CalledProcessError as e:
            print(f"Error converting audio: {e}")
            return False
    
    return False


def format_duration(seconds: float) -> str:
    """Format duration in seconds to a readable string."""
    if seconds < 60:
        return f"{seconds:.2f} seconds"
    
    minutes = int(seconds // 60)
    rem_seconds = int(seconds % 60)
    return f"{minutes}m {rem_seconds}s"


def download_audio_with_retry(video_url: str, output_template: str, verbose: bool = False, po_token: Optional[str] = None) -> tuple[Optional[str], Optional[Dict[str, Any]]]:
    """
    Download audio from YouTube with retry logic and multiple fallback strategies.
    
    Args:
        video_url: YouTube video URL
        output_template: Output file template
        verbose: Enable verbose output
        po_token: Optional PO Token for bypassing restrictions (format: "client.type+TOKEN")
    
    Returns:
        tuple: (audio_file_path, video_info_dict) or (None, None) on failure
    """
    
    # Strategy 1: Android client (most reliable without PO token)
    # Strategy 2: Web client with aggressive settings
    # Strategy 3: iOS client (requires PO token for some videos)
    strategies = [
        {
            'name': 'Android client (recommended)',
            'opts': {
                'format': 'bestaudio[ext=m4a]/bestaudio[ext=webm]/bestaudio/best',
                'outtmpl': output_template,
                'extractor_args': {'youtube': {'player_client': ['android']}},
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
                'socket_timeout': 45,
                'retries': 15,
                'fragment_retries': 15,
                'http_chunk_size': 10485760,  # 10MB chunks
                'quiet': not verbose,
                'no_warnings': not verbose,
            }
        },
        {
            'name': 'Web client with retry',
            'opts': {
                'format': 'bestaudio[ext=m4a]/bestaudio[ext=webm]/bestaudio/worst',
                'outtmpl': output_template,
                'extractor_args': {'youtube': {'player_client': ['web']}},
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
                'socket_timeout': 60,
                'retries': 20,
                'fragment_retries': 20,
                'http_chunk_size': 5242880,  # 5MB chunks
                'quiet': not verbose,
                'no_warnings': not verbose,
            }
        },
        {
            'name': 'Mixed clients (Android + Web)',
            'opts': {
                'format': 'bestaudio/best',
                'outtmpl': output_template,
                'extractor_args': {'youtube': {'player_client': ['android', 'web']}},
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
                'socket_timeout': 60,
                'retries': 20,
                'fragment_retries': 20,
                'quiet': not verbose,
                'no_warnings': not verbose,
            }
        }
    ]
    
    # Add iOS strategy if PO token is provided
    if po_token:
        ios_strategy = {
            'name': 'iOS client with PO Token',
            'opts': {
                'format': 'bestaudio[ext=m4a]/bestaudio/best',
                'outtmpl': output_template,
                'extractor_args': {
                    'youtube': {
                        'player_client': ['ios'],
                        'po_token': po_token
                    }
                },
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
                'socket_timeout': 30,
                'retries': 10,
                'fragment_retries': 10,
                'http_chunk_size': 10485760,
                'quiet': not verbose,
                'no_warnings': not verbose,
            }
        }
        # Insert iOS as first strategy if we have a token
        strategies.insert(0, ios_strategy)
    
    for strategy_idx, strategy in enumerate(strategies, 1):
        if verbose:
            print(f"\n=== Attempting Strategy {strategy_idx}/{len(strategies)}: {strategy['name']} ===")
        
        for attempt in range(1, MAX_DOWNLOAD_RETRIES + 1):
            try:
                if verbose:
                    print(f"Attempt {attempt}/{MAX_DOWNLOAD_RETRIES}...")
                
                with yt_dlp.YoutubeDL(strategy['opts']) as ydl:
                    info_dict = ydl.extract_info(video_url, download=True)
                    
                    # Check if audio file was created
                    audio_file = "temp_audio.wav"
                    if ensure_wav_file(audio_file, verbose):
                        if verbose:
                            print(f"✓ Download successful with {strategy['name']}!")
                        return audio_file, info_dict
                    else:
                        if verbose:
                            print("Audio file not found after download, retrying...")
                        
            except yt_dlp.utils.DownloadError as e:
                error_msg = str(e)
                if verbose:
                    print(f"✗ Download error: {error_msg}")
                
                # Check for specific errors that indicate we should try next strategy
                if "Operation timed out" in error_msg or "SABR" in error_msg:
                    if attempt < MAX_DOWNLOAD_RETRIES:
                        if verbose:
                            print(f"Retrying in {RETRY_SLEEP_SECONDS} seconds...")
                        time.sleep(RETRY_SLEEP_SECONDS)
                        continue
                    else:
                        if verbose:
                            print(f"Max retries reached for {strategy['name']}, trying next strategy...")
                        break
                else:
                    # For other errors, try next strategy immediately
                    if verbose:
                        print("Trying next strategy...")
                    break
                    
            except Exception as e:
                if verbose:
                    print(f"✗ Unexpected error: {type(e).__name__}: {e}")
                
                if attempt < MAX_DOWNLOAD_RETRIES:
                    if verbose:
                        print(f"Retrying in {RETRY_SLEEP_SECONDS} seconds...")
                    time.sleep(RETRY_SLEEP_SECONDS)
                else:
                    break
    
    # All strategies failed
    return None, None


def run_whisper_transcription(audio_file: str, whisper_model: str, output_format: str, 
                              output_file: str, verbose: bool = False) -> bool:
    """
    Runs whisper transcription.
    
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Build base command
        command = [
            WHISPER_CPP_PATH,
            "-f", audio_file,
            "-m", whisper_model,
            "-t", "8"
        ]
        
        # Add output format flag
        if output_format == "txt":
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
        
        if verbose:
            print("Running Whisper command:", " ".join(command))
        
        # Don't use capture_output to avoid buffer truncation on long transcriptions
        # Let output go directly to stdout/stderr or suppress entirely
        if verbose:
            # Show all output in verbose mode
            result = subprocess.run(command, check=True)
        else:
            # Suppress output in non-verbose mode but don't capture (avoids truncation)
            result = subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        
        if verbose:
            print("✓ Transcription completed successfully.")
        
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"✗ Whisper transcription failed with exit code {e.returncode}")
        if hasattr(e, 'stderr') and e.stderr:
            print(f"Error output: {e.stderr.decode()}")
        return False
    except FileNotFoundError:
        print(f"✗ Whisper binary not found at: {WHISPER_CPP_PATH}")
        print("Please check the WHISPER_CPP_PATH configuration.")
        return False


def create_markdown_file(markdown_file: str, transcript_file: str, video_info: Dict[str, Any],
                        front_matter: Dict[str, Any], verbose: bool = False) -> bool:
    """Create a Markdown file with front matter and transcript."""
    try:
        if not os.path.exists(transcript_file):
            print(f"✗ Transcript file not found: {transcript_file}")
            return False
        
        with open(transcript_file, "r", encoding='utf-8') as f:
            transcript = f.read()
        
        obsidian_front_matter = {
            'title': video_info.get('title', 'Untitled'),
            'date': datetime.datetime.now().strftime('%Y-%m-%d'),
            'transcribed': datetime.datetime.now().isoformat(),
            'url': video_info.get('webpage_url', video_info.get('url', '')),
            'duration': video_info.get('duration_string', ''),
            'channel': video_info.get('channel', video_info.get('uploader', '')),
            'tags': ['podcast', 'youtube', 'transcription'],
        }
        
        obsidian_front_matter.update(front_matter)
        
        front_matter_lines = ['---']
        for key, value in obsidian_front_matter.items():
            if isinstance(value, list):
                front_matter_lines.append(f"{key}: [{', '.join(repr(v) for v in value)}]")
            elif isinstance(value, str):
                if ':' in value or '#' in value or value.startswith(('*', '-', '[')):
                    front_matter_lines.append(f'{key}: "{value}"')
                else:
                    front_matter_lines.append(f'{key}: {value}')
            else:
                front_matter_lines.append(f'{key}: {value}')
        front_matter_lines.append('---\n')
        
        front_matter_str = '\n'.join(front_matter_lines)
        
        with open(markdown_file, "w", encoding='utf-8') as f:
            f.write(front_matter_str)
            f.write('\n')
            f.write(transcript)
        
        if verbose:
            print(f"✓ Markdown file created: {markdown_file}")
        
        return True
        
    except Exception as e:
        print(f"✗ Error creating Markdown file: {e}")
        return False


def estimate_transcription_time(stats_file: str, audio_duration: float, max_records: int = 100) -> Optional[float]:
    """Estimate transcription time based on historical data."""
    try:
        if not os.path.exists(stats_file):
            return None
        
        with open(stats_file, "r", encoding='utf-8') as f:
            existing_stats = json.load(f)
        
        successful_runs = [
            s for s in existing_stats 
            if s.get('success') and 'transcription_duration' in s and 'audio_duration' in s
        ]
        
        if not successful_runs:
            return None
        
        recent_runs = successful_runs[-max_records:]
        
        ratios = []
        for run in recent_runs:
            trans_duration = run['transcription_duration']
            audio_dur = run['audio_duration']
            if audio_dur > 0:
                ratios.append(trans_duration / audio_dur)
        
        if not ratios:
            return None
        
        avg_ratio = sum(ratios) / len(ratios)
        estimated_time = audio_duration * avg_ratio
        
        return estimated_time
        
    except Exception:
        return None


def save_statistics(stats_file: str, stats: Dict[str, Any], verbose: bool = False) -> bool:
    """Save transcription statistics to a JSON file."""
    try:
        try:
            with open(stats_file, "r", encoding='utf-8') as f:
                existing_stats = json.load(f)
        except FileNotFoundError:
            existing_stats = []
        
        existing_stats.append(stats)
        
        with open(stats_file, "w", encoding='utf-8') as f:
            json.dump(existing_stats, f, indent=2)
        
        if verbose:
            print(f"✓ Statistics saved to: {stats_file}")
        
        return True
        
    except Exception as e:
        print(f"✗ Error saving statistics: {e}")
        return False


def cleanup_temp_files(audio_file: str, verbose: bool = False):
    """Clean up temporary audio files."""
    for ext in ['.wav', '.webm', '.m4a', '.mp3']:
        temp_file = audio_file.replace('.wav', ext)
        if os.path.exists(temp_file):
            try:
                os.remove(temp_file)
                if verbose:
                    print(f"Cleaned up: {temp_file}")
            except Exception as e:
                if verbose:
                    print(f"Warning: Could not remove {temp_file}: {e}")


def process_local_file(input_file: str, output_dir: str, verbose: bool = False) -> tuple[Optional[str], Optional[Dict[str, Any]]]:
    """
    Process a local audio file for transcription.
    
    Args:
        input_file: Path to local audio file
        output_dir: Directory to store processed files
        verbose: Enable verbose output
        
    Returns:
        tuple: (audio_file_path, info_dict) or (None, None) on failure
    """
    if not os.path.exists(input_file):
        print(f"✗ File not found: {input_file}")
        return None, None
        
    filename = os.path.basename(input_file)
    name, ext = os.path.splitext(filename)
    
    # Create info dict similar to yt-dlp output
    info_dict = {
        'title': name.replace('_', ' '),
        'webpage_url': f"file://{os.path.abspath(input_file)}",
        'channel': 'Local File',
        'uploader': 'User',
        'duration': 0,  # Will try to get actual duration
        'duration_string': 'Unknown',
        'view_count': 0,
        'upload_date': datetime.datetime.fromtimestamp(os.path.getmtime(input_file)).strftime('%Y%m%d')
    }
    
    # Try to get duration using ffprobe
    try:
        cmd = [
            'ffprobe', 
            '-v', 'error', 
            '-show_entries', 'format=duration', 
            '-of', 'default=noprint_wrappers=1:nokey=1', 
            input_file
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            duration = float(result.stdout.strip())
            info_dict['duration'] = duration
            info_dict['duration_string'] = format_duration(duration)
    except Exception as e:
        if verbose:
            print(f"Warning: Could not determine duration: {e}")
            
    # Convert to 16kHz mono WAV if needed
    output_wav = os.path.join(output_dir, f"{name}_16k.wav")
    
    if verbose:
        print(f"Converting {filename} to 16kHz mono WAV...")
        
    try:
        command = [
            FFMPEG_PATH,
            '-y',  # Overwrite output file
            '-i', input_file,
            '-ar', '16000',
            '-ac', '1',
            '-c:a', 'pcm_s16le',
            output_wav
        ]
        
        if verbose:
            subprocess.run(command, check=True)
        else:
            subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
            
        return output_wav, info_dict
        
    except subprocess.CalledProcessError as e:
        print(f"✗ Error converting audio: {e}")
        return None, None


def main():
    parser = argparse.ArgumentParser(
        description="Automate audio transcription with Whisper (YouTube URL or local file).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
        Examples:
        %(prog)s "https://www.youtube.com/watch?v=VIDEO_ID"
        %(prog)s "/path/to/audio/file.mp3"
        %(prog)s "https://www.youtube.com/watch?v=VIDEO_ID" --output_dir ./transcripts --verbose
        """
    )
    parser.add_argument("input_source", help="YouTube video URL or path to local audio file")
    parser.add_argument("--output_dir", default=None, 
                       help=f"Output directory for Markdown files (default: {DEFAULT_TRANSCRIPTS_DIR})")
    parser.add_argument("--save_to_obsidian", action="store_true", 
                       help=f"Save to Obsidian vault at: {OBSIDIAN_PATH}")
    parser.add_argument("--whisper_model", default=WHISPER_MODEL, help="Path to Whisper model file")
    parser.add_argument("--output_format", default="otxt", 
                       help="Whisper output format (txt, vtt, srt, json). Prefix with 'o' for original filename.")
    parser.add_argument("--front_matter", default="{}", help="JSON string for Markdown front matter")
    parser.add_argument("--stats_file", default=None, 
                       help=f"File to store statistics (default: {DEFAULT_STATS_FILE})")
    parser.add_argument("--keep_transcript", action="store_true",
                       help="Keep the .txt transcript file alongside the .md file")
    parser.add_argument("--po_token", default=None, 
                       help="PO Token for bypassing YouTube restrictions (format: 'client.type+TOKEN'). See https://github.com/yt-dlp/yt-dlp/wiki/PO-Token-Guide")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")

    args = parser.parse_args()

    # Set defaults for output_dir and stats_file
    if args.output_dir is None:
        args.output_dir = DEFAULT_TRANSCRIPTS_DIR
    if args.stats_file is None:
        args.stats_file = DEFAULT_STATS_FILE

    # Initialize tracking variables
    run_uid = str(uuid.uuid4())
    run_timestamp = datetime.datetime.now().isoformat()
    start_time = time.time()
    audio_file = None
    success = False
    is_local_file = os.path.exists(args.input_source)

    if args.verbose:
        print(f"\n{'='*60}")
        print(f"Transcription Tool")
        print(f"{'='*60}")
        print(f"Input: {args.input_source}")
        print(f"Type: {'Local File' if is_local_file else 'YouTube URL'}")
        print(f"Run ID: {run_uid}")
        print(f"Timestamp: {run_timestamp}")
        print(f"{'='*60}\n")

    try:
        if args.save_to_obsidian:
            output_dir = OBSIDIAN_PATH
            if args.verbose:
                print(f"Saving to Obsidian vault: {output_dir}")
        else:
            output_dir = args.output_dir
        
        os.makedirs(output_dir, exist_ok=True)
        
        if is_local_file:
            print("Processing local file...")
            audio_file, info_dict = process_local_file(
                args.input_source,
                output_dir, # Use output_dir for temporary converted file
                args.verbose
            )
        else:
            # Download audio with retry logic
            print("Downloading audio from YouTube...")
            audio_file, info_dict = download_audio_with_retry(
                args.input_source, 
                'temp_audio.%(ext)s',
                args.verbose,
                args.po_token
            )
        
        if audio_file is None or info_dict is None:
            print("\n✗ Failed to process input.")
            return

        video_title = info_dict.get('title', 'unknown_video')
        smart_filename = create_smart_filename(video_title, max_length=60, include_date=True)
        
        if not is_local_file:
            print(f"\n✓ Audio downloaded successfully!")
        else:
            print(f"\n✓ Local file processed successfully!")
            
        print(f"Title: {video_title}")
        if args.verbose:
            print(f"Smart filename: {smart_filename}")

        # Verify audio file exists
        if not os.path.exists(audio_file):
            raise FileNotFoundError(f"Audio file not found: {audio_file}")

        # Estimate transcription time
        audio_duration = info_dict.get('duration', 0)
        estimated_time = estimate_transcription_time(args.stats_file, audio_duration)
        
        # Run Whisper transcription
        print("\nTranscribing audio with Whisper...")
        if estimated_time:
            print(f"Estimated transcription time: ~{format_duration(estimated_time)} (based on historical data)")
        else:
            if audio_duration > 0:
                print(f"Audio duration: {format_duration(audio_duration)}")
        
        transcription_start = time.time()
        temp_whisper_output = os.path.join(output_dir, f"temp_transcript_{uuid.uuid4().hex[:8]}")
        output_format = args.output_format.replace("o", "")
        
        if not run_whisper_transcription(audio_file, args.whisper_model, output_format, 
                                        temp_whisper_output, args.verbose):
            print("\n✗ Transcription failed.")
            return
        
        transcription_duration = time.time() - transcription_start

        # Create Markdown file
        print("\nCreating Markdown file...")
        markdown_file = get_unique_filename(output_dir, smart_filename, '.md')
        transcript_file = f"{temp_whisper_output}.txt"
        
        try:
            front_matter = json.loads(args.front_matter)
        except json.JSONDecodeError as e:
            print(f"Warning: Invalid JSON in front_matter, using empty dict: {e}")
            front_matter = {}
        
        if not create_markdown_file(markdown_file, transcript_file, info_dict, front_matter, args.verbose):
            print("\n✗ Failed to create Markdown file.")
            return
        
        # Optionally keep transcript file
        if not args.keep_transcript and os.path.exists(transcript_file):
            os.remove(transcript_file)
            if args.verbose:
                print(f"✓ Cleaned up temporary transcript file: {transcript_file}")
        elif args.keep_transcript and os.path.exists(transcript_file):
            if args.verbose:
                print(f"✓ Kept transcript file: {transcript_file}")

        # Store statistics
        end_time = time.time()
        duration = end_time - start_time
        stats = {
            "run_uid": run_uid,
            "run_timestamp": run_timestamp,
            "video_title": video_title,
            "smart_filename": os.path.basename(markdown_file),
            "video_url": args.input_source,
            "run_duration": duration,
            "transcription_duration": transcription_duration,
            "audio_duration": audio_duration,
            "success": True,
            "saved_to_obsidian": args.save_to_obsidian,
            "is_local_file": is_local_file
        }

        save_statistics(args.stats_file, stats, args.verbose)
        
        success = True
        print(f"\n{'='*60}")
        print(f"✓ Transcription completed successfully!")
        print(f"{'='*60}")
        print(f"Markdown file: {markdown_file}")
        print(f"Total duration: {format_duration(duration)}")
        print(f"Transcription duration: {format_duration(transcription_duration)}")
        
        if estimated_time:
            diff = transcription_duration - estimated_time
            diff_abs = abs(diff)
            direction = "slower" if diff > 0 else "faster"
            print(f"Estimation accuracy: {format_duration(diff_abs)} {direction} than estimated")
        if args.save_to_obsidian:
            print(f"✓ Saved to Obsidian vault")
        print(f"{'='*60}\n")

    except KeyboardInterrupt:
        print("\n\n✗ Process interrupted by user.")
        
    except Exception as e:
        print(f"\n✗ An unexpected error occurred: {type(e).__name__}: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        
    finally:
        # Clean up temporary audio files
        if audio_file:
            # For local files, we created a temporary 16k wav that needs cleanup
            # For YouTube, we have the downloaded file
            if is_local_file:
                if os.path.exists(audio_file) and "_16k.wav" in audio_file:
                    try:
                        os.remove(audio_file)
                        if args.verbose:
                            print(f"Cleaned up temporary file: {audio_file}")
                    except Exception as e:
                        if args.verbose:
                            print(f"Warning: Could not remove {audio_file}: {e}")
            else:
                cleanup_temp_files(audio_file, args.verbose)
        
        # Save failure statistics if needed
        if not success:
            end_time = time.time()
            duration = end_time - start_time
            stats = {
                "run_uid": run_uid,
                "run_timestamp": run_timestamp,
                "video_url": args.input_source,
                "run_duration": duration,
                "success": False,
                "is_local_file": is_local_file
            }
            save_statistics(args.stats_file, stats, False)


if __name__ == "__main__":
    main()