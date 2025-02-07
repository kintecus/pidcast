import argparse
from typing import Dict, Any
import subprocess
import time
from datetime import datetime
import yaml

def load_metadata(config_file: str) -> Dict[str, Any]:
    """Load metadata configuration from YAML file."""
    with open(config_file, 'r') as f:
        return yaml.safe_load(f)

def process_youtube_video(url: str, output_folder: str, metadata: Dict[str, Any]):
    """Process a YouTube video and save as .md file."""
    # Download audio
    command = (
        f'ytdl {url} --extract-audio --audio-format wav -o audio.wav'
    )
    subprocess.run(command.split(), check=True)
    
    # Transcribe using Whisper
    # Assuming Whisper is installed in PATH, and the model is available
    transcribe_cmd = ('python3', 'whisper.cpp', '-r', 'en-us', 'audio.wav')
    subprocess.run(transcribe_cmd, check=True)
    
    # Generate transcript text
    with open('transcript.txt') as f:
        transcript = f.read()
    
    # Create metadata front matter
    front_matter = generate_front_matter(metadata, 'youtube', url, transcript)
    
    # Save to .md file
    md_file = os.path.join(output_folder, 'processed_content.md')
    with open(md_file, 'w') as f:
        f.write(f'{front_matter}\n{transcript}')
    
def process_web_page(url: str, output_folder: str, metadata: Dict[str, Any]):
    """Process a web page and save as .md file."""
    # Example processing logic
    page = requests.get(url)
    soup = BeautifulSoup(page.text, 'html.parser')
    text = soup.get_text()
    
    # Generate front matter based on metadata
    front_matter = generate_front_matter(metadata, 'web', url, text)
    
    # Save to .md file
    md_file = os.path.join(output_folder, 'processed_content.md')
    with open(md_file, 'w') as f:
        f.write(f'{front_matter}\n{text}')
    
def generate_front_matter(metadata: Dict[str, Any], content_type: str, url: str, content: str) -> str:
    """Generate front matter based on metadata configuration."""
    front_matter = []
    front_matter.append('---')
    for key, value in metadata.items():
        if key == 'url':
            front_matter.append(f'  {key}: {url}')
        else:
            if content_type == 'youtube':
                # Handle YouTube-specific metadata
                pass
            elif content_type == 'web':
                # Handle web page-specific metadata
                pass
    front_matter.append('---')
    return '\n'.join(front_matter) + '\n' + content

def main() -> None:
    """Main function to process and save content."""
    # Parse arguments
    parser = argparse.ArgumentParser(description='Process content and save as .md file.')
    parser.add_argument('--url', type=str, required=True, help='URL of the content.')
    parser.add_argument('--config', type=str, default='metadata.yaml', 
                       help='Path to metadata configuration file.')
    parser.add_argument('--output_folder', type=str, default='output',
                       help='Folder to save processed files.')
    args = parser.parse_args()
    
    # Load metadata
    metadata = load_metadata(args.config)
    
    # Determine content type
    if 'youtube.com' in args.url:
        process_youtube_video(args.url, args.output_folder, metadata)
    else:
        process_web_page(args.url, args.output_folder, metadata)

if __name__ == '__main__':
    main()