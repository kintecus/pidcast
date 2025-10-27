# YouTube Transcription Script - Refactored Version

## Key Improvements

### 1. **Multi-Strategy Download with Automatic Fallback**

The script now tries three different download strategies in sequence:

#### Strategy 1: iOS Client (Most Reliable)

- Uses `player_client: ['ios']` which bypasses SABR streaming issues
- Prefers m4a format (most compatible)
- 10MB chunk size for reliable downloads
- 10 retries with fragment recovery

#### Strategy 2: iOS + Web Client Fallback

- Combines iOS and web clients for broader compatibility
- Uses smaller 5MB chunks for unstable connections
- 15 retries with longer timeout (45s)

#### Strategy 3: Android Client (Last Resort)

- Falls back to Android client if others fail
- Maximum retries (20) with 60s timeout
- Works with any available format

### 2. **Robust Retry Logic**

- Each strategy attempts up to 3 full download cycles
- 10-second sleep between retries
- Intelligent error detection (timeouts vs other errors)
- Automatic progression to next strategy on persistent failures

### 3. **Better Error Handling**

- Specific exception handling for DownloadError, subprocess errors, etc.
- Graceful degradation with helpful error messages
- Keyboard interrupt handling (Ctrl+C)
- Comprehensive cleanup in finally block

### 4. **Enhanced User Feedback**

- Clear progress indicators (✓ and ✗ symbols)
- Strategy and attempt counters
- Detailed error messages with actionable solutions
- Better verbose output formatting

### 5. **Improved Code Structure**

- Type hints for better code clarity
- Separated concerns into focused functions
- Better return value handling (success/failure booleans)
- Consistent error reporting

### 6. **Additional Features**

- Automatic metadata addition to front matter (video URL, title, date)
- UTF-8 encoding for international characters
- Multiple temp file cleanup (.wav, .webm, .m4a, .mp3)
- Success/failure tracking in statistics

## Configuration Changes

### New yt-dlp Options Added:

```python
'extractor_args': {'youtube': {'player_client': ['ios']}}  # Bypass SABR
'socket_timeout': 30                                       # Network timeout
'retries': 10                                              # General retries
'fragment_retries': 10                                     # Fragment recovery
'http_chunk_size': 10485760                               # 10MB chunks
```

## Usage Examples

### Basic usage:

```bash
python yt_transcribe.py "https://www.youtube.com/watch?v=VIDEO_ID"
```

### With verbose output (recommended for troubleshooting):

```bash
python yt_transcribe.py "https://www.youtube.com/watch?v=VIDEO_ID" --verbose
```

### With custom output directory and front matter:

```bash
python yt_transcribe.py "https://www.youtube.com/watch?v=VIDEO_ID" \
  --output_dir ./transcripts \
  --front_matter '{"tags": ["podcast", "ai"], "author": "John Doe"}' \
  --verbose
```

## Troubleshooting

If downloads still fail after all strategies:

1. **Update yt-dlp**:

   ```bash
   pip install --upgrade yt-dlp
   ```

2. **Check network connectivity**:

   - Try a different network
   - Disable VPN if enabled (or enable if disabled)
   - Check firewall settings

3. **Verify video availability**:

   - Make sure the video is public and available in your region
   - Check if the video has age restrictions

4. **Try with verbose flag**:
   ```bash
   python yt_transcribe.py "URL" --verbose
   ```
   This will show detailed information about which strategy and attempt failed.

## What Was Fixed

### Original Issues:

- ❌ Single download attempt (no retries)
- ❌ No fallback strategies for SABR streaming
- ❌ Generic error handling
- ❌ No network timeout configuration
- ❌ Hard to diagnose failures

### Now Fixed:

- ✅ Multiple retry attempts with backoff
- ✅ Three fallback strategies optimized for SABR
- ✅ Specific error handling with helpful messages
- ✅ Configurable timeouts and chunk sizes
- ✅ Detailed progress and error reporting
- ✅ Graceful failure with actionable suggestions

## Performance Notes

- **iOS client** is typically fastest and most reliable
- **Chunk sizes** are optimized for balance between speed and reliability
- **Fragment retries** ensure partial downloads can resume
- The script will automatically find the best working strategy

## Future Enhancements (Optional)

Consider adding:

- Cookie support for age-restricted videos
- Proxy support for geo-restricted content
- Parallel downloads for playlists
- Resume capability for interrupted transcriptions
- Progress bars for download and transcription
