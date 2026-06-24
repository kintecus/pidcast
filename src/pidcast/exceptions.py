"""Custom exception hierarchy for pidcast."""


class PidcastError(Exception):
    """Base exception for all pidcast errors."""

    pass


class DownloadError(PidcastError):
    """Failed to download audio from source."""

    pass


class TranscriptionError(PidcastError):
    """Whisper transcription failed."""

    pass


class TranscriptionPaused(PidcastError):  # noqa: N818 - a control signal, not an error
    """Transcription was deliberately paused (e.g. via Ctrl-C).

    Not an error: already-streamed segments are checkpointed and the job can be
    resumed with ``pidcast resume``. Carries the last offset reached so callers
    can report progress.
    """

    def __init__(self, message: str = "", last_offset_ms: int = 0) -> None:
        super().__init__(message)
        self.last_offset_ms = last_offset_ms


class AnalysisError(PidcastError):
    """LLM analysis failed."""

    pass


class ConfigurationError(PidcastError):
    """Configuration or setup error."""

    pass


class FileProcessingError(PidcastError):
    """Error processing local audio file."""

    pass


class LibraryError(PidcastError):
    """Base exception for library operations."""

    pass


class FeedFetchError(LibraryError):
    """Failed to fetch RSS feed."""

    pass


class FeedParseError(LibraryError):
    """Failed to parse RSS feed."""

    pass


class ShowNotFoundError(LibraryError):
    """Show not found in library."""

    pass


class DuplicateShowError(LibraryError):
    """Show already exists in library."""

    pass


class DiarizationError(PidcastError):
    """Speaker diarization failed."""

    pass


class ApplePodcastsResolutionError(PidcastError):
    """Failed to resolve Apple Podcasts URL to audio."""

    pass
