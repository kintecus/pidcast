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


class AnalysisError(PidcastError):
    """LLM analysis failed."""

    pass


class ConfigurationError(PidcastError):
    """Configuration or setup error."""

    pass


class FileProcessingError(PidcastError):
    """Error processing local audio file."""

    pass


class FeedFetchError(PidcastError):
    """Failed to fetch RSS feed."""

    pass


class FeedParseError(PidcastError):
    """Failed to parse RSS feed."""

    pass


class DiarizationError(PidcastError):
    """Speaker diarization failed."""

    pass


class ApplePodcastsResolutionError(PidcastError):
    """Failed to resolve Apple Podcasts URL to audio."""

    pass
