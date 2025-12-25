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
