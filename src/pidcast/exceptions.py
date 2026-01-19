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
