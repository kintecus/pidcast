"""Reference transcript management for evals."""

import json
import re
from dataclasses import dataclass
from pathlib import Path

from pidcast.exceptions import ConfigurationError, FileProcessingError


@dataclass
class ReferenceTranscript:
    """Metadata about a reference transcript for evals."""

    transcript_id: str  # e.g., "tech-talk-01"
    name: str
    description: str
    file_path: Path  # Path to .md file
    content_type: str  # e.g., "technical_talk", "interview"
    duration_seconds: int
    source_url: str


class ReferenceTranscriptManager:
    """Manages reference transcript registry."""

    def __init__(self, registry_file: Path, project_root: Path):
        """
        Initialize reference transcript manager.

        Args:
            registry_file: Path to reference_transcripts.json
            project_root: Project root directory for resolving relative paths
        """
        self.registry_file = registry_file
        self.project_root = project_root
        self._transcripts: dict[str, ReferenceTranscript] = {}
        self._load_registry()

    def _load_registry(self) -> None:
        """Load registry from JSON, validate files exist."""
        if not self.registry_file.exists():
            raise ConfigurationError(
                f"Reference transcripts registry not found: {self.registry_file}. "
                "Create config/reference_transcripts.json with transcript definitions."
            )

        try:
            with open(self.registry_file) as f:
                data = json.load(f)
        except json.JSONDecodeError as e:
            raise ConfigurationError(
                f"Invalid JSON in {self.registry_file}: {e}"
            )

        if "transcripts" not in data:
            raise ConfigurationError(
                f"Missing 'transcripts' key in {self.registry_file}"
            )

        # Parse transcripts
        for transcript_data in data["transcripts"]:
            try:
                transcript_id = transcript_data["transcript_id"]

                # Resolve file path (support relative paths)
                file_path_str = transcript_data["file_path"]
                file_path = Path(file_path_str)
                if not file_path.is_absolute():
                    file_path = self.project_root / file_path

                # Validate file exists
                if not file_path.exists():
                    raise ConfigurationError(
                        f"Reference transcript file not found: {file_path} "
                        f"(for transcript '{transcript_id}')"
                    )

                transcript = ReferenceTranscript(
                    transcript_id=transcript_id,
                    name=transcript_data["name"],
                    description=transcript_data["description"],
                    file_path=file_path,
                    content_type=transcript_data["content_type"],
                    duration_seconds=transcript_data["duration_seconds"],
                    source_url=transcript_data["source_url"],
                )
                self._transcripts[transcript_id] = transcript

            except KeyError as e:
                raise ConfigurationError(
                    f"Missing required field {e} in reference transcript "
                    f"'{transcript_data.get('transcript_id', '?')}'"
                )

    def get_transcript(self, transcript_id: str) -> ReferenceTranscript:
        """
        Retrieve reference transcript by ID.

        Args:
            transcript_id: Transcript identifier

        Returns:
            ReferenceTranscript instance

        Raises:
            ConfigurationError: If transcript ID not found
        """
        if transcript_id not in self._transcripts:
            available = ", ".join(sorted(self._transcripts.keys()))
            raise ConfigurationError(
                f"Reference transcript '{transcript_id}' not found. "
                f"Available transcripts: {available}"
            )
        return self._transcripts[transcript_id]

    def list_transcript_ids(self) -> list[str]:
        """
        List all available transcript IDs.

        Returns:
            List of transcript ID strings
        """
        return sorted(self._transcripts.keys())

    def read_transcript_content(self, transcript_id: str) -> str:
        """
        Read the full markdown content of a reference transcript.

        Extracts the transcript text, stripping YAML front matter if present.

        Args:
            transcript_id: Transcript identifier

        Returns:
            Transcript text content

        Raises:
            FileProcessingError: If file cannot be read
        """
        transcript = self.get_transcript(transcript_id)

        try:
            content = transcript.file_path.read_text()
        except Exception as e:
            raise FileProcessingError(
                f"Failed to read transcript file {transcript.file_path}: {e}"
            )

        # Strip YAML front matter if present
        content = self._strip_yaml_front_matter(content)

        return content.strip()

    def _strip_yaml_front_matter(self, content: str) -> str:
        """
        Remove YAML front matter from markdown content.

        Args:
            content: Markdown content potentially with YAML front matter

        Returns:
            Content without front matter
        """
        # YAML front matter is delimited by --- at start and end
        pattern = r"^---\s*\n.*?\n---\s*\n"
        return re.sub(pattern, "", content, count=1, flags=re.DOTALL)

    def get_all_transcripts(self) -> list[ReferenceTranscript]:
        """
        Get all reference transcripts.

        Returns:
            List of all ReferenceTranscript instances
        """
        return list(self._transcripts.values())
