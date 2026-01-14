"""Transcript chunking and synthesis for long content."""

import logging
import re
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# Default chunking parameters
DEFAULT_CHUNK_TARGET_TOKENS = 20000  # Target tokens per chunk
DEFAULT_CHUNK_OVERLAP_TOKENS = 200  # Overlap between chunks for context
CHUNK_MIN_CHARS = 1000  # Minimum chunk size to avoid tiny fragments
CHAR_TO_TOKEN_RATIO = 4  # Approximate chars per token


@dataclass
class TranscriptChunk:
    """A chunk of transcript with metadata."""

    index: int
    total_chunks: int
    text: str
    char_start: int
    char_end: int
    estimated_tokens: int


def estimate_tokens(text: str) -> int:
    """Estimate token count from text length."""
    return len(text) // CHAR_TO_TOKEN_RATIO


def needs_chunking(transcript: str, max_tokens: int) -> bool:
    """Check if transcript needs to be chunked.

    Args:
        transcript: Full transcript text
        max_tokens: Maximum tokens the model can handle

    Returns:
        True if transcript exceeds token limit
    """
    estimated = estimate_tokens(transcript)
    return estimated > max_tokens


def find_semantic_boundary(text: str, target_pos: int, search_range: int = 1000) -> int:
    """Find best semantic boundary near target position.

    Looks for paragraph breaks, then sentence boundaries, then word boundaries.
    Prioritizes natural break points for more coherent chunks.

    Args:
        text: Text to search in
        target_pos: Ideal split position
        search_range: How far to search from target

    Returns:
        Best boundary position
    """
    search_start = max(0, target_pos - search_range)
    search_end = min(len(text), target_pos + search_range)
    search_text = text[search_start:search_end]

    # Priority 1: Double newline (paragraph break)
    para_breaks = list(re.finditer(r"\n\n+", search_text))
    if para_breaks:
        # Find closest to target
        closest = min(para_breaks, key=lambda m: abs(m.end() + search_start - target_pos))
        return search_start + closest.end()

    # Priority 2: Sentence boundary (. ! ? followed by space/newline)
    sentences = list(re.finditer(r"[.!?]\s+", search_text))
    if sentences:
        closest = min(sentences, key=lambda m: abs(m.end() + search_start - target_pos))
        return search_start + closest.end()

    # Priority 3: Single newline
    newlines = list(re.finditer(r"\n", search_text))
    if newlines:
        closest = min(newlines, key=lambda m: abs(m.end() + search_start - target_pos))
        return search_start + closest.end()

    # Fallback: word boundary
    words = list(re.finditer(r"\s+", search_text))
    if words:
        closest = min(words, key=lambda m: abs(m.end() + search_start - target_pos))
        return search_start + closest.end()

    return target_pos


def chunk_transcript(
    transcript: str,
    target_tokens: int = DEFAULT_CHUNK_TARGET_TOKENS,
    overlap_tokens: int = DEFAULT_CHUNK_OVERLAP_TOKENS,
) -> list[TranscriptChunk]:
    """Split transcript into semantic chunks.

    Splits at natural boundaries (paragraphs, sentences) rather than
    arbitrary character positions for more coherent chunks.

    Args:
        transcript: Full transcript text
        target_tokens: Target tokens per chunk
        overlap_tokens: Tokens to overlap between chunks for context

    Returns:
        List of TranscriptChunk objects
    """
    target_chars = target_tokens * CHAR_TO_TOKEN_RATIO
    overlap_chars = overlap_tokens * CHAR_TO_TOKEN_RATIO

    chunks = []
    pos = 0
    chunk_index = 0

    while pos < len(transcript):
        # Calculate end of this chunk
        chunk_end = pos + target_chars

        if chunk_end >= len(transcript):
            # Last chunk - take everything remaining
            chunk_text = transcript[pos:].strip()
            if len(chunk_text) >= CHUNK_MIN_CHARS or not chunks:
                chunks.append(
                    TranscriptChunk(
                        index=chunk_index,
                        total_chunks=0,  # Will be updated after
                        text=chunk_text,
                        char_start=pos,
                        char_end=len(transcript),
                        estimated_tokens=estimate_tokens(chunk_text),
                    )
                )
            elif chunks:
                # Merge small final chunk with previous
                prev = chunks[-1]
                merged_text = transcript[prev.char_start :].strip()
                chunks[-1] = TranscriptChunk(
                    index=prev.index,
                    total_chunks=0,
                    text=merged_text,
                    char_start=prev.char_start,
                    char_end=len(transcript),
                    estimated_tokens=estimate_tokens(merged_text),
                )
            break

        # Find best boundary
        boundary = find_semantic_boundary(transcript, chunk_end)
        chunk_text = transcript[pos:boundary].strip()

        if len(chunk_text) >= CHUNK_MIN_CHARS:
            chunks.append(
                TranscriptChunk(
                    index=chunk_index,
                    total_chunks=0,
                    text=chunk_text,
                    char_start=pos,
                    char_end=boundary,
                    estimated_tokens=estimate_tokens(chunk_text),
                )
            )
            chunk_index += 1

        # Move position, accounting for overlap
        pos = max(boundary - overlap_chars, pos + CHUNK_MIN_CHARS)

    # Update total_chunks in all chunks
    total = len(chunks)
    for chunk in chunks:
        chunk.total_chunks = total

    logger.info(f"Split transcript into {total} chunks")
    for i, chunk in enumerate(chunks):
        logger.debug(f"  Chunk {i + 1}: ~{chunk.estimated_tokens} tokens")

    return chunks


def format_chunk_for_analysis(
    chunk: TranscriptChunk,
    video_title: str,
) -> dict[str, str]:
    """Format a chunk with metadata for the analysis prompt.

    Args:
        chunk: The transcript chunk
        video_title: Original video title

    Returns:
        Dict of variables for prompt substitution
    """
    return {
        "transcript": chunk.text,
        "title": video_title,
        "chunk_num": str(chunk.index + 1),
        "total_chunks": str(chunk.total_chunks),
    }


def format_chunks_for_synthesis(
    chunk_results: list[str],
    video_title: str,
    video_duration: str,
) -> str:
    """Prepare chunk analysis results for synthesis prompt.

    Args:
        chunk_results: List of analysis results from each chunk
        video_title: Original video title
        video_duration: Original video duration string

    Returns:
        Formatted input for synthesis prompt
    """
    parts = [
        f"# Video: {video_title}",
        f"# Duration: {video_duration}",
        f"# Analyzed in {len(chunk_results)} chunks",
        "",
    ]

    for i, result in enumerate(chunk_results):
        parts.append(f"## Chunk {i + 1} of {len(chunk_results)}")
        parts.append(result.strip())
        parts.append("")

    return "\n".join(parts)
