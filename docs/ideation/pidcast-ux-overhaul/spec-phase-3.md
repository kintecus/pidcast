# Implementation Spec: Pidcast UX Overhaul - Phase 3

**PRD**: ./prd-phase-3.md
**Estimated Effort**: L (Large)

## Technical Approach

This phase implements semantic chunking for long transcripts, enabling reliable analysis of 45+ minute videos. The approach:

1. **Detect long content**: Check if transcript exceeds model's context window
2. **Semantic chunking**: Split at paragraph/topic boundaries, not mid-sentence
3. **Chunk analysis**: Analyze each chunk with a chunk-aware prompt
4. **Synthesis**: Combine chunk results into coherent final output

We'll add a new `chunking.py` module that handles splitting and synthesis, keeping the analysis module focused on LLM calls.

## File Changes

### New Files

| File Path | Purpose |
|-----------|---------|
| `src/pidcast/chunking.py` | Transcript chunking and synthesis logic |

### Modified Files

| File Path | Changes |
|-----------|---------|
| `config/prompts.yaml` | Add chunk_analysis and synthesis prompt types |
| `src/pidcast/analysis.py` | Integrate chunking, add progress display |
| `src/pidcast/config.py` | Add chunking-related constants |

## Implementation Details

### 1. Chunking Constants

**Overview**: Add constants for chunking behavior.

```python
# In config.py

# Chunking configuration
CHUNK_TARGET_TOKENS = 6000  # Target tokens per chunk (leaves room for prompt)
CHUNK_OVERLAP_TOKENS = 100  # Overlap between chunks for context
CHUNK_MIN_CHARS = 500  # Minimum chunk size to avoid tiny fragments
CHAR_TO_TOKEN_RATIO = 4  # Approximate chars per token
```

### 2. Chunking Module

**Pattern to follow**: Sentence boundary detection similar to `truncate_transcript()`

**Overview**: New module for splitting transcripts and synthesizing results.

```python
# src/pidcast/chunking.py
"""Transcript chunking and synthesis for long content."""

import re
import logging
from dataclasses import dataclass

from .config import (
    CHUNK_TARGET_TOKENS,
    CHUNK_OVERLAP_TOKENS,
    CHUNK_MIN_CHARS,
    CHAR_TO_TOKEN_RATIO,
)

logger = logging.getLogger(__name__)


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
    """Estimate token count from text."""
    return len(text) // CHAR_TO_TOKEN_RATIO


def needs_chunking(transcript: str, max_tokens: int) -> bool:
    """Check if transcript needs to be chunked.

    Args:
        transcript: Full transcript text
        max_tokens: Model's max context tokens (minus prompt overhead)

    Returns:
        True if transcript exceeds token limit
    """
    estimated = estimate_tokens(transcript)
    return estimated > max_tokens


def find_semantic_boundary(text: str, target_pos: int, search_range: int = 500) -> int:
    """Find best semantic boundary near target position.

    Looks for paragraph breaks, then sentence boundaries, then word boundaries.

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
    para_breaks = list(re.finditer(r'\n\n+', search_text))
    if para_breaks:
        # Find closest to target
        closest = min(para_breaks, key=lambda m: abs(m.end() + search_start - target_pos))
        return search_start + closest.end()

    # Priority 2: Sentence boundary (. ! ? followed by space/newline)
    sentences = list(re.finditer(r'[.!?]\s+', search_text))
    if sentences:
        closest = min(sentences, key=lambda m: abs(m.end() + search_start - target_pos))
        return search_start + closest.end()

    # Priority 3: Single newline
    newlines = list(re.finditer(r'\n', search_text))
    if newlines:
        closest = min(newlines, key=lambda m: abs(m.end() + search_start - target_pos))
        return search_start + closest.end()

    # Fallback: word boundary
    words = list(re.finditer(r'\s+', search_text))
    if words:
        closest = min(words, key=lambda m: abs(m.end() + search_start - target_pos))
        return search_start + closest.end()

    return target_pos


def chunk_transcript(
    transcript: str,
    target_tokens: int = CHUNK_TARGET_TOKENS,
    overlap_tokens: int = CHUNK_OVERLAP_TOKENS,
) -> list[TranscriptChunk]:
    """Split transcript into semantic chunks.

    Args:
        transcript: Full transcript text
        target_tokens: Target tokens per chunk
        overlap_tokens: Tokens to overlap between chunks

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
            if len(chunk_text) >= CHUNK_MIN_CHARS:
                chunks.append(TranscriptChunk(
                    index=chunk_index,
                    total_chunks=0,  # Will be updated after
                    text=chunk_text,
                    char_start=pos,
                    char_end=len(transcript),
                    estimated_tokens=estimate_tokens(chunk_text),
                ))
            break

        # Find best boundary
        boundary = find_semantic_boundary(transcript, chunk_end)
        chunk_text = transcript[pos:boundary].strip()

        if len(chunk_text) >= CHUNK_MIN_CHARS:
            chunks.append(TranscriptChunk(
                index=chunk_index,
                total_chunks=0,
                text=chunk_text,
                char_start=pos,
                char_end=boundary,
                estimated_tokens=estimate_tokens(chunk_text),
            ))
            chunk_index += 1

        # Move position, accounting for overlap
        pos = max(boundary - overlap_chars, pos + CHUNK_MIN_CHARS)

    # Update total_chunks in all chunks
    total = len(chunks)
    for chunk in chunks:
        chunk.total_chunks = total

    logger.info(f"Split transcript into {total} chunks")
    for i, chunk in enumerate(chunks):
        logger.debug(f"  Chunk {i+1}: {chunk.estimated_tokens} tokens")

    return chunks


def synthesize_chunk_results(
    chunk_results: list[str],
    video_title: str,
    total_duration: str,
) -> str:
    """Prepare chunk results for synthesis prompt.

    Args:
        chunk_results: List of analysis results from each chunk
        video_title: Original video title
        total_duration: Original video duration

    Returns:
        Formatted input for synthesis prompt
    """
    parts = [
        f"# Video: {video_title}",
        f"# Duration: {total_duration}",
        f"# Analyzed in {len(chunk_results)} chunks",
        "",
    ]

    for i, result in enumerate(chunk_results):
        parts.append(f"## Chunk {i+1} of {len(chunk_results)}")
        parts.append(result.strip())
        parts.append("")

    return "\n".join(parts)
```

### 3. New Prompts for Chunking

**Overview**: Add chunk_analysis and synthesis prompts to prompts.yaml.

```yaml
# Add to config/prompts.yaml

  chunk_analysis:
    name: Chunk Analysis
    description: Analyze a portion of a longer transcript
    max_output_tokens: 1000
    system_prompt: |
      You are analyzing a portion of a longer video transcript.
      Extract key information that can be synthesized with other chunks later.
      Focus on facts, insights, and notable points - not overall structure.
    user_prompt: |
      Analyze this section of a transcript.

      Title: {title}
      This is chunk {chunk_num} of {total_chunks}.

      # Transcript Section
      {transcript}

      # Instructions
      Provide:
      1. **Key Points** (3-5 bullet points of main ideas in this section)
      2. **Notable Details** (specific facts, quotes, or data mentioned)
      3. **Topics Covered** (brief list of subjects discussed)

      Keep analysis focused and factual. Do not summarize the entire video.

  synthesis:
    name: Synthesis
    description: Combine chunk analyses into coherent summary
    max_output_tokens: 1500
    system_prompt: |
      You are synthesizing multiple chunk analyses into a coherent summary.
      Remove redundancy, identify overarching themes, and create a unified narrative.
    user_prompt: |
      Synthesize these chunk analyses into a coherent executive brief.

      {chunk_analyses}

      # Instructions
      Create a unified summary:

      1. **Executive Summary** (2-3 sentences covering the entire content)
         - What is this content about overall?
         - What's the main value for viewers?

      2. **Key Points** (5-10 bullet points, deduplicated across chunks)
         - Prioritize by importance, not chronology
         - Merge related points from different chunks

      3. **Notable Insights** (2-3 standout quotes or data points)

      The output should read as if analyzing the original full transcript,
      not as a collection of chunk summaries.
```

### 4. Analysis Module Integration

**Overview**: Update analysis to detect long content and use chunking.

```python
# Updates to analyze_transcript_with_llm() in analysis.py

def analyze_transcript_with_llm(
    transcript: str,
    video_info: VideoInfo,
    analysis_type: str,
    prompts_config: PromptsConfig,
    api_key: str,
    model: str | None = None,
    verbose: bool = False,
) -> AnalysisResult:
    """Analyze transcript, using chunking for long content."""

    from .chunking import needs_chunking, chunk_transcript, synthesize_chunk_results
    from rich.progress import Progress, SpinnerColumn, TextColumn

    # Get model config for context limit
    models_config = load_models_config(DEFAULT_MODELS_FILE)
    model_to_use = model or models_config.default_model
    model_config = models_config.models.get(model_to_use)

    # Calculate available tokens for transcript (minus prompt overhead)
    prompt_overhead = 2000  # System + user prompt templates
    max_transcript_tokens = (model_config.tpm if model_config else 8000) - prompt_overhead

    # Check if chunking needed
    if needs_chunking(transcript, max_transcript_tokens):
        return _analyze_with_chunking(
            transcript, video_info, prompts_config, api_key, model_to_use, verbose
        )

    # Standard single-call analysis
    return _analyze_single(
        transcript, video_info, analysis_type, prompts_config, api_key, model_to_use, verbose
    )


def _analyze_with_chunking(
    transcript: str,
    video_info: VideoInfo,
    prompts_config: PromptsConfig,
    api_key: str,
    model: str,
    verbose: bool,
) -> AnalysisResult:
    """Analyze long transcript using chunking and synthesis."""

    from .chunking import chunk_transcript, synthesize_chunk_results
    from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn

    logger.info(f"Transcript too long for single analysis, using chunking...")

    # Split into chunks
    chunks = chunk_transcript(transcript)
    logger.info(f"Analyzing {len(chunks)} chunks...")

    # Analyze each chunk
    chunk_results = []
    total_tokens = 0
    total_cost = 0.0

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
    ) as progress:
        task = progress.add_task("Analyzing chunks...", total=len(chunks))

        for chunk in chunks:
            progress.update(task, description=f"Chunk {chunk.index + 1}/{chunk.total_chunks}")

            # Use chunk_analysis prompt with chunk-specific variables
            result = _analyze_chunk(
                chunk, video_info, prompts_config, api_key, model, verbose
            )
            chunk_results.append(result.analysis_text)
            total_tokens += result.tokens_total
            total_cost += result.estimated_cost or 0

            progress.advance(task)

    # Synthesize results
    logger.info("Synthesizing chunk results...")
    synthesis_input = synthesize_chunk_results(
        chunk_results, video_info.title, video_info.duration_string
    )

    synthesis_result = _run_synthesis(
        synthesis_input, prompts_config, api_key, model, verbose
    )

    total_tokens += synthesis_result.tokens_total
    total_cost += synthesis_result.estimated_cost or 0

    return AnalysisResult(
        analysis_text=synthesis_result.analysis_text,
        analysis_type="executive_summary",
        analysis_name="Executive Summary (Chunked)",
        model=model,
        provider="groq",
        tokens_input=total_tokens,  # Approximate
        tokens_output=0,
        tokens_total=total_tokens,
        estimated_cost=total_cost,
        duration=0,  # Would need to track total
        truncated=False,
    )


def _analyze_chunk(
    chunk: "TranscriptChunk",
    video_info: VideoInfo,
    prompts_config: PromptsConfig,
    api_key: str,
    model: str,
    verbose: bool,
) -> AnalysisResult:
    """Analyze a single chunk."""
    template = prompts_config.prompts["chunk_analysis"]

    variables = {
        "transcript": chunk.text,
        "title": video_info.title,
        "chunk_num": str(chunk.index + 1),
        "total_chunks": str(chunk.total_chunks),
    }

    # Call API (reuse existing logic)
    return _call_analysis_api(template, variables, api_key, model, verbose)


def _run_synthesis(
    chunk_analyses: str,
    prompts_config: PromptsConfig,
    api_key: str,
    model: str,
    verbose: bool,
) -> AnalysisResult:
    """Run synthesis on chunk results."""
    template = prompts_config.prompts["synthesis"]

    variables = {
        "chunk_analyses": chunk_analyses,
    }

    return _call_analysis_api(template, variables, api_key, model, verbose)
```

### 5. Progress Display

**Overview**: Show progress during chunked analysis.

```python
# Using Rich progress bar (already in dependencies)

from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeRemainingColumn

with Progress(
    SpinnerColumn(),
    TextColumn("[cyan]{task.description}"),
    BarColumn(complete_style="green"),
    TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
    TimeRemainingColumn(),
) as progress:
    task = progress.add_task("Analyzing...", total=len(chunks))

    for i, chunk in enumerate(chunks):
        progress.update(task, description=f"Chunk {i+1}/{len(chunks)}")
        # ... analyze chunk ...
        progress.advance(task)

    progress.update(task, description="Synthesizing results...")
```

## Testing Requirements

### Manual Testing

- [ ] 60-minute transcript successfully produces coherent summary
- [ ] Progress shows "Analyzing chunk N of M"
- [ ] Final summary doesn't read as disjointed chunks
- [ ] Key points are deduplicated
- [ ] Short transcripts skip chunking (single-call path)
- [ ] Chunking works with model fallback

### Test Cases

| Scenario | Expected |
|----------|----------|
| 10-minute video | No chunking, single analysis |
| 45-minute video | Chunks into ~4-5 parts |
| 90-minute video | Chunks into ~8-10 parts |
| Chunk at paragraph | Split occurs at double newline |
| Chunk mid-paragraph | Split occurs at sentence boundary |

## Error Handling

| Error Scenario | Handling Strategy |
|----------------|-------------------|
| Single chunk fails | Retry with fallback model, then skip chunk with warning |
| Synthesis fails | Return concatenated chunk results with warning |
| All chunks fail | Raise AnalysisError with details |
| Very short chunks | Merge with adjacent chunk if < 500 chars |

## Validation Commands

```bash
# Lint
uv run ruff check src/pidcast/
uv run ruff format src/pidcast/

# Test with known long transcript
uv run pidcast --analyze_existing data/transcripts/2026-01-05_Taking_AI_Doom_Seriously_For_62_Minutes.md

# Test short transcript (should not chunk)
uv run pidcast --analyze_existing data/transcripts/short_video.md

# Verify progress display
# Should see "Chunk 1/N ... Chunk N/N ... Synthesizing..."

# Check output quality
# Summary should be coherent, not "Chunk 1 discussed X, Chunk 2 discussed Y"
```

---

*This spec is ready for implementation. Follow the patterns and validate at each step.*
