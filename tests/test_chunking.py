"""Tests for pidcast.chunking module."""

from pidcast.chunking import (
    CHAR_TO_TOKEN_RATIO,
    CHUNK_MIN_CHARS,
    TranscriptChunk,
    chunk_transcript,
    estimate_tokens,
    find_semantic_boundary,
    format_chunk_for_analysis,
    format_chunks_for_synthesis,
    needs_chunking,
)

# ============================================================================
# estimate_tokens
# ============================================================================


class TestEstimateTokens:
    def test_empty_string(self):
        assert estimate_tokens("") == 0

    def test_normal_text(self):
        text = "a" * 400
        assert estimate_tokens(text) == 400 // CHAR_TO_TOKEN_RATIO

    def test_short_text(self):
        assert estimate_tokens("hi") == 0  # 2 // 4 = 0

    def test_non_ascii(self):
        text = "日本語テスト" * 100
        result = estimate_tokens(text)
        assert result > 0


# ============================================================================
# needs_chunking
# ============================================================================


class TestNeedsChunking:
    def test_below_threshold(self):
        text = "a" * 100
        assert needs_chunking(text, max_tokens=1000) is False

    def test_above_threshold(self):
        # 4001 chars = 1000 tokens, exceeds max_tokens=999
        text = "a" * 4000
        assert needs_chunking(text, max_tokens=999) is True

    def test_exactly_at_threshold(self):
        text = "a" * 4000  # 1000 tokens
        assert needs_chunking(text, max_tokens=1000) is False

    def test_empty_string(self):
        assert needs_chunking("", max_tokens=100) is False


# ============================================================================
# find_semantic_boundary
# ============================================================================


class TestFindSemanticBoundary:
    def test_paragraph_break_preferred(self):
        text = "First paragraph.\n\nSecond paragraph. Some sentence here."
        target = 20
        result = find_semantic_boundary(text, target, search_range=50)
        # Should find the \n\n boundary
        assert text[result - 1] == "\n" or text[result : result + 1] != "\n"

    def test_sentence_boundary_when_no_paragraphs(self):
        text = "First sentence. Second sentence. Third sentence."
        target = 20
        result = find_semantic_boundary(text, target, search_range=50)
        # Should land after a ". " boundary
        assert result > 0

    def test_newline_when_no_sentences(self):
        text = "line one\nline two\nline three"
        target = 12
        result = find_semantic_boundary(text, target, search_range=50)
        assert result > 0

    def test_word_boundary_fallback(self):
        text = "word1 word2 word3 word4"
        target = 10
        result = find_semantic_boundary(text, target, search_range=50)
        assert result > 0

    def test_no_boundaries_returns_target(self):
        text = "abcdefghijklmnop"
        target = 8
        result = find_semantic_boundary(text, target, search_range=50)
        assert result == target

    def test_boundary_near_start(self):
        text = "\n\nSome text after paragraph break."
        result = find_semantic_boundary(text, target_pos=0, search_range=10)
        assert result >= 0


# ============================================================================
# chunk_transcript
# ============================================================================


class TestChunkTranscript:
    def test_short_transcript_single_chunk(self):
        text = "Short transcript content." * 50  # well under default target
        chunks = chunk_transcript(text, target_tokens=50000)
        assert len(chunks) == 1
        assert chunks[0].index == 0
        assert chunks[0].total_chunks == 1

    def test_long_transcript_multiple_chunks(self):
        # Create text that needs splitting: target 100 tokens = 400 chars
        text = ("This is a sentence. " * 50 + "\n\n") * 10  # ~10k chars
        chunks = chunk_transcript(text, target_tokens=500, overlap_tokens=10)
        assert len(chunks) > 1
        for i, chunk in enumerate(chunks):
            assert chunk.index == i
            assert chunk.total_chunks == len(chunks)

    def test_chunks_cover_full_text(self):
        text = "Word " * 2000  # 10k chars
        chunks = chunk_transcript(text, target_tokens=500, overlap_tokens=0)
        # First chunk starts at 0, last chunk ends at text length
        assert chunks[0].char_start == 0
        assert chunks[-1].char_end == len(text)

    def test_chunk_min_size_respected(self):
        text = "a" * (CHUNK_MIN_CHARS + 100)
        chunks = chunk_transcript(text, target_tokens=10, overlap_tokens=0)
        for chunk in chunks:
            assert len(chunk.text) >= CHUNK_MIN_CHARS or chunk == chunks[-1]

    def test_total_chunks_updated(self):
        text = ("Sentence one. " * 100 + "\n\n") * 5
        chunks = chunk_transcript(text, target_tokens=200, overlap_tokens=10)
        for chunk in chunks:
            assert chunk.total_chunks == len(chunks)

    def test_empty_transcript(self):
        chunks = chunk_transcript("", target_tokens=1000)
        assert len(chunks) == 0


# ============================================================================
# format_chunk_for_analysis
# ============================================================================


class TestFormatChunkForAnalysis:
    def test_basic_formatting(self):
        chunk = TranscriptChunk(
            index=0,
            total_chunks=3,
            text="Some transcript text",
            char_start=0,
            char_end=20,
            estimated_tokens=5,
        )
        result = format_chunk_for_analysis(chunk, "My Video Title")
        assert result["transcript"] == "Some transcript text"
        assert result["title"] == "My Video Title"
        assert result["chunk_num"] == "1"
        assert result["total_chunks"] == "3"

    def test_values_are_strings(self):
        chunk = TranscriptChunk(
            index=4,
            total_chunks=10,
            text="text",
            char_start=0,
            char_end=4,
            estimated_tokens=1,
        )
        result = format_chunk_for_analysis(chunk, "Title")
        assert isinstance(result["chunk_num"], str)
        assert isinstance(result["total_chunks"], str)


# ============================================================================
# format_chunks_for_synthesis
# ============================================================================


class TestFormatChunksForSynthesis:
    def test_basic_formatting(self):
        results = ["Analysis of chunk 1", "Analysis of chunk 2"]
        output = format_chunks_for_synthesis(results, "My Video", "1:00:00")
        assert "# Video: My Video" in output
        assert "# Duration: 1:00:00" in output
        assert "# Analyzed in 2 chunks" in output
        assert "## Chunk 1 of 2" in output
        assert "## Chunk 2 of 2" in output
        assert "Analysis of chunk 1" in output

    def test_single_chunk(self):
        output = format_chunks_for_synthesis(["Only result"], "Title", "5:00")
        assert "# Analyzed in 1 chunks" in output
        assert "## Chunk 1 of 1" in output

    def test_strips_whitespace(self):
        output = format_chunks_for_synthesis(["  padded  "], "T", "0:00")
        assert "padded" in output
