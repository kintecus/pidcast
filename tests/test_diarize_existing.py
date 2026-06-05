"""Tests for audio resolution in --diarize-existing mode."""

from pidcast.workflow import _url_to_local_path, resolve_existing_audio


class TestUrlToLocalPath:
    def test_file_url_resolves_to_path(self):
        result = _url_to_local_path("file:///Users/x/Music/meeting.wav")
        assert result is not None
        assert str(result) == "/Users/x/Music/meeting.wav"

    def test_file_url_with_percent_encoding(self):
        result = _url_to_local_path("file:///Users/x/My%20Audio/clip.m4a")
        assert str(result) == "/Users/x/My Audio/clip.m4a"

    def test_bare_path(self):
        result = _url_to_local_path("/Users/x/Music/meeting.wav")
        assert str(result) == "/Users/x/Music/meeting.wav"

    def test_remote_url_returns_none(self):
        assert _url_to_local_path("https://youtube.com/watch?v=abc") is None

    def test_empty_returns_none(self):
        assert _url_to_local_path("") is None


class TestResolveExistingAudio:
    def test_resolves_from_front_matter_url_when_names_differ(self, tmp_path):
        # The real-world case: audio is smart-named, transcript is date-prefixed.
        transcript = tmp_path / "2026-06-05_meeting_1607.md"
        transcript.write_text("---\n")
        audio = tmp_path / "meeting-2026-06-05-1607.wav"
        audio.write_bytes(b"\x00")

        result = resolve_existing_audio(transcript, f"file://{audio}")
        assert result == audio

    def test_falls_back_to_single_sibling_audio(self, tmp_path):
        transcript = tmp_path / "transcript.md"
        transcript.write_text("---\n")
        audio = tmp_path / "some-other-name.m4a"
        audio.write_bytes(b"\x00")

        # No usable url, but exactly one audio file sits beside the transcript.
        result = resolve_existing_audio(transcript, "")
        assert result == audio

    def test_ambiguous_siblings_do_not_resolve(self, tmp_path):
        transcript = tmp_path / "transcript.md"
        transcript.write_text("---\n")
        (tmp_path / "a.wav").write_bytes(b"\x00")
        (tmp_path / "b.mp3").write_bytes(b"\x00")

        # Two audio files and no url -> ambiguous, caller must use --audio.
        assert resolve_existing_audio(transcript, "") is None

    def test_legacy_stem_swap_still_works(self, tmp_path):
        transcript = tmp_path / "episode.md"
        transcript.write_text("---\n")
        audio = tmp_path / "episode.wav"
        audio.write_bytes(b"\x00")
        # Add a second, differently-named audio so the sibling-glob would be
        # ambiguous — proving the stem-swap path is what resolves it.
        (tmp_path / "decoy.mp3").write_bytes(b"\x00")

        result = resolve_existing_audio(transcript, "")
        assert result == audio

    def test_front_matter_url_wins_over_sibling(self, tmp_path):
        transcript = tmp_path / "t.md"
        transcript.write_text("---\n")
        meta_audio = tmp_path / "from-meta.wav"
        meta_audio.write_bytes(b"\x00")
        # A stem-swap candidate also exists; the front-matter url should win.
        (tmp_path / "t.wav").write_bytes(b"\x00")

        result = resolve_existing_audio(transcript, f"file://{meta_audio}")
        assert result == meta_audio

    def test_nothing_found_returns_none(self, tmp_path):
        transcript = tmp_path / "t.md"
        transcript.write_text("---\n")
        assert resolve_existing_audio(transcript, "") is None

    def test_nonexistent_front_matter_url_falls_through(self, tmp_path):
        transcript = tmp_path / "t.md"
        transcript.write_text("---\n")
        audio = tmp_path / "real.wav"
        audio.write_bytes(b"\x00")

        # url points at a missing file; resolution should fall back to the
        # single sibling rather than returning the dead path.
        result = resolve_existing_audio(transcript, "file:///does/not/exist.wav")
        assert result == audio
