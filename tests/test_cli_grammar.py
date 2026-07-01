"""Tests for the verb-first CLI grammar introduced in the UX refactor.

Pins the new public surface:
  - ``parse_arguments(argv)`` accepts an explicit argv list (no sys.argv reliance).
  - Every verb subparser sets ``args.command`` and ``args.func``.
  - The bare-input shortcut injects ``transcribe`` per a precise rule
    (``_inject_default_verb``): only when argv[0] is a non-empty token that is
    not a known verb and does not start with ``-``.
  - ``list <noun>`` parses the discovery nouns.
"""

import pytest

from pidcast import cli

# ---------------------------------------------------------------------------
# Bare-input shortcut (_inject_default_verb)
# ---------------------------------------------------------------------------


def test_bare_url_injects_transcribe():
    assert cli._inject_default_verb(["https://youtu.be/x"]) == [
        "transcribe",
        "https://youtu.be/x",
    ]


def test_bare_local_file_injects_transcribe():
    # A filename that is not a known verb is treated as transcribe input.
    assert cli._inject_default_verb(["episode.mp3"]) == ["transcribe", "episode.mp3"]


def test_known_verb_is_not_reinjected():
    assert cli._inject_default_verb(["transcribe", "x.mp3"]) == ["transcribe", "x.mp3"]
    assert cli._inject_default_verb(["lib", "list"]) == ["lib", "list"]
    assert cli._inject_default_verb(["analyze", "t.md"]) == ["analyze", "t.md"]


def test_verb_name_shadows_filename():
    # A file literally named like a verb is shadowed by the verb; documented
    # behavior is to use `transcribe ./info` explicitly.
    assert cli._inject_default_verb(["info"]) == ["info"]


def test_flag_first_is_not_injected():
    # argparse handles --help/-v on the root parser; never inject before a flag.
    assert cli._inject_default_verb(["--help"]) == ["--help"]
    assert cli._inject_default_verb(["-v"]) == ["-v"]


def test_empty_argv_is_unchanged():
    assert cli._inject_default_verb([]) == []


# ---------------------------------------------------------------------------
# Verb dispatch: every verb parses and carries command + func
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("argv", "command"),
    [
        (["transcribe", "x.mp3"], "transcribe"),
        (["analyze", "t.md"], "analyze"),
        (["diarize", "t.md"], "diarize"),
        (["setup"], "setup"),
        (["doctor"], "doctor"),
        (["resume"], "resume"),
        (["info"], "info"),
        (["list", "models"], "list"),
        (["lib", "list"], "lib"),
    ],
)
def test_verb_sets_command_and_func(argv, command):
    args = cli.parse_arguments(argv)
    assert args.command == command
    assert callable(args.func)


def test_transcribe_test_flag_maps_to_test_segment_dest():
    # `--test` is the new spelling; internal dest stays test_segment.
    args = cli.parse_arguments(["transcribe", "x.mp3", "--test", "5", "--start-at", "10"])
    assert args.test_segment == 5
    assert args.start_at == 10


def test_analyze_takes_transcript_positional():
    args = cli.parse_arguments(["analyze", "/path/to/t.md", "-a", "summary"])
    assert args.transcript == "/path/to/t.md"
    assert args.analysis_type == "summary"


def test_diarize_takes_transcript_and_audio():
    args = cli.parse_arguments(["diarize", "/path/to/t.md", "--audio", "/a.wav"])
    assert args.transcript == "/path/to/t.md"
    assert args.audio == "/a.wav"


@pytest.mark.parametrize(
    "noun",
    ["analyses", "models", "whisper-models", "presets", "profiles"],
)
def test_list_nouns_parse(noun):
    args = cli.parse_arguments(["list", noun])
    assert args.command == "list"
    assert args.thing == noun


def test_build_transcribe_namespace_yields_transcribe_defaults():
    # Resume relies on this to materialize a fully-defaulted transcribe Namespace.
    ns = cli.build_transcribe_namespace("/job/source.wav")
    assert ns.command == "transcribe"
    assert ns.input == "/job/source.wav"
    # A dest read as a bare attribute downstream must be present.
    assert hasattr(ns, "no_analyze")
    assert hasattr(ns, "save")
