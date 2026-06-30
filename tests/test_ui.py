"""Tests for the single-owner run reporter (ui.RunReporter).

The reporter owns ONE rich Console/Live/Progress for a run so progress never
gets "pushed down" by interleaved prints. In a non-TTY environment (pipes,
pytest) it must degrade to plain logging and never start a Live region.
"""

import logging

from pidcast import ui


def test_active_reporter_is_none_by_default():
    assert ui.active_reporter() is None


def test_reporter_registers_and_clears_itself_as_active():
    reporter = ui.RunReporter(verbose=False)
    assert ui.active_reporter() is None
    with reporter:
        assert ui.active_reporter() is reporter
    # Cleared on exit, even though we never started a Live (non-TTY).
    assert ui.active_reporter() is None


def test_reporter_clears_active_on_exception():
    reporter = ui.RunReporter(verbose=False)
    try:
        with reporter:
            assert ui.active_reporter() is reporter
            raise RuntimeError("boom")
    except RuntimeError:
        pass
    assert ui.active_reporter() is None


def test_non_tty_does_not_start_live():
    # Under pytest stdout is not a TTY, so no Live should be created.
    reporter = ui.RunReporter(verbose=False)
    with reporter:
        assert reporter.is_live is False


def test_task_handle_updates_are_safe_without_live(caplog):
    # A task handle must be a no-op-safe object when there's no Live region,
    # so producers can always call task.update() unconditionally.
    reporter = ui.RunReporter(verbose=False)
    with reporter:
        task = reporter.task("Transcribing", total=1000)
        task.update(completed=500)
        task.update(completed=1000)
    # No exception == pass.


def test_log_emits_without_live(caplog):
    reporter = ui.RunReporter(verbose=False)
    with caplog.at_level(logging.INFO), reporter:
        reporter.log("downloaded 42 MB")
    assert "downloaded 42 MB" in caplog.text


def test_error_emits_without_live(caplog):
    reporter = ui.RunReporter(verbose=False)
    with caplog.at_level(logging.ERROR), reporter:
        reporter.error("it broke")
    assert "it broke" in caplog.text


def test_nested_reporter_does_not_clobber_outer_active():
    # A second reporter entering while one is active must restore the previous
    # active reporter on exit (so lib sync's per-episode reporters nest safely).
    outer = ui.RunReporter(verbose=False)
    inner = ui.RunReporter(verbose=False)
    with outer:
        assert ui.active_reporter() is outer
        with inner:
            assert ui.active_reporter() is inner
        assert ui.active_reporter() is outer
    assert ui.active_reporter() is None
