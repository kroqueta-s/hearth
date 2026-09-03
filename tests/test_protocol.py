# SPDX-License-Identifier: MIT
"""Hold a real conversation with hearth, and check `docs/protocol.md`.

**None of this needs a GPU, a runner, or ComfyUI.** Every claim checked here is
about the protocol itself, and the one that matters most is the one that is
easiest to break by accident:

- **Control methods are answered while the GPU queue is busy** (§2). A hearth
  that answers strictly in order looks fine in every test that sends one request
  at a time, and freezes a user interface the moment a generation starts.
- **GPU requests run in the order they arrived**, one at a time.
- Replies carry their `id`, an unknown method is an error rather than a crash,
  and `shutdown` ends it.

`selftest_long_job` is what makes this possible without hardware: it occupies the
GPU queue and reports counted progress, and uses no GPU at all.

Run it with hearth's own virtual environment::

    .venv\\Scripts\\python.exe .\\tests\\test_protocol.py
"""

from __future__ import annotations

import sys
import time
from collections.abc import Callable
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "client"))

from hearth_client import Hearth, RequestFailed  # noqa: E402

from hearth import config  # noqa: E402


def _hearth() -> Hearth:
    """Start hearth on the python running these tests."""
    return Hearth.start(sys.executable, REPO_ROOT)


def _wait(hearth: Hearth, done: Callable[[], bool], timeout: float = 60.0) -> None:
    """Poll until a condition holds, or give up.

    **Nothing here sleeps and hopes.** A test that waits without a deadline
    turns a broken build into a hung one.
    """
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        hearth.poll()
        if done():
            return
        time.sleep(0.02)
    raise AssertionError(f"gave up after {timeout}s")


def test_ping_reports_the_protocol_version() -> None:
    """§6: a caller checks this once, at startup."""
    with _hearth() as hearth:
        result = hearth.call("ping", timeout=60)
        assert result["ok"] is True, result
        assert result["role"] == "hearth", result
        assert result["protocol"] == config.PROTOCOL_VERSION, result


def test_status_starts_no_runner() -> None:
    """§4: `status` reports inventory and state, and starts nothing to do it.

    **`known` is what has already been asked for.** Asking every runner would
    mean starting every runner's python while a window opens.
    """
    with _hearth() as hearth:
        result = hearth.call("status", timeout=60)
        assert result["loaded"] is None, result
        assert result["busy"] is None, result
        assert isinstance(result["available"], list), result
        assert result["known"] == {}, "status started a runner it should not have"
        assert isinstance(result["image_models"], dict), result


def test_control_is_answered_while_the_gpu_queue_is_busy() -> None:
    """§2: **the claim the whole design rests on.**

    A `ping` sent after a long job must come back *before* it, or the interface
    that sent it has nothing to show a person for the length of a generation.
    """
    with _hearth() as hearth:
        answers: list[str] = []
        hearth.send(
            "selftest_long_job",
            {"seconds": 3, "interval": 0.25},
            done=lambda ok, payload: answers.append("job"),
        )
        # Sent second, and it must be answered first.
        time.sleep(0.3)
        hearth.send("ping", done=lambda ok, payload: answers.append("ping"))
        _wait(hearth, lambda: "ping" in answers, timeout=15)
        assert answers == ["ping"], f"the ping waited for the job: {answers}"
        _wait(hearth, lambda: "job" in answers, timeout=30)
        assert answers == ["ping", "job"], answers


def test_progress_is_counted_and_carries_a_total() -> None:
    """§5: a total that is real, so a caller may draw a percentage."""
    with _hearth() as hearth:
        seen: list[tuple[int | None, int | None]] = []
        hearth.call(
            "selftest_long_job",
            {"seconds": 1, "interval": 0.25},
            on_progress=lambda stage, message, step, total: seen.append((step, total)),
            timeout=30,
        )
        assert seen, "no progress arrived"
        assert all(step is not None and total is not None for step, total in seen), seen
        assert seen[0][0] == 1, seen
        assert seen[-1][0] == seen[-1][1], "the last step was not reported"


def test_gpu_requests_run_in_order() -> None:
    """§2: the GPU queue is strictly serial, in the order requests arrived."""
    with _hearth() as hearth:
        finished: list[int] = []
        for index in (1, 2):
            hearth.send(
                "selftest_long_job",
                {"seconds": 0.5, "interval": 0.25},
                done=lambda ok, payload, i=index: finished.append(i),
            )
        _wait(hearth, lambda: len(finished) == 2, timeout=30)
        assert finished == [1, 2], finished


def test_an_unknown_method_is_an_error_not_a_crash() -> None:
    """§6: the caller is told, and hearth stays up to answer the next one."""
    with _hearth() as hearth:
        try:
            hearth.call("no_such_method", timeout=30)
            raise AssertionError("it answered with a result")
        except RequestFailed as exc:
            assert exc.type == "ValueError", exc
        assert hearth.call("ping", timeout=30)["ok"] is True, "it did not survive"


def test_cancel_with_nothing_running_says_so() -> None:
    """§5: cancelling nothing is an answer, not an error."""
    with _hearth() as hearth:
        result = hearth.call("cancel", timeout=30)
        assert result["canceled"] is False, result
        assert result.get("why"), result


def test_out_dir_is_the_callers_to_choose() -> None:
    """§3: a caller can keep one piece of work in one directory.

    Asked for through `image_to_image` **with an input that is not there**, so it
    fails immediately and generates nothing - and the directory it was told to
    use is already there when it does. That ordering is the claim: the path a
    caller chose is honoured before anything expensive is attempted.
    """
    target = REPO_ROOT / "output" / "test_out_dir"
    with _hearth() as hearth:
        try:
            hearth.call(
                "image_to_image",
                {"prompt": "x", "image_path": str(REPO_ROOT / "no_such_image.png"),
                 "out_dir": str(target)},
                timeout=60,
            )
            raise AssertionError("it accepted an image that is not there")
        except RequestFailed as exc:
            # Either the missing file, or ComfyUI not running when it looked.
            assert exc.type in ("FileNotFoundError", "RuntimeError"), exc
    assert target.is_dir(), f"hearth did not use the out_dir it was given: {target}"
    target.rmdir()


def test_unknown_image_parameters_are_rejected() -> None:
    """§3.1: a misspelled setting fails loudly instead of running at its default."""
    with _hearth() as hearth:
        try:
            hearth.call("text_to_image", {"prompt": "x", "denoize": 0.5}, timeout=60)
            raise AssertionError("it accepted a parameter that does not exist")
        except RequestFailed as exc:
            assert exc.type == "ValueError", exc
            assert "denoize" in exc.message, exc.message


def test_shutdown_is_answered_and_ends_it() -> None:
    """§3: hearth answers, then exits."""
    hearth = _hearth()
    hearth.stop()
    assert not hearth.is_running(), "hearth is still running after shutdown"


def test_a_write_straight_to_fd_1_does_not_reach_the_protocol() -> None:
    """§1: **replacing `sys.stdout` is not enough.**

    A C extension writes to the file descriptor, past every Python object in
    the way: measured in a sibling repository, `pymeshfix` emitted `Loading
    ..0%` hundreds of times that way. One such line in the middle of the stream
    is a parse error the caller cannot explain and cannot recover from, so file
    descriptor 1 itself is pointed at stderr and the protocol keeps a duplicate.
    """
    with _hearth() as hearth:
        result = hearth.call(
            "selftest_long_job",
            {"seconds": 0.4, "interval": 0.1, "poison_fd1": True},
            timeout=60,
        )
        assert result["steps"] >= 1, result
        # **It has to keep working afterwards.** A guard that survives one line
        # and then leaves the stream misaligned is no guard at all.
        assert hearth.call("ping", timeout=30)["ok"] is True


def main() -> int:
    """Run every test."""
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"  OK   {t.__name__}")
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f"  FAIL {t.__name__}: {type(exc).__name__}: {exc}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
