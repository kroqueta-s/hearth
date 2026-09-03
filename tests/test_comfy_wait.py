# SPDX-License-Identifier: MIT
r"""Cancelling an image actually stops the wait. **Without ComfyUI, and fast.**

An image route waits on ComfyUI's history until the prompt appears, for up to
`COMFY_TIMEOUT_SEC` - half an hour by default. Cancelling has to end that wait,
and until it was wired it did not: `cancel` answered instantly with
`{"canceled": false, "why": "the prompt had already finished"}`, which was untrue
in every particular. The prompt was still running, nothing had finished, and the
caller went on waiting.

**Three things have to hold, and each was broken in a different way:**

1. The wait asks `should_stop` between polls, so a cancel ends it within a poll
   rather than at the timeout.
2. `cancel` learns which prompt to drop. The id does not exist when the work
   starts - the workflow has not been submitted yet - so it arrives afterwards,
   and recording it must not forget a cancel that arrived in between.
3. **A cancel that raced the submission still takes effect.** It found no id and
   could only set the flag; whoever creates the id has to act on it.

ComfyUI is replaced by a stand-in that answers `history` and `queue` over a
local socket, so this needs no ComfyUI, no models and no GPU.

Run it with hearth's own virtual environment::

    .venv\Scripts\python.exe .\tests\test_comfy_wait.py
"""

from __future__ import annotations

import json
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from hearth import comfy  # noqa: E402


class _Comfy:
    """A ComfyUI that never finishes anything, and says what it was asked."""

    def __init__(self) -> None:
        self.interrupted = False
        self.deleted: list[str] = []
        self.running = ["p1"]
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args: object) -> None:
                """Silent: the test's own output is the only thing worth reading."""

            def _send(self, payload: dict) -> None:
                body = json.dumps(payload).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self) -> None:  # noqa: N802 - the base class names it
                if self.path.startswith("/history/"):
                    # **Never finished.** That is the whole point: without a
                    # cancel this waits until the timeout.
                    self._send({})
                elif self.path == "/queue":
                    self._send(
                        {
                            "queue_running": [[0, p] for p in outer.running],
                            "queue_pending": [],
                        }
                    )
                else:
                    self._send({})

            def do_POST(self) -> None:  # noqa: N802 - the base class names it
                length = int(self.headers.get("Content-Length", 0))
                body = self.rfile.read(length) if length else b"{}"
                if self.path == "/interrupt":
                    outer.interrupted = True
                elif self.path == "/queue":
                    payload = json.loads(body or b"{}")
                    outer.deleted.extend(payload.get("delete", []))
                self._send({})

        self.server = HTTPServer(("127.0.0.1", 0), Handler)
        self.port = self.server.server_port
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    @property
    def url(self) -> str:
        """Where to point a client."""
        return f"http://127.0.0.1:{self.port}"

    def stop(self) -> None:
        """Shut the stand-in down."""
        self.server.shutdown()


def test_the_wait_ends_when_should_stop_says_so() -> None:
    """**Within a poll, not at the timeout.**

    The wait used to be the whole problem: half an hour of blocking with no way
    in. A cancel that takes effect in a second and a half is the difference
    between a button that works and one that lies.
    """
    fake = _Comfy()
    try:
        client = comfy.ComfyUIClient(base_url=fake.url)
        stop = False

        def should_stop() -> bool:
            return stop

        started = time.monotonic()

        # Ask it to stop half a second in, from another thread, the way `cancel`
        # does.
        def ask() -> None:
            nonlocal stop
            time.sleep(0.5)
            stop = True

        threading.Thread(target=ask, daemon=True).start()
        try:
            client.wait_for("p1", timeout_sec=30.0, poll_sec=0.2, should_stop=should_stop)
        except comfy.Interrupted:
            elapsed = time.monotonic() - started
            assert elapsed < 5.0, f"the wait took {elapsed:.1f}s to notice"
            return
        raise AssertionError("the wait ignored should_stop and returned normally")
    finally:
        fake.stop()


def test_a_running_prompt_is_interrupted_and_a_pending_one_is_deleted() -> None:
    """**Only this prompt.** Interrupting is global, so it is used sparingly.

    ComfyUI's `/interrupt` stops whatever is running, not a named prompt. So a
    prompt that is merely queued is deleted instead - interrupting would kill
    somebody else's job, which hearth has no business doing (§6).
    """
    fake = _Comfy()
    try:
        client = comfy.ComfyUIClient(base_url=fake.url)

        fake.running = ["p1"]
        assert client.cancel_prompt("p1") is True
        assert fake.interrupted is True, "a running prompt was not interrupted"
        assert fake.deleted == [], "a running prompt should not be deleted"

        fake.interrupted = False
        fake.running = ["someone_else"]
        assert client.cancel_prompt("p1") is False, "p1 was not running or pending"
        assert fake.interrupted is False, "another client's job was interrupted"
    finally:
        fake.stop()


def test_a_cancel_that_races_the_submission_is_not_forgotten() -> None:
    """**The window between starting and having an id to cancel.**

    `begin_external` runs before the workflow is submitted, because a caller
    asking `status` in that second deserves an answer. So `cancel` can arrive
    with no id to act on. It sets the flag; whoever creates the id has to check
    it, or the prompt runs on with nobody left to stop it.
    """
    from hearth import manager  # noqa: PLC0415 - imported here to keep it cheap

    board = manager.Manager()
    board.begin_external(f"{manager.EXTERNAL_PREFIX}test")

    # The cancel arrives first, with nothing to name.
    assert board.is_canceling() is False
    with board._lock:  # noqa: SLF001 - standing in for `cancel` without ComfyUI
        board._canceling = True  # noqa: SLF001

    # The submission finishes afterwards and finds the flag already up.
    assert board.note_external_prompt("p1") is True, (
        "the prompt id was recorded and the pending cancel was forgotten"
    )
    assert board.is_canceling() is True, "recording the id cleared the cancel"


def test_a_shutdown_is_not_forgotten_by_work_that_starts_a_moment_later() -> None:
    """**The same window as a racing cancel, one door along.**

    `_serve_gpu` refuses what is still queued once a shutdown begins, but a
    request that got past that check reaches `begin_external` a moment later.
    Starting fresh there would clear the cancelling flag that `shutdown` had
    just set - and the prompt would go to ComfyUI and keep running with nobody
    left to collect it.
    """
    from hearth import manager  # noqa: PLC0415 - imported here to keep it cheap

    board = manager.Manager()
    with board._lock:  # noqa: SLF001 - standing in for `shutdown` without runners
        board._shutting_down = True  # noqa: SLF001
        board._canceling = True  # noqa: SLF001

    board.begin_external(f"{manager.EXTERNAL_PREFIX}test")
    assert board.is_canceling() is True, (
        "work starting cleared a shutdown that was already under way"
    )


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
