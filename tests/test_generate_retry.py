# SPDX-License-Identifier: MIT
"""A runner that dies without answering is asked once more. **No GPU needed.**

On gfx1151 the driver takes the process away during a large decode: PAL fails
to submit the command, the errors are sticky, and torch's abort handler ends
the process. There is nothing for the runner to catch, because the runner is
gone - so the recovery has to be hearth's, and it is one retry.

What has to be true, and is checked here:

- A generating call whose runner died is tried again, and the answer comes back.
- The result says it took more than one attempt, because a caller comparing
  times against the measurements would otherwise see a load it cannot explain.
- **It gives up when the retries are spent** rather than looping.
- **An error the runner answered with is not retried.** The runner is still
  running and will say the same thing again; retrying only wastes a load.

`SLEEPY_DIE_TIMES` is the driver, standing in with `os._exit`: no traceback,
no `error` event, no flush.

Run it with hearth's own virtual environment::

    .venv\\Scripts\\python.exe .\\tests\\test_generate_retry.py
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
FAKE = REPO_ROOT / "tests" / "fake_runner"


def _env(*, deaths: int, retries: int, out_dir: Path, how: str = "") -> dict[str, str]:
    """One runner that sleeps, set to die a given number of times first.

    `HEARTH_LOCK_PORT=0` is not optional: the default is a real port and a
    hearth started by the operator's Blender already holds it, which would fail
    these with `GpuBusyError` and say nothing about retrying.
    """
    return {
        **os.environ,
        "HEARTH_RUNNERS": "sleepy",
        "HEARTH_RUNNER_SLEEPY_PYTHON": sys.executable,
        "HEARTH_RUNNER_SLEEPY_MODULE": "runners.sleepy",
        "HEARTH_RUNNER_SLEEPY_CWD": str(FAKE),
        "HEARTH_LOCK_PORT": "0",
        "HEARTH_GPU_BUSY_PORT": "0",
        "HEARTH_GENERATE_RETRIES": str(retries),
        "SLEEPY_LOAD_SEC": "0.05",
        "SLEEPY_DIE_TIMES": str(deaths),
        "SLEEPY_DIE_HOW": how,
        "PYTHONIOENCODING": "utf-8",
        "PYTHONUNBUFFERED": "1",
    }


def _converse(
    requests: list[dict[str, Any]], env: dict[str, str], timeout: float = 120.0
) -> list[dict[str, Any]]:
    """Send requests to a real hearth and collect every reply."""
    proc = subprocess.Popen(
        [sys.executable, "-m", "hearth"],
        cwd=str(REPO_ROOT),
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
        encoding="utf-8",
        bufsize=1,
        env=env,
    )
    assert proc.stdin is not None and proc.stdout is not None
    try:
        events: list[dict[str, Any]] = []
        deadline = time.monotonic() + timeout
        for request in requests:
            proc.stdin.write(json.dumps(request) + "\n")
            proc.stdin.flush()
            while time.monotonic() < deadline:
                line = proc.stdout.readline()
                if not line:
                    return events
                line = line.strip()
                if not line:
                    continue
                events.append(json.loads(line))
                if events[-1].get("id") == request["id"] and events[-1].get("event") in (
                    "result",
                    "error",
                ):
                    break
        return events
    finally:
        try:
            proc.stdin.close()
            proc.wait(timeout=10)
        except (OSError, ValueError, subprocess.TimeoutExpired):
            proc.kill()


def _answers(events: list[dict[str, Any]], request_id: int) -> dict[str, Any]:
    """The one `result` or `error` for a request."""
    for event in events:
        if event.get("id") == request_id and event.get("event") in ("result", "error"):
            return event
    raise AssertionError(f"nothing answered request {request_id}: {events}")


def _generate(deaths: int, retries: int) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Ask for a mesh from a runner set to die `deaths` times first."""
    answer, events, _kept = _generate_keeping(deaths, retries)
    return answer, events


def _generate_keeping(
    deaths: int, retries: int, how: str = ""
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, str]]:
    """The same, plus every stderr record hearth left in the run directory, by name."""
    with tempfile.TemporaryDirectory() as raw:
        out = Path(raw)
        image = out / "in.png"
        image.write_bytes(b"")
        events = _converse(
            [
                {
                    "id": 1,
                    "method": "image_to_mesh",
                    "params": {
                        "model": "sleepy",
                        "image_path": str(image),
                        "out_dir": str(out),
                        "seconds": 0.05,
                    },
                },
                {"id": 2, "method": "shutdown"},
            ],
            _env(deaths=deaths, retries=retries, out_dir=out, how=how),
        )
        kept = {p.name: p.read_text(encoding="utf-8") for p in out.glob("runner_stderr_*.txt")}
        return _answers(events, 1), events, kept


def test_a_runner_that_died_is_asked_once_more() -> None:
    """One death, one retry: the mesh comes back and the count says two."""
    answer, events = _generate(deaths=1, retries=1)
    assert answer.get("event") == "result", answer
    result = answer["result"]
    assert result["model"] == "sleepy", result
    assert result.get("attempts") == 2, result
    # **The caller is told while it happens**, not only afterwards.
    assert any(e.get("stage") == "retry" for e in events if e.get("event") == "progress"), events


def test_one_attempt_says_nothing_about_attempts() -> None:
    """A call that worked first time carries no count, because there is nothing to explain."""
    answer, _events = _generate(deaths=0, retries=1)
    assert answer.get("event") == "result", answer
    assert "attempts" not in answer["result"], answer["result"]


def test_it_gives_up_when_the_retries_are_spent() -> None:
    """Two deaths against one retry is an error, not a loop."""
    answer, _events = _generate(deaths=2, retries=1)
    assert answer.get("event") == "error", answer


#: What the pretend runner writes on its way out (`runners/sleepy/pipeline.py`).
DEATH_LINE = "sleepy: the driver took the process away (pretend)"


def test_each_death_leaves_its_stderr_beside_the_output() -> None:
    """**A death that was retried past still leaves a record**, named in the result.

    The runner's stderr was kept only in memory and only its last twenty lines
    reached anyone, so an abort that a retry covered left nothing at all - and
    a driver fault that happens one run in several cannot be studied from runs
    that did not keep it.
    """
    answer, _events, kept = _generate_keeping(deaths=1, retries=1)
    assert answer.get("event") == "result", answer
    logs = answer["result"].get("runner_logs") or []
    assert [Path(p).name for p in logs] == ["runner_stderr_1.txt"], answer["result"]
    record = kept.get("runner_stderr_1.txt", "")
    assert DEATH_LINE in record, record
    assert "exit code: 3 " in record, record


def test_an_abort_leaves_the_python_line_it_happened_on() -> None:
    """**An abort's record says where in Python it happened**, not only that it did.

    The driver's lines name the failure and torch's native stack resolves to
    the wrong symbols, so the one thing that places a death is Python's fault
    handler, which hearth turns on for every runner.
    """
    answer, _events, kept = _generate_keeping(deaths=1, retries=1, how="abort")
    assert answer.get("event") == "result", answer
    record = kept.get("runner_stderr_1.txt", "")
    assert "Fatal Python error: Aborted" in record, record
    assert "in image_to_mesh" in record, record


def test_a_death_that_is_the_answer_names_its_records() -> None:
    """When the retries are spent, the error says where every death's stderr is."""
    answer, _events, kept = _generate_keeping(deaths=2, retries=1)
    assert answer.get("event") == "error", answer
    assert sorted(kept) == ["runner_stderr_1.txt", "runner_stderr_2.txt"], sorted(kept)
    message = str(answer["error"].get("message", ""))
    assert "runner_stderr_1.txt" in message and "runner_stderr_2.txt" in message, message


def test_a_call_that_worked_keeps_no_record() -> None:
    """Nothing died, so nothing is written and nothing is named."""
    answer, _events, kept = _generate_keeping(deaths=0, retries=1)
    assert answer.get("event") == "result", answer
    assert not kept and "runner_logs" not in answer["result"], (kept, answer["result"])


def test_retries_can_be_turned_off() -> None:
    """At zero, the first death is the answer."""
    answer, _events = _generate(deaths=1, retries=0)
    assert answer.get("event") == "error", answer


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
