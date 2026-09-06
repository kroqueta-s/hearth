# SPDX-License-Identifier: MIT
r"""`segment_mesh` goes through hearth unchanged, and only where it is declared.

**This is the first method whose answer is not a mesh.** Everything hearth does
for it is the same as for the others - name the runner, pass the rest through,
relay the progress - and the point of this file is that "the same" really is
enough, without a special case anywhere:

- **a runner is asked only when its capability table says it can be**
  (contract §2 and §3). A runner that does not declare `segment_mesh` is refused
  by name of the *method*, never by name of the model;
- **the runner's answer arrives whole.** No `mesh_path`, no `up_axis`, and
  hearth invents neither - `_contract_shape` used to say a runner was missing an
  axis whichever method it had answered, which made a correct runner look
  defective once per call;
- **`out_dir` is honoured and `model` is consumed**, exactly as for
  `texture_mesh`: the runner never sees `model`, and does see a directory;
- **an argument the runner never declared is refused by the runner**, not
  quietly dropped by hearth (contract §3: hearth validates nothing).

`tests/fake_runner/` provides all of it: a runner that declares the method and
answers with labels, without a model and without a graphics card.

Run it with hearth's own virtual environment::

    .venv\Scripts\python.exe .\tests\test_segment_mesh.py
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

#: A mesh only has to exist: the fake runner opens nothing.
_PLY = "ply\nformat ascii 1.0\nelement vertex 0\nend_header\n"


def _env(*, declares: bool = True) -> dict[str, str]:
    """One runner that sleeps, with or without the optional method declared.

    `HEARTH_LOCK_PORT=0` is not optional: the default is a real port, and a
    hearth started by the operator's Blender already holds it - the tests would
    then fail with `GpuBusyError` and say nothing about segmenting.
    """
    return {
        **os.environ,
        "HEARTH_RUNNERS": "sleepy",
        "HEARTH_RUNNER_SLEEPY_PYTHON": sys.executable,
        "HEARTH_RUNNER_SLEEPY_MODULE": "runners.sleepy",
        "HEARTH_RUNNER_SLEEPY_CWD": str(FAKE),
        "HEARTH_LOCK_PORT": "0",
        "HEARTH_GPU_BUSY_PORT": "0",
        "SLEEPY_LOAD_SEC": "0.05",
        # Turning the declaration off is how "only where it is declared" is
        # checked **without naming a model anywhere**.
        "SLEEPY_NO_SEGMENT": "0" if declares else "1",
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
        # **One at a time, waiting for each answer.** `shutdown` is a control
        # method and is answered while work is still queued (protocol §2), so
        # writing everything at once and reading to the end of the shutdown
        # reply would step straight over the answer this is about.
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


def test_capabilities_carries_the_declaration() -> None:
    """§3: it is data, and it is how a caller knows the method exists at all."""
    events = _converse(
        [
            {"id": 1, "method": "capabilities", "params": {"model": "sleepy"}},
            {"id": 2, "method": "shutdown"},
        ],
        _env(),
    )
    table = _answers(events, 1)["result"]
    assert table["capabilities"]["segment_mesh"] is True, table
    # **Its settings are its own**, not `image_to_mesh`'s (contract §3).
    assert "n_point_per_face" in table["method_params"]["segment_mesh"], table


def test_the_answer_arrives_whole_and_carries_no_axis() -> None:
    """The result is passed through: no mesh, no axis, nothing invented."""
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp)
        mesh = out / "in.ply"
        mesh.write_text(_PLY, encoding="ascii")
        events = _converse(
            [
                {
                    "id": 1,
                    "method": "segment_mesh",
                    "params": {
                        "model": "sleepy",
                        "mesh_path": str(mesh),
                        "out_dir": str(out / "run"),
                        "n_point_per_face": 200,
                    },
                },
                {"id": 2, "method": "shutdown"},
            ],
            _env(),
        )
        answer = _answers(events, 1)
        assert answer["event"] == "result", answer
        result = answer["result"]
        assert result["model"] == "sleepy", result
        assert result["run_dir"] == str(out / "run"), result
        assert result["source_mesh"] == str(mesh), result
        assert Path(result["segments_path"]).is_file(), result
        assert result["k_values"] == [2, 3, 4, 5], result
        assert len(result["faces_sha256"]) == 64, result
        # **Nothing was generated and nothing was moved**, so there is nothing
        # to orient. hearth must not fill one in - a wrong axis is invisible.
        assert "up_axis" not in result, result
        assert "mesh_path" not in result, result
        # The runner's own setting reached it, with the value that was used.
        assert result["params_used"]["n_point_per_face"] == 200, result


def test_a_runner_that_does_not_declare_it_is_refused_by_method_not_by_name() -> None:
    """§2: hearth calls a method a table did not claim on no runner, ever."""
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp)
        mesh = out / "in.ply"
        mesh.write_text(_PLY, encoding="ascii")
        events = _converse(
            [
                {
                    "id": 1,
                    "method": "segment_mesh",
                    "params": {"model": "sleepy", "mesh_path": str(mesh), "out_dir": str(out)},
                },
                {"id": 2, "method": "shutdown"},
            ],
            _env(declares=False),
        )
        answer = _answers(events, 1)
        assert answer["event"] == "error", answer
        said = answer["error"]["message"]
        assert "segment_mesh" in said and "capabilities" in said, said


def test_an_undeclared_argument_is_refused_by_the_runner() -> None:
    """§3: hearth validates nothing; the runner says what it does not accept."""
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp)
        mesh = out / "in.ply"
        mesh.write_text(_PLY, encoding="ascii")
        events = _converse(
            [
                {
                    "id": 1,
                    "method": "segment_mesh",
                    "params": {
                        "model": "sleepy",
                        "mesh_path": str(mesh),
                        "out_dir": str(out),
                        "octree_resolution": 384,
                    },
                },
                {"id": 2, "method": "shutdown"},
            ],
            _env(),
        )
        answer = _answers(events, 1)
        assert answer["event"] == "error", answer
        assert "octree_resolution" in answer["error"]["message"], answer


def main() -> int:
    """Run every test."""
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for test in tests:
        try:
            test()
            print(f"  OK   {test.__name__}")
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f"  FAIL {test.__name__}: {type(exc).__name__}: {exc}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
