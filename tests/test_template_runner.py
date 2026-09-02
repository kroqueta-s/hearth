# SPDX-License-Identifier: MIT
"""Hold a real conversation with the runner template.

**A template nobody runs stops matching the contract without anyone noticing.**
So this starts `templates/runner/` as a child process and talks to it exactly as
hearth would, checking the parts of the contract that can be checked without a
model:

- `capabilities` is answered **without loading anything**, and looks like §3.
- An unknown method comes back as an `error`, **not as a crash**.
- **A failing method still answers.** The template's `image_to_mesh` raises
  `NotImplementedError` on purpose, and the point is that the caller is told so
  rather than left waiting.
- `shutdown` ends it.
- **Nothing but protocol reaches stdout**, which is what the stdout guard is for.

Run it with any python 3.12 that has python-dotenv (hearth's own works)::

    .venv\\Scripts\\python.exe .\\tests\\test_template_runner.py
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
TEMPLATE = REPO_ROOT / "templates" / "runner"


def _converse(requests: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Send requests to the template runner and return everything it wrote to stdout.

    Raises:
        AssertionError: If a line on stdout is not protocol.
    """
    proc = subprocess.run(
        [sys.executable, "-m", "runners.example"],
        cwd=str(TEMPLATE),
        input="".join(json.dumps(r) + "\n" for r in requests),
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=120,
    )
    events = []
    for raw in proc.stdout.splitlines():
        line = raw.strip()
        if not line:
            continue
        try:
            events.append(json.loads(line))
        except ValueError:  # noqa: PERF203
            raise AssertionError(
                f"a non-protocol line reached stdout: {line!r}\n"
                "the stdout guard is the only thing preventing this"
            ) from None
    return events


def test_capabilities_answers_without_loading() -> None:
    """§3: capabilities is data, and comes back without a model behind it."""
    events = _converse([{"id": 1, "method": "capabilities"}, {"id": 2, "method": "shutdown"}])
    assert events, "the runner said nothing"
    first = events[0]
    assert first["id"] == 1 and first["event"] == "result", first
    caps = first["result"]
    assert caps["name"], caps
    assert isinstance(caps["capabilities"], dict), caps
    assert caps["capabilities"]["image_to_mesh"] is True, caps
    assert isinstance(caps["params"], dict), caps
    for name, spec in caps["params"].items():
        assert "type" in spec and "default" in spec, (name, spec)


def test_capabilities_declares_a_contract_version() -> None:
    """§3: the version of the contract this runner was written against.

    **A caller never refuses a runner over this**; it uses it to say why
    something is unavailable instead of guessing.
    """
    events = _converse([{"id": 1, "method": "capabilities"}, {"id": 2, "method": "shutdown"}])
    caps = events[0]["result"]
    assert isinstance(caps.get("contract"), int), caps
    assert caps["contract"] >= 2, caps


def test_warm_answers_and_never_raises() -> None:
    """§9: `warm` reports what happened rather than failing.

    The template's weights directory is not there, which is the interesting
    case: hearth calls this **while another model is generating**, so a warm
    that cannot happen must come back as `warmed: false` and not as an error
    the caller has to handle.
    """
    events = _converse([{"id": 1, "method": "warm"}, {"id": 2, "method": "shutdown"}])
    assert events[0]["event"] == "result", events
    assert events[0]["result"]["warmed"] is False, events
    assert events[0]["result"].get("message"), "it did not say why"
    assert events[1]["result"] == {"bye": True}, "it did not survive to shut down"


def test_shutdown_is_answered_and_ends_it() -> None:
    """§2: shutdown replies once and the process exits."""
    events = _converse([{"id": 1, "method": "shutdown"}])
    assert events[-1]["event"] == "result", events
    assert events[-1]["result"] == {"bye": True}, events


def test_an_unknown_method_is_an_error_not_a_crash() -> None:
    """§6: the caller is told, and the runner stays up to answer the next one."""
    events = _converse(
        [
            {"id": 1, "method": "no_such_method"},
            {"id": 2, "method": "capabilities"},
            {"id": 3, "method": "shutdown"},
        ]
    )
    assert events[0]["event"] == "error", events
    assert events[0]["error"]["type"] == "ValueError", events
    # **It survived.** A runner that dies on a bad request takes the session with it.
    assert events[1]["id"] == 2 and events[1]["event"] == "result", events


def test_a_failing_method_still_answers() -> None:
    """§6: **no exception escapes.** The template raises on purpose; it must reply.

    A runner that dies here leaves its caller waiting forever, which is the
    failure this rule exists to prevent.
    """
    events = _converse(
        [
            {"id": 1, "method": "image_to_mesh", "params": {"image_path": "x", "out_dir": "y"}},
            {"id": 2, "method": "shutdown"},
        ]
    )
    assert events[0]["id"] == 1 and events[0]["event"] == "error", events
    assert events[0]["error"]["message"], events
    assert events[1]["result"] == {"bye": True}, "it did not survive to shut down"


def test_every_reply_carries_its_request_id() -> None:
    """§1: a reply is matched to a request by id, so it always has one."""
    events = _converse(
        [
            {"id": 7, "method": "capabilities"},
            {"id": 9, "method": "shutdown"},
        ]
    )
    assert [e["id"] for e in events] == [7, 9], events


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
