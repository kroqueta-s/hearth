# SPDX-License-Identifier: MIT
r"""`compose_prompt` goes through hearth unchanged, and only where it is declared.

**The second method whose answer is not a mesh**, and the first whose answer is
words. Everything hearth does for it is what it does for `segment_mesh`: name the
runner, pass the rest through, relay the progress. What this file holds:

- **a runner is asked only when its capability table says it can be**, and a
  refusal names the method, never a model;
- **`text` and `format` reach the runner exactly as sent**: hearth does not
  translate, fill in or check a format - which image model it came from is the
  caller's business;
- **the answer arrives whole**, with no `mesh_path` and no axis invented;
- **`out_dir` is honoured and `model` is consumed**;
- **an argument the runner never declared is refused by the runner.**

`tests/fake_runner/` echoes the description back, so none of this needs a model.

Run it with hearth's own virtual environment::

    .venv\Scripts\python.exe .\tests\test_compose_prompt.py
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "tests"))

import test_segment_mesh as shared  # noqa: E402

#: Not ASCII, and built from code points so this file stays ASCII: the point is
#: that a description in any language reaches the runner byte for byte.
_TEXT = "".join(chr(c) for c in (0x93A7, 0x3092, 0x7740, 0x305F, 0x732B, 0x306E, 0x9A0E, 0x58EB))
_FORMAT = {"style": "tags", "negative": False, "negative_why": "cfg 1", "max_words": 60}


def _env(*, declares: bool = True) -> dict[str, str]:
    """The same sleeping runner, with `compose_prompt` declared or withdrawn."""
    return {**shared._env(), "SLEEPY_NO_COMPOSE": "0" if declares else "1"}  # noqa: SLF001


def test_capabilities_carries_the_declaration() -> None:
    """Contract §3: it is data, and it is how a caller knows the method exists at all."""
    events = shared._converse(  # noqa: SLF001
        [
            {"id": 1, "method": "capabilities", "params": {"model": "sleepy"}},
            {"id": 2, "method": "shutdown"},
        ],
        _env(),
    )
    table = shared._answers(events, 1)["result"]  # noqa: SLF001
    assert table["capabilities"]["compose_prompt"] is True, table
    assert "seed" in table["method_params"]["compose_prompt"], table


def test_the_words_arrive_whole_and_the_format_is_passed_through() -> None:
    """Unicode in, the caller's format untouched, and nothing mesh-shaped invented."""
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "run"
        events = shared._converse(  # noqa: SLF001
            [
                {
                    "id": 1,
                    "method": "compose_prompt",
                    "params": {
                        "model": "sleepy",
                        "text": _TEXT,
                        "format": _FORMAT,
                        "out_dir": str(out),
                        "seed": 5,
                    },
                },
                {"id": 2, "method": "shutdown"},
            ],
            _env(),
        )
        answer = shared._answers(events, 1)  # noqa: SLF001
        assert answer["event"] == "result", answer
        result = answer["result"]
        assert result["model"] == "sleepy", result
        assert result["run_dir"] == str(out), result
        assert result["prompt"] == f"echo: {_TEXT}", result
        assert result["format_used"] == _FORMAT, result
        assert result["negative"] == "", "the format said the model reads no negative"
        assert result["params_used"] == {"seed": 5}, result
        assert Path(result["record_path"]).is_file(), result
        assert "mesh_path" not in result and "up_axis" not in result, result


def test_a_runner_that_does_not_declare_it_is_refused_by_method_not_by_name() -> None:
    """Contract §2: hearth calls a method a table did not claim on no runner, ever."""
    with tempfile.TemporaryDirectory() as tmp:
        events = shared._converse(  # noqa: SLF001
            [
                {
                    "id": 1,
                    "method": "compose_prompt",
                    "params": {"model": "sleepy", "text": "a cat", "out_dir": tmp},
                },
                {"id": 2, "method": "shutdown"},
            ],
            _env(declares=False),
        )
        answer = shared._answers(events, 1)  # noqa: SLF001
        assert answer["event"] == "error", answer
        said = answer["error"]["message"]
        assert "compose_prompt" in said and "capabilities" in said, said


def test_an_undeclared_argument_is_refused_by_the_runner() -> None:
    """Contract §3: hearth validates nothing; the runner says what it does not accept."""
    with tempfile.TemporaryDirectory() as tmp:
        events = shared._converse(  # noqa: SLF001
            [
                {
                    "id": 1,
                    "method": "compose_prompt",
                    "params": {"model": "sleepy", "text": "a cat", "out_dir": tmp, "steps": 30},
                },
                {"id": 2, "method": "shutdown"},
            ],
            _env(),
        )
        answer = shared._answers(events, 1)  # noqa: SLF001
        assert answer["event"] == "error" and "steps" in answer["error"]["message"], answer


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
    sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main())
