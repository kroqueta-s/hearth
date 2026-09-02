# SPDX-License-Identifier: MIT
"""Send one request to hearth from the command line.

**This is the first tool to reach for when a generation fails**: it tells apart
a problem in the caller, in hearth, and in a runner, by cutting the caller out.

    .venv\\Scripts\\python.exe tools\\rpc_call.py ping
    .venv\\Scripts\\python.exe tools\\rpc_call.py status
    .venv\\Scripts\\python.exe tools\\rpc_call.py image_to_mesh --params-file args.json

**Do not pass JSON to `--params` from PowerShell.** It strips the double quotes
on the way to a native executable and the result is a `JSONDecodeError`.
**Use `--params-file`.**

Progress is drawn as it arrives, and the final result or error is printed as
JSON. The exit code is 0 on success and 2 on failure.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent


BAR_WIDTH = 24


class ProgressLine:
    """Draw the runner's progress: a bar on a terminal, plain lines elsewhere.

    The runner reports `step`, and `total` as well whenever the length of the
    loop is known. **A percentage is shown only when a total arrived** - nothing
    here estimates, and there is deliberately no ETA (on this hardware the same
    loop ran 167 s per step on its first run and 14.7 s afterwards).

    Bars are drawn with `#` and `-` on purpose: the console code page on a
    Japanese Windows install cannot encode the block-drawing characters, and a
    progress bar must never be the thing that raises.
    """

    def __init__(self, stream: Any) -> None:
        self._stream = stream
        self._tty = bool(getattr(stream, "isatty", lambda: False)())
        self._active = ""
        self._last_key: tuple[str, int] | None = None

    def update(
        self,
        elapsed: float,
        stage: str,
        message: str,
        step: int | None,
        total: int | None,
    ) -> None:
        """Show one progress event."""
        if step is None:
            self.note(f"[{elapsed:7.1f}s] {stage}: {message}")
            return
        if total:
            percent = min(100, int(100 * step / total))
            filled = round(BAR_WIDTH * step / total)
            bar = "#" * filled + "-" * (BAR_WIDTH - filled)
            line = f"[{elapsed:7.1f}s] {stage:<10} [{bar}] {percent:3d}%  ({step}/{total})"
            key = (stage, percent // 5)
        else:
            # No total: report the count and nothing more. **Never guess one.**
            line = f"[{elapsed:7.1f}s] {stage:<10} step {step}"
            key = (stage, step // 10)
        if self._tty:
            self._draw(line)
            if total and step >= total:
                self._end_line()
            return
        # Redirected output (a log, or CI): one line per bucket, so a long loop
        # cannot bury everything else.
        if key != self._last_key:
            self._last_key = key
            print(line, file=self._stream, flush=True)

    def note(self, text: str) -> None:
        """Print a line that is not a bar, without leaving a half-drawn bar behind."""
        if self._tty and self._active:
            self._clear()
        print(text, file=self._stream, flush=True)
        if self._tty and self._active:
            self._draw(self._active)

    def close(self) -> None:
        """Finish the current bar, if one is on screen."""
        if self._tty and self._active:
            self._end_line()

    def _draw(self, line: str) -> None:
        pad = max(0, len(self._active) - len(line))
        self._stream.write("\r" + line + " " * pad)
        self._stream.flush()
        self._active = line

    def _clear(self) -> None:
        self._stream.write("\r" + " " * len(self._active) + "\r")
        self._stream.flush()

    def _end_line(self) -> None:
        self._stream.write("\n")
        self._stream.flush()
        self._active = ""
        self._last_key = None


def main() -> int:
    """Start hearth, send one request, and show what comes back."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("method", help="the method to call: ping, status, load, image_to_mesh, ...")
    parser.add_argument(
        "--params", default="{}", help="arguments as JSON (PowerShell strips the quotes)"
    )
    parser.add_argument("--params-file", default=None, help="a file holding the arguments (preferred)")
    parser.add_argument("--python", default=sys.executable, help="the python that runs hearth")
    args = parser.parse_args()

    proc = subprocess.Popen(
        [args.python, "-m", "hearth"],
        cwd=str(REPO_ROOT),
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        bufsize=1,
    )
    assert proc.stdin is not None and proc.stdout is not None

    # **`utf-8-sig`, not `utf-8`.** Windows PowerShell's
    # `Set-Content -Encoding utf8` **always writes a BOM**, so an argument file
    # written the obvious way dies with `Unexpected UTF-8 BOM`. Without a BOM,
    # `utf-8-sig` reads exactly like `utf-8`.
    raw = (
        Path(args.params_file).read_text(encoding="utf-8-sig")
        if args.params_file
        else args.params
    )
    request = {"id": 1, "method": args.method, "params": json.loads(raw)}
    printer = ProgressLine(sys.stdout)
    started = time.perf_counter()
    proc.stdin.write(json.dumps(request, ensure_ascii=False) + "\n")
    proc.stdin.flush()

    exit_code = 2
    for raw in proc.stdout:
        line = raw.strip()
        if not line:
            continue
        try:
            event = json.loads(line)
        except ValueError:
            printer.note(f"[non-protocol line] {line}")
            continue
        elapsed = time.perf_counter() - started
        kind = event.get("event")
        if kind == "progress":
            printer.update(
                elapsed,
                str(event.get("stage", "")),
                str(event.get("message", "")),
                event.get("step"),
                event.get("total"),
            )
        elif kind == "result":
            printer.close()
            print(f"[{elapsed:7.1f}s] result:", flush=True)
            print(json.dumps(event["result"], ensure_ascii=False, indent=2))
            exit_code = 0
            break
        elif kind == "error":
            printer.close()
            print(f"[{elapsed:7.1f}s] error: {json.dumps(event['error'], ensure_ascii=False)}")
            break

    proc.stdin.write(json.dumps({"id": 2, "method": "shutdown"}) + "\n")
    proc.stdin.flush()
    proc.stdin.close()
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
