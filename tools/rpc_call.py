# SPDX-License-Identifier: MIT
"""Talk to hearth from the command line.

**This is the first tool to reach for when a generation fails**: it tells apart
a problem in the caller, in hearth, and in a runner, by cutting the caller out.

One request::

    .venv\\Scripts\\python.exe tools\\rpc_call.py status
    .venv\\Scripts\\python.exe tools\\rpc_call.py image_to_mesh --params-file args.json

**Several requests down one process**, which is the only way the loaded model
survives between them - starting hearth again throws the model away and pays the
load a second time::

    .venv\\Scripts\\python.exe tools\\rpc_call.py --flow flow.json
    .venv\\Scripts\\python.exe tools\\rpc_call.py --interactive

A flow file is a list of steps, and **each step's output is handed to the next**
(the image that came out becomes the image that goes in)::

    [{"method": "text_to_image", "params": {"prompt": "a small brass key"}},
     {"method": "image_to_mesh", "params": {"model": "trellis"}}]

In `--interactive`, each line is a method name and optionally its JSON
arguments; a blank line or EOF ends it::

    status
    load {"model": "trellis"}

**Do not pass JSON to `--params` from PowerShell.** It strips the double quotes
on the way to a native executable and the result is a `JSONDecodeError`.
**Use `--params-file`.**

Progress is drawn as it arrives and the final result is printed as JSON. The
exit code is 0 when everything succeeded and 2 when anything did not.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "client"))

from hearth_client import Flow, Hearth, RequestFailed  # noqa: E402

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
        self._started = time.perf_counter()

    def restart(self) -> None:
        """Start the clock again, for the next step of a flow."""
        self._started = time.perf_counter()

    def update(self, stage: str, message: str, step: int | None, total: int | None) -> None:
        """Show one progress event."""
        elapsed = time.perf_counter() - self._started
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


def _show(printer: ProgressLine, label: str, payload: dict[str, Any]) -> None:
    """Print one answer as JSON, under a label."""
    printer.close()
    print(f"{label}:", flush=True)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def _read_params(args: argparse.Namespace) -> dict[str, Any]:
    """Read the arguments for a single request.

    **`utf-8-sig`, not `utf-8`.** Windows PowerShell's `Set-Content -Encoding
    utf8` **always writes a BOM**, so an argument file written the obvious way
    dies with `Unexpected UTF-8 BOM`. Without a BOM, `utf-8-sig` reads exactly
    like `utf-8`.
    """
    raw = (
        Path(args.params_file).read_text(encoding="utf-8-sig")
        if args.params_file
        else args.params
    )
    return dict(json.loads(raw))


def _one(hearth: Hearth, printer: ProgressLine, method: str, params: dict[str, Any]) -> int:
    """Send one request and show what comes back."""
    printer.restart()
    try:
        result = hearth.call(method, params, on_progress=printer.update)
    except RequestFailed as exc:
        _show(printer, "error", {"type": exc.type, "message": exc.message})
        return 2
    _show(printer, "result", result)
    return 0


def _flow(hearth: Hearth, printer: ProgressLine, path: Path) -> int:
    """Run a list of steps down one process, feeding each into the next."""
    steps = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(steps, list):
        raise SystemExit("a flow file holds a list of steps")

    def on_step(index: int, method: str, result: dict[str, Any]) -> None:
        _show(printer, f"step {index + 1}/{len(steps)}  {method}", result)
        printer.restart()

    flow = Flow(hearth, steps)
    try:
        flow.run(on_progress=printer.update, on_step=on_step)
    except RequestFailed as exc:
        _show(printer, f"error in step {len(flow.results) + 1}",
              {"type": exc.type, "message": exc.message})
        return 2
    return 0


def _interactive(hearth: Hearth, printer: ProgressLine) -> int:
    """Read `method {json}` lines and run them down one process."""
    printer.note("one request per line: a method, then optional JSON. Blank line ends it.")
    worst = 0
    for raw in sys.stdin:
        line = raw.lstrip("﻿").strip()
        if not line:
            break
        method, _, rest = line.partition(" ")
        try:
            params = json.loads(rest) if rest.strip() else {}
        except ValueError as exc:
            printer.note(f"not JSON: {exc}")
            worst = 2
            continue
        worst = max(worst, _one(hearth, printer, method, dict(params)))
    return worst


def main() -> int:
    """Start hearth, send what was asked for, and show what comes back."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("method", nargs="?", help="the method to call: ping, status, ...")
    parser.add_argument(
        "--params", default="{}", help="arguments as JSON (PowerShell strips the quotes)"
    )
    parser.add_argument(
        "--params-file", default=None, help="a file holding the arguments (preferred)"
    )
    parser.add_argument(
        "--flow", default=None, help="a JSON file of steps to run down one hearth process"
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="read requests from stdin, keeping one hearth process for all of them",
    )
    parser.add_argument("--python", default=sys.executable, help="the python that runs hearth")
    args = parser.parse_args()

    if not args.method and not args.flow and not args.interactive:
        parser.error("give a method, --flow, or --interactive")

    printer = ProgressLine(sys.stdout)
    with Hearth.start(args.python, REPO_ROOT) as hearth:
        if args.flow:
            return _flow(hearth, printer, Path(args.flow))
        if args.interactive:
            return _interactive(hearth, printer)
        return _one(hearth, printer, args.method, _read_params(args))


if __name__ == "__main__":
    raise SystemExit(main())
