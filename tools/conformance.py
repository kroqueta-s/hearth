# SPDX-License-Identifier: MIT
"""Hold hearth's side of the conversation with a real runner, and check it.

`tests/test_template_runner.py` proves the **template** matches
`docs/runner_contract.md`. **Nothing proved that the runners actually installed
do**, and they are three separate repositories that change on their own. This
runs the same kind of conversation against whichever one is named:

    .venv\\Scripts\\python.exe tools\\conformance.py <runner>
    .venv\\Scripts\\python.exe tools\\conformance.py --all

It is **cheap and safe by default**: nothing here loads a model, generates
anything, or touches the GPU. Every check is something the contract says a
runner answers *without* a model behind it.

**This lives in hearth and speaks to runners from the outside**, which is what
keeps rule 3 intact: a runner never imports hearth, so its conformance cannot be
checked from inside it.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from hearth import config  # noqa: E402
from hearth.runner_client import RunnerError, RunnerProcess  # noqa: E402

# The version of `docs/runner_contract.md` this checks against.
CONTRACT_VERSION = 2


class Checks:
    """One runner's findings."""

    def __init__(self, name: str) -> None:
        self.name = name
        self.failed = 0
        self.warned = 0

    def ok(self, what: str, detail: str = "") -> None:
        print(f"  ok    {what}{f': {detail}' if detail else ''}")

    def warn(self, what: str, detail: str) -> None:
        self.warned += 1
        print(f"  warn  {what}: {detail}")

    def fail(self, what: str, detail: str) -> None:
        self.failed += 1
        print(f"  FAIL  {what}: {detail}")

    def expect(self, condition: bool, what: str, detail: str) -> bool:
        if condition:
            self.ok(what)
        else:
            self.fail(what, detail)
        return condition


def _check_capabilities(checks: Checks, caps: dict[str, Any]) -> None:
    """§3: capabilities is data, and it is the whole basis of not branching on names."""
    checks.expect(bool(caps.get("name")), "declares a name", "no `name` field")

    contract = caps.get("contract", 1)
    if not isinstance(contract, int):
        checks.fail("declares a contract version", f"`contract` is {contract!r}, not an integer")
    elif contract < CONTRACT_VERSION:
        # **Not a failure.** An older runner missing a newer optional method is
        # exactly the case capabilities already describes (§3).
        checks.warn(
            "contract version",
            f"written against {contract}, this hearth speaks {CONTRACT_VERSION}",
        )
    else:
        checks.ok("contract version", str(contract))

    able = caps.get("capabilities")
    if not checks.expect(isinstance(able, dict), "capabilities is a table", "not a dict"):
        return
    assert isinstance(able, dict)
    non_bool = {k: v for k, v in able.items() if not isinstance(v, bool)}
    checks.expect(not non_bool, "every capability is true or false", f"these are not: {non_bool}")
    checks.expect(
        bool(able.get("image_to_mesh")),
        "declares image_to_mesh",
        "§2 makes it required",
    )

    params = caps.get("params")
    if not checks.expect(isinstance(params, dict), "params is a table", "not a dict"):
        return
    assert isinstance(params, dict)
    if not params:
        checks.warn("params", "declares none, so no caller can build a form for this runner")
    for key, spec in params.items():
        if not isinstance(spec, dict) or "type" not in spec or "default" not in spec:
            checks.fail(f"param {key}", "needs at least `type` and `default` (§3)")


def check(name: str) -> Checks:
    """Run the conversation against one runner.

    Args:
        name: A runner declared in `.env`.

    Returns:
        What was found.
    """
    print(f"\n{name}")
    checks = Checks(name)
    spec = config.runner_spec(name)
    if not Path(spec["python"]).is_file():
        checks.fail("starts", f"python not found: {spec['python']}")
        return checks

    runner = RunnerProcess(name, spec)
    try:
        started = time.perf_counter()
        runner.start()
        caps = runner.call("capabilities")
        # §2: **answered without loading the model.** Nothing here proves no
        # weights were touched, but a runner that loads first cannot answer in
        # this kind of time, so an outlier is worth looking at.
        elapsed = time.perf_counter() - started
        checks.ok("answers capabilities", f"{elapsed:.1f}s from a cold start")
        if elapsed > 60:
            checks.warn("capabilities", f"took {elapsed:.0f}s: is it loading the model to answer?")
        _check_capabilities(checks, caps)

        # §6: an unknown method is an error, and it survives to answer again.
        try:
            runner.call("no_such_method_ok")
            checks.fail("rejects an unknown method", "it answered with a result")
        except RunnerError as exc:
            checks.expect(
                "ValueError" in str(exc), "rejects an unknown method", f"as {exc}"
            )
        try:
            runner.call("capabilities")
            checks.ok("survives a bad request")
        except RunnerError as exc:
            checks.fail("survives a bad request", str(exc))

        stderr = runner.stderr_tail(400)
        if "[protocol] unparsable line" in stderr:
            checks.fail("stdout is protocol only", "something printed past the stdout guard (§1)")
        else:
            checks.ok("stdout is protocol only")
    except (RunnerError, OSError) as exc:
        checks.fail("starts", str(exc))
    finally:
        runner.stop()
    return checks


def main() -> int:
    """Check the runners that were named, or all of them."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runners", nargs="*", help="which runners to check")
    parser.add_argument("--all", action="store_true", help="check every runner in .env")
    args = parser.parse_args()

    names = config.runner_names() if args.all or not args.runners else args.runners
    if not names:
        print("no runners to check (HEARTH_RUNNERS is empty)")
        return 0

    print(f"runner contract v{CONTRACT_VERSION} - checking {', '.join(names)}")
    results = [check(name) for name in names]
    failed = sum(c.failed for c in results)
    warned = sum(c.warned for c in results)
    print(f"\n{failed} failed, {warned} to look at, across {len(results)} runners")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
