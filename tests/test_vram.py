# SPDX-License-Identifier: MIT
r"""Reading the card from outside. **No GPU library, and no waiting for one.**

`status` is polled by an interface while a generation runs, and it is answered on
the thread that reads stdin (`docs/protocol.md` §2). So the cost of asking how
full the card is has to be a memory read, not a measurement: `typeperf -sc 1`
takes **1,141 ms** on this machine, which would put more than a second between a
person pressing cancel and hearth reading it.

What is pinned here:

1. **A sample arrives**, with a dedicated figure that is not obviously nonsense.
2. **Asking is free.** The background thread pays; `status` reads what it left.
3. **A spill is judged per process**, and an unknown reading is never reported as
   a spill - that would abandon work over a counter that was not there.
4. **The `status.vram` shape is the one `docs/protocol.md` §4 promises.**

Run it with hearth's own virtual environment::

    .venv\Scripts\python.exe .\tests\test_vram.py
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

os.environ["HEARTH_VRAM_SAMPLE_SEC"] = "0.5"

from hearth import config, vram  # noqa: E402


def _sampled(sampler: vram.Sampler, seconds: float = 10.0) -> vram.Sample | None:
    """Wait for the first sample, or give up."""
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        sample = sampler.latest()
        if sample is not None:
            return sample
        time.sleep(0.05)
    return None


def test_a_sample_arrives_and_is_not_nonsense() -> None:
    """**Something is always on the card**: the desktop is drawn on it."""
    if not vram.available():
        print("       (skipped: these counters are Windows' own)")
        return
    sampler = vram.Sampler()
    sampler.start()
    try:
        sample = _sampled(sampler)
        assert sample is not None, "no VRAM sample arrived within ten seconds"
        assert (
            sample.dedicated_used_gb > 0.0
        ), f"the card reports {sample.dedicated_used_gb} GB in use, and a desktop uses some"
        assert sample.dedicated_used_gb < 1024.0, sample.dedicated_used_gb
        assert sample.shared_used_gb >= 0.0, sample.shared_used_gb
        assert sample.sampled_at > 0.0, "a sample has to say when it was taken"
    finally:
        sampler.stop()


def test_asking_costs_nothing() -> None:
    """**The whole reason there is a thread.**

    `status` must answer in milliseconds while a generation runs. The budget
    here is deliberately far below that: this call may not measure anything, it
    may only read what the sampler last left.
    """
    if not vram.available():
        print("       (skipped: these counters are Windows' own)")
        return
    sampler = vram.Sampler()
    sampler.start()
    try:
        assert _sampled(sampler) is not None, "no VRAM sample arrived"
        began = time.perf_counter()
        for _ in range(20):
            sampler.status()
        each_ms = (time.perf_counter() - began) * 1000 / 20
        assert (
            each_ms < 5.0
        ), f"reading the last sample took {each_ms:.2f} ms, which is a measurement"
    finally:
        sampler.stop()


def test_the_shape_is_the_one_the_protocol_promises() -> None:
    """`docs/protocol.md` §4. **A caller builds an interface out of these keys.**"""
    if not vram.available():
        print("       (skipped: these counters are Windows' own)")
        return
    sampler = vram.Sampler()
    # **This process, which certainly exists.** Watching it also proves that
    # `by_pid` reports what it was asked for and not the forty other processes
    # that touch the GPU on an ordinary desktop.
    sampler.watch(os.getpid(), "the test")
    sampler.start()
    try:
        assert _sampled(sampler) is not None, "no VRAM sample arrived"
        status = sampler.status()
        assert status is not None
        for key in (
            "dedicated_used_gb",
            "dedicated_total_gb",
            "shared_used_gb",
            "shared_abort_gb",
            "by_pid",
            "sampled_at",
        ):
            assert key in status, f"status.vram is missing {key}: {status}"
        assert status["shared_abort_gb"] == config.VRAM_SHARED_ABORT_GB
        assert set(status["by_pid"]) <= {
            str(os.getpid())
        }, f"by_pid reported processes nobody asked about: {status['by_pid']}"
        sampler.forget(os.getpid())
        assert sampler.status()["by_pid"] == {}, "a forgotten process was still reported"
    finally:
        sampler.stop()


def test_an_unknown_reading_is_not_a_spill() -> None:
    """**Never abandon work over a counter that was not there.**

    `spilled` is what ends a generation. Answering "yes" because there is no
    sample yet would kill the first generation after every start.
    """
    sampler = vram.Sampler()  # never started, so it has no sample
    assert sampler.spilled(os.getpid()) is None, "a missing sample was read as a spill"
    assert sampler.spilled(0) is None, "pid 0 is not a process"


def test_the_error_carries_the_numbers_and_not_only_a_sentence() -> None:
    """`docs/protocol.md` §6: a caller branches on the type and shows the numbers."""
    exc = vram.VramOverError("spilled", shared_gb=2.5, dedicated_gb=31.2, pid=1234)
    assert exc.details == {"shared_gb": 2.5, "dedicated_gb": 31.2, "pid": 1234}, exc.details
    assert str(exc) == "spilled"


def _holding(used: float, by_pid: dict[int, float], parents: dict[int, int]) -> vram.Holding:
    """A reading made up for the arithmetic, with made-up process names."""
    return vram.Holding(
        used_gb=used,
        by_pid=by_pid,
        names={pid: f"p{pid}.exe" for pid in by_pid},
        parents=parents,
    )


def test_a_generation_that_fits_is_not_short() -> None:
    """The desktop alone and a runner that fits: nothing to refuse."""
    holding = _holding(1.9, {10: 1.9}, {10: 1})
    short, others, holders = vram.shortfall(holding, need_gb=18.0, usable_gb=29.9, own_roots=())
    assert short <= 0, short
    assert abs(others - 1.9) < 1e-9, others
    assert holders == [{"pid": 10, "name": "p10.exe", "dedicated_gb": 1.9}], holders


def test_the_measured_abort_is_refused() -> None:
    """**The case that aborted on 2026-09-14**: 16 GB held elsewhere, an 18 GB runner."""
    holding = _holding(17.99, {10: 1.9, 20: 16.09}, {10: 1, 20: 1})
    short, others, holders = vram.shortfall(holding, need_gb=18.0, usable_gb=29.9, own_roots=())
    assert short > 0, short
    assert holders[0]["pid"] == 20, holders


def test_the_runner_s_own_memory_is_not_someone_else_s() -> None:
    """**A loaded runner holds its weights**, and they are already in its declared peak.

    The venv launcher's child is the process with the memory, so the whole family
    counts as the runner's.
    """
    holding = _holding(7.1, {10: 1.9, 30: 0.0, 31: 5.2}, {10: 1, 30: 2, 31: 30})
    short, others, holders = vram.shortfall(holding, need_gb=18.0, usable_gb=29.9, own_roots=(30,))
    assert abs(others - 1.9) < 1e-9, others
    assert short <= 0, short
    assert all(h["pid"] not in (30, 31) for h in holders), holders


def test_a_model_about_to_be_unloaded_is_not_someone_else_s() -> None:
    """**A switch is not refused for the memory the switch gives back.**

    The check runs before the previous model is unloaded. A runner still holding
    16 GB from the last request is one of hearth's own, and passed as such; an
    18 GB runner then fits. Counting it as "others" refused every switch between
    two large models.
    """
    holding = _holding(18.0, {10: 1.9, 40: 0.0, 41: 16.1}, {10: 1, 40: 2, 41: 40})
    refused, _others, _holders = vram.shortfall(holding, need_gb=18.0, usable_gb=29.9, own_roots=())
    assert refused > 0, "the arithmetic itself should see the 16 GB as others"
    short, others, holders = vram.shortfall(
        holding, need_gb=18.0, usable_gb=29.9, own_roots=(0, 40)
    )
    assert abs(others - 1.9) < 1e-9, others
    assert short <= 0, short
    assert all(h["pid"] not in (40, 41) for h in holders), holders


def test_the_card_can_be_read_whole() -> None:
    """On this platform a reading arrives, with names, and never less than nothing."""
    holding = vram.read_now()
    if not vram.available():
        assert holding is None
        return
    assert holding is not None and holding.used_gb > 0, holding
    assert holding.names, "no process names were read"


def test_the_short_error_carries_who_holds_the_card() -> None:
    """`docs/protocol.md` §6: the numbers and the holders travel, not only a sentence."""
    holders = [{"pid": 20, "name": "python.exe", "dedicated_gb": 16.09}]
    exc = vram.VramShortError(
        "no room", need_gb=18.0, others_gb=17.99, usable_gb=29.9, holders=holders
    )
    assert exc.details == {
        "need_gb": 18.0,
        "others_gb": 17.99,
        "usable_gb": 29.9,
        "holders": holders,
    }, exc.details


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
