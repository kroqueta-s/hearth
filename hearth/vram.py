# SPDX-License-Identifier: MIT
r"""How much of the card is actually in use, read from Windows' own counters.

**hearth holds no torch** (`CLAUDE.md`), so it has never been able to say how
full the GPU is - and the one number that was available, ComfyUI's
`/system_stats`, is wrong in the direction that matters: measured on this
machine on 2026-09-06 it reported 44,921 MB of total VRAM and 16,803 MB free
while Windows reported 31,309 MB of 32,768 MB dedicated already in use.
`torch.cuda.mem_get_info`, which is where that figure comes from, counts the
shared pool - system RAM the driver may spill into - as if it were video memory.
An application that believes it loads more weights, the driver spills, and
**nothing fails**: it just becomes several times slower.

So the truth is read from outside, through the performance counters Windows
keeps for every adapter and every process::

    \GPU Adapter Memory(<luid>)\Dedicated Usage
    \GPU Adapter Memory(<luid>)\Shared Usage
    \GPU Process Memory(pid_<pid>_<luid>)\Dedicated Usage
    \GPU Process Memory(pid_<pid>_<luid>)\Shared Usage

**Dependency-free**: `pdh.dll` ships with Windows and is reached through
`ctypes`. Nothing is added to any virtual environment, and no GPU library is
imported.

**Shared usage rising is the spill itself**, per process, which is what makes
this usable as more than a display: ComfyUI and a runner are measured with the
same ruler, and the one that is spilling is named.

## Why a background thread

`typeperf -sc 1` takes **1,141 ms** on this machine (measured 2026-09-06), which
is far too slow for a control method that must answer in milliseconds
(`docs/protocol.md` §2). A PDH query that is opened once and kept open costs
**0.1-0.2 ms** per sample instead (measured 2026-09-06, same machine), so one
thread samples on a timer and `status` returns the last sample it took. That
also makes the cost of asking constant however often a caller asks.

**Windows only.** Everywhere else `sample()` returns None and the callers treat
VRAM as unknown rather than as zero.
"""

from __future__ import annotations

import ctypes
import sys
import threading
import time
from ctypes import wintypes
from dataclasses import dataclass, field
from typing import Any

from . import config

_GB = 1024.0**3

# --- The error a spill raises -------------------------------------------------


class VramOverError(RuntimeError):
    """A process spilled out of dedicated VRAM into shared (system) memory.

    **Raised rather than waited through.** When the dedicated pool is full the
    driver does not fail: it pages into system memory and the work carries on.
    Measured on this machine on 2026-09-06, FLUX at 2048x2048 put ComfyUI at
    29.0 GB of dedicated VRAM and **1.1 GB of shared** on a 32 GB card.

    **How much slower a spill makes it was not measured here** - the run was
    ended by this check at 34 seconds - and it is worth being careful about that
    number, because the obvious evidence for it turned out to be something else.
    A step time of 3.8 s was read as the moment of a spill on this machine, and
    it is not: FLUX at 1024x1024 runs at 3.79-3.94 s/step **whether it spills or
    not** (measured 2026-09-06, both ways). What is true without measuring a
    slowdown is that the memory a spill lands in is the same RAM every other
    process wants - and this machine has 31.6 GB of it, because the other 32 went
    to the card.
    """

    def __init__(self, message: str, *, shared_gb: float, dedicated_gb: float, pid: int) -> None:
        super().__init__(message)
        self.shared_gb = shared_gb
        self.dedicated_gb = dedicated_gb
        self.pid = pid

    @property
    def details(self) -> dict[str, Any]:
        """Extra fields for the error line (`docs/protocol.md` §6)."""
        return {"shared_gb": self.shared_gb, "dedicated_gb": self.dedicated_gb, "pid": self.pid}


class VramShortError(RuntimeError):
    """A generation was not started, because the card does not have room for it.

    **Refused before the load, not discovered after it.** When every process
    together reaches what the card can hold, the driver either spills into
    shared memory or fails the runner's command submit outright, and the
    second ends the runner's process minutes into the work (measured
    2026-09-14). A runner that declares what it needs (`vram_peak_gb`, runner
    contract §3) is therefore not started while others hold the room, and the
    error says who holds it.
    """

    def __init__(
        self,
        message: str,
        *,
        need_gb: float,
        others_gb: float,
        usable_gb: float,
        holders: list[dict[str, Any]],
    ) -> None:
        super().__init__(message)
        self.need_gb = need_gb
        self.others_gb = others_gb
        self.usable_gb = usable_gb
        self.holders = holders

    @property
    def details(self) -> dict[str, Any]:
        """Extra fields for the error line (`docs/protocol.md` §6)."""
        return {
            "need_gb": self.need_gb,
            "others_gb": self.others_gb,
            "usable_gb": self.usable_gb,
            "holders": self.holders,
        }


# --- PDH, through ctypes ------------------------------------------------------

_PDH_FMT_LARGE = 0x00000400
_PDH_MORE_DATA = 0x800007D2

_ADAPTER_DEDICATED = r"\GPU Adapter Memory(*)\Dedicated Usage"
_ADAPTER_SHARED = r"\GPU Adapter Memory(*)\Shared Usage"
_PROCESS_DEDICATED = r"\GPU Process Memory(*)\Dedicated Usage"
_PROCESS_SHARED = r"\GPU Process Memory(*)\Shared Usage"
_PATHS = (_ADAPTER_DEDICATED, _ADAPTER_SHARED, _PROCESS_DEDICATED, _PROCESS_SHARED)


class _CounterValue(ctypes.Structure):
    _fields_ = [("CStatus", wintypes.DWORD), ("largeValue", ctypes.c_longlong)]


class _CounterItem(ctypes.Structure):
    _fields_ = [("szName", ctypes.c_wchar_p), ("FmtValue", _CounterValue)]


class _ProcessEntry(ctypes.Structure):
    """PROCESSENTRY32W, for `pid -> parent pid` without a dependency."""

    _fields_ = [
        ("dwSize", wintypes.DWORD),
        ("cntUsage", wintypes.DWORD),
        ("th32ProcessID", wintypes.DWORD),
        ("th32DefaultHeapID", ctypes.POINTER(ctypes.c_ulong)),
        ("th32ModuleID", wintypes.DWORD),
        ("cntThreads", wintypes.DWORD),
        ("th32ParentProcessID", wintypes.DWORD),
        ("pcPriClassBase", ctypes.c_long),
        ("dwFlags", wintypes.DWORD),
        ("szExeFile", ctypes.c_wchar * 260),
    ]


_TH32CS_SNAPPROCESS = 0x00000002


def available() -> bool:
    """Whether these counters can be read at all. **Windows only.**"""
    return sys.platform == "win32"


def _process_parents() -> dict[int, int]:
    """Every running process's parent, as one snapshot (`_process_table`)."""
    return {pid: parent for pid, (parent, _name) in _process_table().items()}


def _process_table() -> dict[int, tuple[int, str]]:
    """Every running process's parent and executable name, as one snapshot.

    **A venv's `python.exe` re-executes the base interpreter**, so the process
    holding the VRAM is a *child* of the one hearth started - measured
    2026-09-03 and already relied on in `runner_client`. Asking about the pid
    hearth knows would therefore report almost nothing. Costs 7-10 ms on this
    machine (measured 2026-09-06), which is why it happens on the sampling
    thread and never in a control method.
    """
    kernel32 = ctypes.WinDLL("kernel32.dll", use_last_error=True)
    kernel32.CreateToolhelp32Snapshot.restype = wintypes.HANDLE
    kernel32.Process32FirstW.argtypes = [wintypes.HANDLE, ctypes.POINTER(_ProcessEntry)]
    kernel32.Process32NextW.argtypes = [wintypes.HANDLE, ctypes.POINTER(_ProcessEntry)]
    snapshot = kernel32.CreateToolhelp32Snapshot(_TH32CS_SNAPPROCESS, 0)
    if snapshot == ctypes.c_void_p(-1).value:
        return {}
    entry = _ProcessEntry()
    entry.dwSize = ctypes.sizeof(_ProcessEntry)
    table: dict[int, tuple[int, str]] = {}
    try:
        ok = kernel32.Process32FirstW(snapshot, ctypes.byref(entry))
        while ok:
            table[int(entry.th32ProcessID)] = (int(entry.th32ParentProcessID), entry.szExeFile)
            ok = kernel32.Process32NextW(snapshot, ctypes.byref(entry))
    finally:
        kernel32.CloseHandle(snapshot)
    return table


def _family(root: int, parents: dict[int, int]) -> set[int]:
    """`root` and every process descended from it."""
    children: dict[int, list[int]] = {}
    for pid, parent in parents.items():
        children.setdefault(parent, []).append(pid)
    seen = {root}
    stack = [root]
    while stack:
        for child in children.get(stack.pop(), ()):
            if child not in seen:
                seen.add(child)
                stack.append(child)
    return seen


@dataclass
class Sample:
    """One reading of the counters."""

    dedicated_used_gb: float
    dedicated_total_gb: float
    shared_used_gb: float
    by_pid: dict[int, dict[str, float]] = field(default_factory=dict)
    sampled_at: float = 0.0

    def as_dict(self, *, shared_abort_gb: float, watched: dict[int, str]) -> dict[str, Any]:
        """The `status.vram` shape (`docs/protocol.md` §4).

        Args:
            shared_abort_gb: The threshold a caller should draw a line at.
            watched: The pids worth reporting, and what each one is. **Not all
                of them**: forty processes touch the GPU on an ordinary desktop
                and none of the other thirty-odd is hearth's business.
        """
        return {
            "dedicated_used_gb": round(self.dedicated_used_gb, 2),
            "dedicated_total_gb": round(self.dedicated_total_gb, 2),
            "shared_used_gb": round(self.shared_used_gb, 2),
            "shared_abort_gb": shared_abort_gb,
            "by_pid": {
                str(pid): {
                    "dedicated_gb": round(self.by_pid[pid]["dedicated_gb"], 2),
                    "shared_gb": round(self.by_pid[pid]["shared_gb"], 2),
                    "what": what,
                }
                for pid, what in watched.items()
                if pid in self.by_pid
            },
            "sampled_at": round(self.sampled_at, 3),
        }


class _Query:
    """One open PDH query. **Belongs to the thread that created it.**"""

    def __init__(self) -> None:
        self._pdh = ctypes.WinDLL("pdh.dll")
        self._pdh.PdhOpenQueryW.argtypes = [
            ctypes.c_wchar_p,
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_void_p),
        ]
        self._pdh.PdhAddEnglishCounterW.argtypes = [
            ctypes.c_void_p,
            ctypes.c_wchar_p,
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_void_p),
        ]
        self._pdh.PdhCollectQueryData.argtypes = [ctypes.c_void_p]
        self._pdh.PdhGetFormattedCounterArrayW.argtypes = [
            ctypes.c_void_p,
            wintypes.DWORD,
            ctypes.POINTER(wintypes.DWORD),
            ctypes.POINTER(wintypes.DWORD),
            ctypes.c_void_p,
        ]
        self._handle = ctypes.c_void_p()
        if self._pdh.PdhOpenQueryW(None, None, ctypes.byref(self._handle)) != 0:
            raise OSError("PdhOpenQuery failed")
        self._counters: dict[str, ctypes.c_void_p] = {}
        # The buffer PDH asked for last time. Reusing the size avoids the
        # "ask, be told it is too small, ask again" round trip on every sample.
        self._sizes: dict[str, int] = {}
        for path in _PATHS:
            counter = ctypes.c_void_p()
            added = self._pdh.PdhAddEnglishCounterW(self._handle, path, None, ctypes.byref(counter))
            if added != 0:
                raise OSError(f"PdhAddEnglishCounter failed for {path}")
            self._counters[path] = counter
            self._sizes[path] = 0

    def collect(self) -> dict[str, dict[str, int]]:
        """Take one sample of every counter. Returns bytes, per instance name."""
        self._pdh.PdhCollectQueryData(self._handle)
        return {path: self._read(path) for path in _PATHS}

    def _read(self, path: str) -> dict[str, int]:
        size = wintypes.DWORD(self._sizes[path])
        count = wintypes.DWORD(0)
        buffer = ctypes.create_string_buffer(size.value) if size.value else None
        code = self._pdh.PdhGetFormattedCounterArrayW(
            self._counters[path], _PDH_FMT_LARGE, ctypes.byref(size), ctypes.byref(count), buffer
        )
        if (code & 0xFFFFFFFF) == _PDH_MORE_DATA:
            self._sizes[path] = size.value
            buffer = ctypes.create_string_buffer(size.value)
            code = self._pdh.PdhGetFormattedCounterArrayW(
                self._counters[path],
                _PDH_FMT_LARGE,
                ctypes.byref(size),
                ctypes.byref(count),
                buffer,
            )
        if code != 0 or buffer is None:
            # No instance of this counter exists right now - no GPU process, for
            # one. **Not an error**: it is a real reading of nothing.
            return {}
        items = ctypes.cast(buffer, ctypes.POINTER(_CounterItem))
        return {items[i].szName: int(items[i].FmtValue.largeValue) for i in range(count.value)}

    def close(self) -> None:
        handle, self._handle = self._handle, ctypes.c_void_p()
        if handle:
            self._pdh.PdhCloseQuery(handle)


def _adapter_of(instance: str) -> str:
    """The luid part of a `\\GPU Process Memory` instance name."""
    _, _, rest = instance.partition("_")  # drop "pid"
    _, _, luid = rest.partition("_")  # drop the pid itself
    return luid


def _pid_of(instance: str) -> int:
    """The pid in a `\\GPU Process Memory` instance name, or 0."""
    parts = instance.split("_")
    if len(parts) < 2 or parts[0] != "pid":
        return 0
    try:
        return int(parts[1])
    except ValueError:
        return 0


#: How many readings are kept for the record of a death. At the default two
#: seconds apart this is a minute, which covers the ten seconds between a
#: decoder starting and the aborts measured on 2026-09-13 several times over.
HISTORY_SAMPLES = 30


class Sampler:
    """Samples the GPU counters on a timer, so that asking costs nothing.

    **One adapter is chosen, not all of them summed.** This machine reports
    three (measured 2026-09-06): the real GPU plus two software adapters, one of
    which held 2.4 GB of *shared* memory for an unrelated process. Adding them
    together would have read as a 2.4 GB spill at idle and aborted every
    generation. The real one is picked as **the adapter with dedicated memory in
    use**, which a software adapter never has - the desktop alone keeps that
    above zero. A machine with two real cards would need to be told which, and
    is not this machine.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._sample: Sample | None = None
        # **The last minute, not only the last reading.** A runner that dies
        # frees its memory as it goes, so the reading taken after a death says
        # nothing about the moment before it; the record of a death wants the
        # run-up (`Manager._keep_stderr`).
        self._history: list[Sample] = []
        self._watched: dict[int, str] = {}
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()

    # --- What is worth reporting ---------------------------------------------
    def watch(self, pid: int, what: str) -> None:
        """Report this process in `status.vram.by_pid` until it is forgotten."""
        if pid <= 0:
            return
        with self._lock:
            self._watched[int(pid)] = what

    def forget(self, pid: int) -> None:
        """Stop reporting a process (it has ended, or was somebody else's)."""
        with self._lock:
            self._watched.pop(int(pid), None)

    def watched(self) -> dict[int, str]:
        with self._lock:
            return dict(self._watched)

    # --- The thread -----------------------------------------------------------
    def start(self) -> None:
        """Begin sampling. **Does nothing where the counters do not exist.**"""
        if not available() or self._thread is not None:
            return
        self._thread = threading.Thread(target=self._run, name="hearth-vram", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()

    def _run(self) -> None:
        try:
            query = _Query()
        except OSError as exc:
            print(f"[hearth] VRAM counters are unavailable: {exc}", file=sys.stderr)
            return
        try:
            while not self._stop.is_set():
                try:
                    sample = self._take(query)
                except OSError as exc:
                    print(f"[hearth] a VRAM sample failed: {exc}", file=sys.stderr)
                else:
                    with self._lock:
                        self._sample = sample
                        self._history.append(sample)
                        del self._history[:-HISTORY_SAMPLES]
                self._stop.wait(config.VRAM_SAMPLE_SEC)
        finally:
            query.close()

    def _take(self, query: _Query) -> Sample:
        """One reading, resolved onto the adapter and the processes we care about."""
        raw = query.collect()
        adapter_dedicated = raw[_ADAPTER_DEDICATED]
        # The adapter with dedicated memory in use is the real card; see the
        # class docstring for why summing every adapter is wrong.
        luid = max(adapter_dedicated, key=lambda name: adapter_dedicated[name], default="")
        dedicated_used = adapter_dedicated.get(luid, 0) / _GB
        shared_used = raw[_ADAPTER_SHARED].get(luid, 0) / _GB

        with self._lock:
            watched = set(self._watched)
        by_pid: dict[int, dict[str, float]] = {}
        if watched:
            parents = _process_parents()
            per_pid_dedicated = self._per_pid(raw[_PROCESS_DEDICATED], luid)
            per_pid_shared = self._per_pid(raw[_PROCESS_SHARED], luid)
            for root in watched:
                # **The whole family, not just the pid.** See `_process_parents`.
                family = _family(root, parents)
                by_pid[root] = {
                    "dedicated_gb": sum(per_pid_dedicated.get(p, 0) for p in family) / _GB,
                    "shared_gb": sum(per_pid_shared.get(p, 0) for p in family) / _GB,
                }
        total = config.VRAM_DEDICATED_GB or dedicated_used
        return Sample(
            dedicated_used_gb=dedicated_used,
            dedicated_total_gb=total,
            shared_used_gb=shared_used,
            by_pid=by_pid,
            sampled_at=time.time(),
        )

    @staticmethod
    def _per_pid(instances: dict[str, int], luid: str) -> dict[int, int]:
        """Sum one counter per pid, **on one adapter only**."""
        out: dict[int, int] = {}
        for name, value in instances.items():
            if luid and _adapter_of(name) != luid:
                continue
            pid = _pid_of(name)
            if pid:
                out[pid] = out.get(pid, 0) + value
        return out

    # --- Asking ---------------------------------------------------------------
    def latest(self) -> Sample | None:
        """The last sample, or None if there has not been one yet."""
        with self._lock:
            return self._sample

    def history(self) -> list[Sample]:
        """The last `HISTORY_SAMPLES` readings, oldest first."""
        with self._lock:
            return list(self._history)

    def status(self) -> dict[str, Any] | None:
        """The `status.vram` value: the last sample, or None where there is none."""
        sample = self.latest()
        if sample is None:
            return None
        return sample.as_dict(shared_abort_gb=config.VRAM_SHARED_ABORT_GB, watched=self.watched())

    def spilled(self, pid: int) -> tuple[float, float] | None:
        """Whether one process is over the shared-memory threshold.

        Args:
            pid: The process to judge. Its children count as it
                (`_process_parents`).

        Returns:
            `(shared_gb, dedicated_gb)` when it is over, otherwise None.
            **None also when there is no sample**: an unknown reading is never
            reported as a spill.
        """
        limit = config.VRAM_SHARED_ABORT_GB
        if limit <= 0 or pid <= 0:
            return None
        sample = self.latest()
        if sample is None:
            return None
        entry = sample.by_pid.get(int(pid))
        if entry is None or entry["shared_gb"] < limit:
            return None
        return entry["shared_gb"], entry["dedicated_gb"]


# **One sampler for the process.** The counters are global, and a second open
# query would cost a second thread for the same numbers.
SAMPLER = Sampler()


@dataclass
class Holding:
    """Who holds the card right now: the adapter's total, and every process on it."""

    used_gb: float
    by_pid: dict[int, float]
    names: dict[int, str]
    parents: dict[int, int]


def read_now() -> Holding | None:
    """Read the card once, every process included. **None where it cannot be read.**

    Not the sampler's reading: the sampler only resolves the processes it was
    asked to watch, and the question here is about everyone else. A query opened
    and closed on the calling thread (PDH queries belong to one thread) costs a
    few milliseconds plus the process snapshot's 7-10 ms, which is nothing
    against the minute a load takes.
    """
    if not available():
        return None
    try:
        query = _Query()
    except OSError:
        return None
    try:
        raw = query.collect()
    except OSError:
        return None
    finally:
        query.close()
    adapter = raw[_ADAPTER_DEDICATED]
    luid = max(adapter, key=lambda name: adapter[name], default="")
    per_pid = Sampler._per_pid(raw[_PROCESS_DEDICATED], luid)
    table = _process_table()
    return Holding(
        used_gb=adapter.get(luid, 0) / _GB,
        by_pid={pid: value / _GB for pid, value in per_pid.items()},
        names={pid: name for pid, (_parent, name) in table.items()},
        parents={pid: parent for pid, (parent, _name) in table.items()},
    )


def shortfall(
    holding: Holding, *, need_gb: float, usable_gb: float, own_root: int
) -> tuple[float, float, list[dict[str, Any]]]:
    """How far a generation is from fitting, and who holds the difference.

    Args:
        holding: A reading (`read_now`).
        need_gb: What the runner declared it needs, weights included.
        usable_gb: What every process together can hold.
        own_root: The runner's process, when it is already running. **It and
            its children are not "others"**: its weights are part of `need_gb`.

    Returns:
        `(short_gb, others_gb, holders)`. **`short_gb <= 0` means it fits.**
        `holders` is the largest other processes, at most five, largest first.
    """
    own = _family(own_root, holding.parents) if own_root > 0 else set()
    own_gb = sum(holding.by_pid.get(pid, 0.0) for pid in own)
    others_gb = max(holding.used_gb - own_gb, 0.0)
    holders = sorted(
        (
            {"pid": pid, "name": holding.names.get(pid, "?"), "dedicated_gb": round(gb, 2)}
            for pid, gb in holding.by_pid.items()
            if pid not in own and gb >= 0.1
        ),
        key=lambda entry: -entry["dedicated_gb"],
    )[:5]
    return others_gb + need_gb - usable_gb, others_gb, holders
