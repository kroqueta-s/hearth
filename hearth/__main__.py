# SPDX-License-Identifier: MIT
"""hearth's entry point. **It talks over stdin and stdout, and serves no port.**

Start it as a child process::

    & $env:HEARTH_PYTHON -m hearth

**Each model's virtual environment belongs to its runner.** hearth itself never
imports torch: images come from ComfyUI over HTTP, meshes from a runner over
stdio.

## Why there are threads here

Requests arrive on one stream but do not all want the same thing
(`docs/protocol.md` §2). **Work that needs the GPU is queued and run one at a
time**, because there is one GPU. **Control methods are answered as they
arrive**, while that queue is busy - which is the whole point of them: asking
what is loaded, warming what comes next, and cancelling are all things a person
wants precisely while a generation is running.

So replies interleave, and **a caller matches them by `id` rather than by
order**. The wire says this plainly enough that a caller which ignores it will
attribute one request's answer to another.
"""

from __future__ import annotations

import queue
import sys
import threading

from .rpc import Channel, Request, install_stdout_guard, read_requests
from .worker import BACKGROUND_METHODS, CONTROL_METHODS, MANAGER, handle

# Requests waiting for the GPU. **Unbounded on purpose**: refusing to accept a
# request that is merely queued would make a caller invent its own queue.
_GPU_QUEUE: queue.Queue[tuple[Request, Channel] | None] = queue.Queue()


def _serve_gpu() -> None:
    """Run queued requests one at a time, in the order they arrived.

    **This thread is the GPU.** Nothing else in hearth may run a generation, and
    that is what keeps "one model at a time" true without a lock around the
    world.
    """
    while True:
        item = _GPU_QUEUE.get()
        if item is None:  # Shutting down.
            return
        request, channel = item
        handle(request, channel.responder(request.id))


def _serve_background(request: Request, channel: Channel) -> None:
    """Answer one request on a thread of its own (`warm`).

    It neither needs the GPU nor should hold up the control path, since reading
    weights off disk takes seconds that a `status` behind it would wait for.
    """
    handle(request, channel.responder(request.id))


def main() -> int:
    """Read requests until stdin closes, dispatching each to where it belongs.

    Returns:
        The exit code. 0 on a clean exit.
    """
    channel = Channel(install_stdout_guard())
    print("[hearth] started, waiting for requests on stdin.", file=sys.stderr)

    gpu_thread = threading.Thread(target=_serve_gpu, name="hearth-gpu", daemon=True)
    gpu_thread.start()

    try:
        for request in read_requests(sys.stdin):
            if request.method == "shutdown":
                # **Answered here, not in the queue.** A shutdown that waited for
                # a running generation would look like a hang, and the runners
                # are about to be ended anyway.
                channel.responder(request.id).result({"bye": True})
                break
            if request.method in BACKGROUND_METHODS:
                threading.Thread(
                    target=_serve_background,
                    args=(request, channel),
                    name=f"hearth-{request.method}",
                    daemon=True,
                ).start()
            elif request.method in CONTROL_METHODS:
                # Answered on this thread: these are fast and touch no GPU.
                handle(request, channel.responder(request.id))
            else:
                _GPU_QUEUE.put((request, channel))
    finally:
        _GPU_QUEUE.put(None)
        # **Leave no runner behind.** One that outlives hearth keeps the VRAM.
        MANAGER.shutdown()

    print("[hearth] exiting.", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
