# SPDX-License-Identifier: MIT
"""The methods hearth offers its caller.

**hearth's job ends at a raw mesh.** Scaling it to real-world size, repairing
it, and checking it for interference are somebody else's work and none of it
happens here.

The methods come in two classes (`docs/protocol.md` §2), and which one a method
is in is a property of the method, declared at the bottom of this file:

- **GPU** methods are queued and run one at a time. There is one GPU.
- **Control** methods are answered while a generation is running, which is what
  makes a user interface responsive at the only time it matters.

**There is no method that runs a whole flow.** Chaining `text_to_image` into
`image_to_mesh` is the caller's to do, because only the caller knows whether the
person is going to look at the image first (`docs/protocol.md` §3.2).
"""

from __future__ import annotations

import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from . import comfy, config, imagegen, manager, vram
from .comfy import ComfyUIClient
from .comfy_process import ABSENT, COMFY, FAILED, READY
from .manager import CanceledError, GpuBusyError, Manager, assert_gpu_free
from .rpc import Request, Responder

MANAGER = Manager()


def _gpu_busy() -> bool:
    """Whether another process holds the GPU. **This one does not raise.**"""
    try:
        assert_gpu_free()
    except GpuBusyError:
        return True
    return False


def _run_dir(params: dict[str, Any]) -> Path:
    """Where this request writes: what the caller asked for, or a fresh directory.

    **A caller that passes the same `out_dir` through several steps keeps one
    piece of work in one place.** Without that, a flow of four steps leaves four
    directories named by the second they started, and putting them back together
    afterwards is guesswork.
    """
    asked = str(params.get("out_dir") or "").strip()
    run_dir = Path(asked) if asked else config.OUTPUT_DIR / datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


# --- Control: answered while the GPU is busy ---------------------------------


def m_ping(params: dict[str, Any], responder: Responder) -> dict[str, Any]:
    """Answer that hearth is there. **No runner is started**, so this is instant."""
    return {
        "ok": True,
        "pid": os.getpid(),
        "python": sys.version.split()[0],
        "role": "hearth",
        "protocol": config.PROTOCOL_VERSION,
    }


def m_status(params: dict[str, Any], responder: Responder) -> dict[str, Any]:
    """Report what is installed and what state it is in. **This starts nothing.**

    `known` holds only the capability tables that have already been asked for.
    Asking every runner would mean starting every runner's python, and a caller
    that does that while a window opens is felt by the person opening it. Ask for
    one with `capabilities` when a model is chosen (`docs/protocol.md` §4).
    """
    comfy_state = COMFY.status()
    return {
        "loaded": MANAGER.loaded(),
        "busy": MANAGER.busy(),
        "available": MANAGER.available(),
        "known": MANAGER.known_capabilities(),
        "image_models": imagegen.all_capabilities(),
        "default_image_model": config.DEFAULT_IMAGE_MODEL,
        # **Kept, and still the answer to "can I ask for an image".** `comfy`
        # below says more, but a caller written against the older shape is not
        # broken by the new one.
        "comfy_alive": comfy_state["state"] == READY,
        "comfy": comfy_state,
        # **The card as Windows sees it, not as a GPU library reports it**
        # (`vram.py`). `null` where those counters do not exist.
        "vram": vram.SAMPLER.status(),
        "gpu_busy": _gpu_busy(),
        "output_dir": str(config.OUTPUT_DIR),
        "protocol": config.PROTOCOL_VERSION,
    }


def m_capabilities(params: dict[str, Any], responder: Responder) -> dict[str, Any]:
    """One runner's capability table, or every runner's when no model is named.

    **This starts the runner it asks**, which is cheap but not free: answering
    without loading any weights is part of the contract (§2), but a python still
    has to start.
    """
    model = str(params.get("model") or "").strip()
    if model:
        return MANAGER.capabilities(model)
    return MANAGER.all_capabilities()


def m_comfy_start(params: dict[str, Any], responder: Responder) -> dict[str, Any]:
    """Start ComfyUI, or adopt the one already listening. **Answers at once.**

    Loading FLUX takes about a minute, so this returns `state: "starting"` and
    the caller watches `status.comfy` - a control method that blocked for a
    minute would stop `cancel` being read for that minute
    (`docs/protocol.md` §2).

    **This does not touch the GPU**, which is what keeps it a control method:
    starting a child process is not loading weights, and the weights that follow
    are loaded by ComfyUI in its own process.
    """
    return COMFY.start()


def m_comfy_stop(params: dict[str, Any], responder: Responder) -> dict[str, Any]:
    """Stop ComfyUI, **but only the one hearth started**.

    A ComfyUI somebody else launched is left running and `why` says so. It is
    the same rule as `GpuBusyError`: hearth does not end other people's
    processes on a user's behalf (`docs/protocol.md` §6).
    """
    return COMFY.stop()


def m_cancel(params: dict[str, Any], responder: Responder) -> dict[str, Any]:
    """End the generation that is running, by ending its runner's process.

    **The weights go with it** (`docs/runner_contract.md` §9), so the next
    generation with that model pays a full load. The cancelled request answers
    with `CanceledError`.
    """
    return MANAGER.cancel()


# --- GPU: queued, one at a time ----------------------------------------------


def m_load(params: dict[str, Any], responder: Responder) -> dict[str, Any]:
    """Switch to a model and load its weights.

    **ComfyUI comes down first, exactly as it does before a generation.** A load
    asked for on its own puts the same weights on the same card, so leaving an
    image model resident here would spill into shared memory just as quietly.
    """
    _free_comfy(responder)
    return MANAGER.load(str(params["model"]), relay=responder.progress)


def m_unload(params: dict[str, Any], responder: Responder) -> dict[str, Any]:
    """Unload the current model and give the VRAM back."""
    return MANAGER.unload(relay=responder.progress)


def _split_params(params: dict[str, Any], *taken: str) -> tuple[str, dict[str, Any]]:
    """Separate `model` from everything that passes straight through to the runner.

    **hearth never validates a runner's own arguments**: only the runner knows
    what its values mean.

    Args:
        params: The RPC arguments.
        *taken: Keys hearth consumes itself, which the runner never sees.

    Returns:
        (runner name, the arguments to pass on).
    """
    model = str(params["model"])
    consumed = {"model", "out_dir", *taken}
    passthrough = {k: v for k, v in params.items() if k not in consumed}
    return model, passthrough


def m_image_to_mesh(params: dict[str, Any], responder: Responder) -> dict[str, Any]:
    """One image to a raw mesh. **Preprocessing is the runner's job.**

    With `texture: true`, a runner that supports it goes on to bake a texture.
    """
    model, passthrough = _split_params(params, "image_path")
    run_dir = _run_dir(params)
    _free_comfy(responder)
    result = MANAGER.generate(
        model,
        "image_to_mesh",
        {"image_path": str(params["image_path"]), "out_dir": str(run_dir), **passthrough},
        relay=responder.progress,
    )
    return {"run_dir": str(run_dir), "input_image": str(params["image_path"]), **result}


def m_multi_image_to_mesh(params: dict[str, Any], responder: Responder) -> dict[str, Any]:
    """Several views of one subject to a raw mesh.

    **Only runners that say so support this** (`capabilities.multi_image_to_mesh`).
    What the views have to be - how many, from where - is the runner's business
    and belongs in its `notes`.
    """
    model, passthrough = _split_params(params, "image_paths")
    paths = [str(p) for p in params["image_paths"]]
    run_dir = _run_dir(params)
    _free_comfy(responder)
    result = MANAGER.generate(
        model,
        "multi_image_to_mesh",
        {"image_paths": paths, "out_dir": str(run_dir), **passthrough},
        relay=responder.progress,
    )
    return {"run_dir": str(run_dir), "input_images": paths, **result}


def m_texture_mesh(params: dict[str, Any], responder: Responder) -> dict[str, Any]:
    """An existing mesh plus a reference image, to a textured mesh.

    **Only runners that say so support this** (`capabilities.texture_mesh` in
    `status`). The mesh can come from anywhere: another model's output, or one
    made by hand.
    """
    model, passthrough = _split_params(params, "mesh_path", "image_path")
    run_dir = _run_dir(params)
    _free_comfy(responder)
    result = MANAGER.generate(
        model,
        "texture_mesh",
        {
            "mesh_path": str(params["mesh_path"]),
            "image_path": str(params["image_path"]),
            "out_dir": str(run_dir),
            **passthrough,
        },
        relay=responder.progress,
    )
    return {"run_dir": str(run_dir), **result}


def m_segment_mesh(params: dict[str, Any], responder: Responder) -> dict[str, Any]:
    """An existing mesh to a part label for every face. **It makes no mesh.**

    **Only runners that say so support this** (`capabilities.segment_mesh`), and
    that is the whole of how it is offered: nothing here knows which runner can
    segment, or that segmenting is different from generating.

    It is the first method in this file that answers with something other than a
    mesh or an image, and hearth's part in it is exactly the same as in the
    others - name the runner, pass the rest through, relay the progress. **The
    labels are not interpreted here**: what a label means, and which face is
    which, is between the runner and the caller.
    """
    model, passthrough = _split_params(params, "mesh_path")
    run_dir = _run_dir(params)
    _free_comfy(responder)
    result = MANAGER.generate(
        model,
        "segment_mesh",
        {"mesh_path": str(params["mesh_path"]), "out_dir": str(run_dir), **passthrough},
        relay=responder.progress,
    )
    return {"run_dir": str(run_dir), "source_mesh": str(params["mesh_path"]), **result}


# --- Making an image ----------------------------------------------------------
#
# **The image is worth working on, not something to throw away.** Spending a
# minute turning an image you dislike into a mesh helps nobody, so these stop at
# the image and leave it in `run_dir`. Hand its path to `image_to_mesh` when it
# is the one you want - or back to `image_to_image` when it nearly is.

# The input file each route consumes itself, which is therefore not one of the
# model's own parameters.
_IMAGE_INPUT = {"image_to_image": "image_path", "sketch_to_image": "sketch_path"}


def _image(method: str, params: dict[str, Any], responder: Responder) -> dict[str, Any]:
    """Run one image route and report the settings it actually used.

    Args:
        method: One of `imagegen.ROUTES`.
        params: The RPC arguments.
        responder: Where progress goes.

    Returns:
        `run_dir`, `image_path`, `image_model` and `params_used`. A route that
        consumes an input file also reports `source_path` and
        `source_argument`. **The input is never reported as `image_path`**:
        that key names what came out.

    Raises:
        RuntimeError: If ComfyUI is not running.
        ValueError: If a parameter was never declared for this route.
    """
    consumed = {"out_dir", "model", "image_model", _IMAGE_INPUT.get(method, "")}
    used = imagegen.effective_params(
        method, {k: v for k, v in params.items() if k not in consumed}
    )
    model = str(params.get("image_model") or config.DEFAULT_IMAGE_MODEL)

    run_dir = _run_dir(params)
    client = ComfyUIClient()
    _await_comfy(responder)
    # **An image is busy too.** It is somebody else's process, so `cancel` takes
    # the prompt out of ComfyUI's queue rather than killing anything (§5); but a
    # person waiting eight minutes for an image has the same right to stop it as
    # one waiting for a mesh, and `busy: null` during one is simply untrue.
    label = f"{manager.EXTERNAL_PREFIX}{model}"
    # **Marked busy before anything slow happens, not after.** Measured against a
    # live ComfyUI on 2026-09-03: a cancel sent a tenth of a second after the
    # request was answered `nothing is generating` and the image went on to
    # completion. Unloading the 3D model below takes seconds, and a person who
    # has just pressed a button is exactly who presses cancel next. From here
    # on, `cancel` has something to act on - and if one has already arrived,
    # `should_stop` ends the wait at its first look.
    MANAGER.begin_external(label)
    try:
        return _image_now(method, params, used, model, run_dir, client, responder)
    finally:
        MANAGER.end_external()


def _image_now(  # noqa: PLR0913 - one call, and every argument is already computed
    method: str,
    params: dict[str, Any],
    used: dict[str, Any],
    model: str,
    run_dir: Path,
    client: comfy.ComfyUIClient,
    responder: Responder,
) -> dict[str, Any]:
    """Generate one image, with the manager already told that work has begun."""
    # **The 3D model comes down before an image model goes up.** They share one
    # card; going over does not fail, it silently falls back to shared memory and
    # runs several times slower.
    _free_mesh(responder)

    shared: dict[str, Any] = {
        "negative": str(used["negative"]),
        "seed": int(used["image_seed"]),
        "steps": int(used["image_steps"]),
        "image_model": model,
        "relay": responder.progress,
        # **The two halves of cancelling an image.** `on_queued` tells the
        # manager which prompt to drop - it cannot be known before the workflow
        # is submitted - and `should_stop` is what turns a cancel into an
        # interrupted wait rather than a lie answered instantly.
        "on_queued": _queued(client),
        "should_stop": MANAGER.is_canceling,
        # **A spill ends the wait.** Going over the card does not fail, it gets
        # several times slower, so the wait is the only place it can be caught.
        "spill_check": COMFY.spilled,
    }
    prompt = str(used["prompt"])
    responder.progress("image", f"generating with {model}")
    try:
        image_path = _generate_image(method, client, run_dir, prompt, params, used, shared)
    except comfy.Interrupted as exc:
        raise manager.CanceledError(str(exc)) from None

    out: dict[str, Any] = {
        "run_dir": str(run_dir),
        "image_path": str(image_path),
        "image_model": model,
        # **What it actually ran with**, defaults filled in, so that "again with
        # one thing changed" is something a caller can offer.
        "params_used": used,
    }
    source = _IMAGE_INPUT.get(method)
    if source:
        # **Never under `image_path`.** That key names what came out, and for
        # `image_to_image` the input is an image too - echoing it there
        # overwrote the only record of where the new image was written. A caller
        # then had the input twice and the result not at all, and nothing
        # errored: the path it read was a real file, just the wrong one.
        out["source_path"] = str(params[source])
        out["source_argument"] = source
    return out



def _queued(client: comfy.ComfyUIClient) -> Any:
    """Tell the manager which prompt is running, and act on a cancel that raced.

    A cancel asked for while the workflow was still being submitted found no id
    to act on and could only set the flag. **Somebody has to take the prompt out
    of the queue**, and by the time it exists this is the only code that knows
    about it.
    """

    def queued(prompt_id: str) -> None:
        if MANAGER.note_external_prompt(prompt_id) and prompt_id:
            client.cancel_prompt(prompt_id)

    return queued


def _generate_image(  # noqa: PLR0913 - one call per route, and they differ
    method: str,
    client: ComfyUIClient,
    run_dir: Path,
    prompt: str,
    params: dict[str, Any],
    used: dict[str, Any],
    shared: dict[str, Any],
) -> Path:
    """Run the route ComfyUI was asked for, and hand back the image it wrote."""
    if method == "text_to_image":
        return imagegen.text_to_image(
            client,
            run_dir,
            prompt,
            width=int(used["width"]),
            height=int(used["height"]),
            **shared,
        )
    if method == "image_to_image":
        return imagegen.image_to_image(
            client,
            run_dir,
            Path(str(params["image_path"])),
            prompt,
            denoise=float(used["denoise"]),
            **shared,
        )
    return imagegen.sketch_to_image(
        client,
        run_dir,
        Path(str(params["sketch_path"])),
        prompt,
        strength=float(used["strength"]),
        **shared,
    )


def m_text_to_image(params: dict[str, Any], responder: Responder) -> dict[str, Any]:
    """A text prompt to one image. **It stops there.**"""
    return _image("text_to_image", params, responder)


def m_image_to_image(params: dict[str, Any], responder: Responder) -> dict[str, Any]:
    """An image plus a prompt, to one image. **For reworking an image you already have.**

    A lower `denoise` stays closer to the original.
    """
    return _image("image_to_image", params, responder)


def m_sketch_to_image(params: dict[str, Any], responder: Responder) -> dict[str, Any]:
    """A sketch plus a prompt, to one image, through ControlNet.

    A higher `strength` follows the sketch more closely.
    """
    return _image("sketch_to_image", params, responder)


def m_selftest_long_job(params: dict[str, Any], responder: Responder) -> dict[str, Any]:
    """Occupy the GPU queue for a while, reporting progress. **It uses no GPU.**

    **This is not a decoration.** It is the only way to test the two things a
    caller most needs to be true, without owning a GPU or waiting on a real
    generation: that a long job does not freeze the caller's interface, and that
    control methods are still answered while one is running
    (`docs/protocol.md` §2).

    Args:
        params: `seconds` (default 10), `interval` between reports, and
            `poison_fd1` - write one line of rubbish straight to file
            descriptor 1, the way a C extension does. **The reply has to survive
            it**, which is the whole point of the stdout guard.
        responder: Where progress goes.

    Returns:
        The seconds it took and the number of steps reported.
    """
    seconds = float(params.get("seconds", 10))
    interval = max(0.05, float(params.get("interval", 1.0)))
    if params.get("poison_fd1"):
        # Bypasses `sys.stdout` entirely, which is exactly what makes this worth
        # testing: no Python-level replacement can catch it.
        os.write(1, b"not json, straight to fd 1\n")
    # The count is known here, so it is reported: a total that is real is the
    # only kind allowed (`docs/runner_contract.md` §8).
    total = max(1, int(seconds / interval))
    started = time.perf_counter()
    for step in range(1, total + 1):
        time.sleep(interval)
        responder.progress("tick", "waiting on purpose", step=step, total=total)
    return {"elapsed_sec": round(time.perf_counter() - started, 2), "steps": total}


# --- Sharing one card ---------------------------------------------------------


def _await_comfy(responder: Responder) -> None:
    """Block until ComfyUI is ready, or say why it never will be.

    **Only a `starting` ComfyUI is waited for.** hearth may have been asked to
    start it a second ago - by `HEARTH_COMFY_AUTOSTART`, or by a caller that
    pressed Start and then Make image - and failing because the weights are
    still loading would be a race a person cannot win. An `absent` one is not
    waited for at all: nothing is coming.

    Raises:
        RuntimeError: If ComfyUI is not running and nothing is starting it.
    """
    state = COMFY.state()
    if state == READY:
        return
    if state in (ABSENT, FAILED):
        detail = COMFY.status().get("why") or "Start it first, or set HEARTH_COMFY_AUTOSTART=1."
        raise RuntimeError(f"ComfyUI ({config.COMFY_BASE_URL}) is not running. {detail}")
    started = time.monotonic()
    deadline = started + config.COMFY_START_TIMEOUT_SEC
    while time.monotonic() < deadline:
        state = COMFY.state()
        if state == READY:
            return
        if state in (ABSENT, FAILED):
            break
        responder.progress(
            "comfy", f"waiting for ComfyUI to start ({int(time.monotonic() - started)}s)"
        )
        time.sleep(2.0)
    why = COMFY.status().get("why") or "it never answered"
    raise RuntimeError(f"ComfyUI ({config.COMFY_BASE_URL}) did not start: {why}")


def _free_comfy(responder: Responder) -> None:
    """If ComfyUI is up, ask it to free its VRAM. Best effort.

    **Always call this before loading a 3D model.** They share one GPU, and an
    image model left resident does not leave room for a 3D one.
    """
    # **Asked of the state, not of ComfyUI.** A watcher keeps that state true
    # (`comfy_process.watch`), and probing here cost the whole connect timeout
    # every time ComfyUI was not running - on the machine this was written for a
    # closed port is dropped rather than refused, so that was seconds added to
    # every mesh generation to learn something already known.
    if COMFY.state() != READY:
        return
    responder.progress("free_vram", "asking ComfyUI to release its models")
    ComfyUIClient().free_models()


def _free_mesh(responder: Responder) -> None:
    """Unload the 3D model before an image is generated.

    The mirror image of `_free_comfy`, and it exists for the same reason. It can
    be turned off with `HEARTH_FREE_MESH_BEFORE_IMAGE=0` **when the two are known
    to fit together**, which is a measurement, not a hope: going over the card
    does not raise, it quietly falls back to shared memory.
    """
    if not config.FREE_MESH_BEFORE_IMAGE:
        return
    if MANAGER.loaded() is None:
        return
    responder.progress("free_vram", "unloading the 3D model to make room for the image model")
    MANAGER.unload(relay=responder.progress)


# **Which class a method is in** (`docs/protocol.md` §2). Control methods are
# answered while the GPU is busy; everything else is queued and run in order.
CONTROL_METHODS = {
    "ping": m_ping,
    "status": m_status,
    "capabilities": m_capabilities,
    "cancel": m_cancel,
    # **Starting a process is not using the GPU.** Both answer immediately and
    # the loading happens in ComfyUI's own process, watched through `status`.
    "comfy_start": m_comfy_start,
    "comfy_stop": m_comfy_stop,
}

GPU_METHODS = {
    "load": m_load,
    "unload": m_unload,
    # Making an image, and stopping there
    "text_to_image": m_text_to_image,
    "image_to_image": m_image_to_image,
    "sketch_to_image": m_sketch_to_image,
    # An image to a mesh
    "image_to_mesh": m_image_to_mesh,
    "multi_image_to_mesh": m_multi_image_to_mesh,
    # An existing mesh to a textured one
    "texture_mesh": m_texture_mesh,
    # An existing mesh to a part label per face. **Not every runner generates**
    "segment_mesh": m_segment_mesh,
    # Proving a long job does not freeze anything
    "selftest_long_job": m_selftest_long_job,
}

METHODS = {**CONTROL_METHODS, **GPU_METHODS}


def handle(request: Request, responder: Responder) -> None:
    """Handle one request and answer exactly once, with a result or an error.

    **No exception escapes.** A hearth that dies leaves its caller waiting
    forever.
    """
    method = METHODS.get(request.method)
    if method is None:
        responder.error(ValueError(f"unknown method: {request.method}"))
        return
    try:
        responder.result(method(request.params, responder))
    except CanceledError as exc:
        # Asked for, not a failure - but still the end of this request.
        responder.error(exc)
    except Exception as exc:  # noqa: BLE001 - whatever happens, still answer
        import traceback

        traceback.print_exc()
        responder.error(exc)
