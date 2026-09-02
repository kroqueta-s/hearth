# SPDX-License-Identifier: MIT
"""The methods hearth offers its caller.

**hearth's job ends at a raw mesh.** Scaling it to real-world size, repairing
it, and checking it for interference are somebody else's work and none of it
happens here.
"""

from __future__ import annotations

import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from . import config, imagegen
from .comfy import ComfyUIClient
from .manager import GpuBusyError, Manager, assert_gpu_free
from .rpc import Request, Responder

MANAGER = Manager()


def _gpu_busy() -> bool:
    """Whether another process holds the GPU. **This one does not raise.**"""
    try:
        assert_gpu_free()
    except GpuBusyError:
        return True
    return False


def _new_run_dir() -> Path:
    """Create and return `<output dir>/<date and time>` for one run."""
    run_dir = config.OUTPUT_DIR / datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def m_ping(params: dict[str, Any], responder: Responder) -> dict[str, Any]:
    """Answer that hearth is there. **No runner is started**, so this is instant."""
    return {"ok": True, "pid": os.getpid(), "python": sys.version.split()[0], "role": "hearth"}


def m_status(params: dict[str, Any], responder: Responder) -> dict[str, Any]:
    """Report what is installed and what state it is in.

    **A caller reads this to decide what to offer.** Asking a runner for its
    capabilities does start it, but answering without loading any weights is
    part of the contract, so it stays cheap.
    """
    return {
        "loaded": MANAGER.loaded(),
        "available": MANAGER.available(),
        "models": MANAGER.all_capabilities(),
        "comfy_alive": ComfyUIClient().is_alive(),
        "gpu_busy": _gpu_busy(),
        "output_dir": str(config.OUTPUT_DIR),
    }


def m_load(params: dict[str, Any], responder: Responder) -> dict[str, Any]:
    """Switch to a model and load its weights."""
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
    passthrough = {k: v for k, v in params.items() if k != "model" and k not in taken}
    return model, passthrough


def m_texture_mesh(params: dict[str, Any], responder: Responder) -> dict[str, Any]:
    """An existing mesh plus a reference image, to a textured mesh.

    **Only runners that say so support this** (`capabilities.texture_mesh` in
    `status`). The mesh can come from anywhere: another model's output, or one
    made by hand.
    """
    model, passthrough = _split_params(params, "mesh_path", "image_path")
    run_dir = _new_run_dir()
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


def m_image_to_mesh(params: dict[str, Any], responder: Responder) -> dict[str, Any]:
    """One image to a raw mesh. **Preprocessing is the runner's job.**

    With `texture: true`, a runner that supports it goes on to bake a texture.
    """
    model, passthrough = _split_params(params, "image_path")
    run_dir = _new_run_dir()
    _free_comfy(responder)
    result = MANAGER.generate(
        model,
        "image_to_mesh",
        {"image_path": str(params["image_path"]), "out_dir": str(run_dir), **passthrough},
        relay=responder.progress,
    )
    return {"run_dir": str(run_dir), "input_image": str(params["image_path"]), **result}


# --- Making an image, and stopping there -------------------------------------
#
# **The intermediate image is worth working on, not something to throw away.**
# Spending a minute turning an image you dislike into a mesh helps nobody, so
# making the image and making the mesh are separate methods: **look at the
# image, then go on to 3D.**
# All three leave the image in `run_dir` and return its path. Hand that path to
# `image_to_mesh` next.

_IMAGE_KEYS = ("prompt", "negative", "image_seed", "image_steps", "image_model")


def _image_args(params: dict[str, Any]) -> dict[str, Any]:
    """Pull out the arguments every image route shares.

    `image_model` is one of the names listed in `HEARTH_IMAGE_MODELS`, defaulting
    to `HEARTH_IMAGE_MODEL`. **No model is ever named in code.**
    """
    args: dict[str, Any] = {
        "negative": str(params.get("negative", "")),
        "seed": int(params.get("image_seed", 0)),
        "steps": int(params.get("image_steps", 25)),
    }
    if params.get("image_model"):
        args["image_model"] = str(params["image_model"])
    return args


def m_text_to_image(params: dict[str, Any], responder: Responder) -> dict[str, Any]:
    """A text prompt to one image. **It stops there.**"""
    run_dir = _new_run_dir()
    client = ComfyUIClient()
    imagegen.require_alive(client)
    responder.progress("image", "generating an image from the prompt")
    image_path = imagegen.text_to_image(
        client, run_dir, str(params["prompt"]), **_image_args(params)
    )
    return {"run_dir": str(run_dir), "image_path": str(image_path), "prompt": params["prompt"]}


def m_image_to_image(params: dict[str, Any], responder: Responder) -> dict[str, Any]:
    """An image plus a prompt, to one image. **For reworking an image you already have.**

    A lower `denoise` stays closer to the original (0.6 by default).
    """
    run_dir = _new_run_dir()
    client = ComfyUIClient()
    imagegen.require_alive(client)
    responder.progress("image", "reworking the image")
    image_path = imagegen.image_to_image(
        client,
        run_dir,
        Path(str(params["image_path"])),
        str(params["prompt"]),
        denoise=float(params.get("denoise", 0.6)),
        **_image_args(params),
    )
    return {
        "run_dir": str(run_dir),
        "image_path": str(image_path),
        "source_image": str(params["image_path"]),
        "prompt": params["prompt"],
    }


def m_sketch_to_image(params: dict[str, Any], responder: Responder) -> dict[str, Any]:
    """A sketch plus a prompt, to one image, through ControlNet.

    A higher `strength` follows the sketch more closely (0.8 by default).
    """
    run_dir = _new_run_dir()
    client = ComfyUIClient()
    imagegen.require_alive(client)
    responder.progress("image", "generating an image from the sketch")
    image_path = imagegen.sketch_to_image(
        client,
        run_dir,
        Path(str(params["sketch_path"])),
        str(params["prompt"]),
        strength=float(params.get("strength", 0.8)),
        **_image_args(params),
    )
    return {
        "run_dir": str(run_dir),
        "image_path": str(image_path),
        "sketch_path": str(params["sketch_path"]),
        "prompt": params["prompt"],
    }


# --- Image and mesh in one go: **a shortcut over the two stages above** -------
#
# These are exactly "make an image" followed by `image_to_mesh`, and **nothing
# new happens in them**. They exist for going straight through without stopping
# to look. **To work on the image, use the `*_to_image` methods above and hand
# the one you like to `image_to_mesh`.**


def _image_then_mesh(
    params: dict[str, Any],
    responder: Responder,
    taken: tuple[str, ...],
    make_image: Any,
) -> dict[str, Any]:
    """Make an image, then call `image_to_mesh` with it.

    Args:
        params: The RPC arguments.
        responder: Where progress goes.
        taken: Keys hearth consumes itself, which the runner never sees.
        make_image: A callable `(client, run_dir) -> path to the image`.

    Returns:
        `run_dir`, `input_image` and `prompt`, plus whatever the runner returned.
    """
    model, passthrough = _split_params(params, *taken)
    run_dir = _new_run_dir()
    client = ComfyUIClient()
    imagegen.require_alive(client)
    image_path = make_image(client, run_dir)
    _free_comfy(responder)
    result = MANAGER.generate(
        model,
        "image_to_mesh",
        {"image_path": str(image_path), "out_dir": str(run_dir), **passthrough},
        relay=responder.progress,
    )
    return {
        "run_dir": str(run_dir),
        "input_image": str(image_path),
        "prompt": params["prompt"],
        **result,
    }


def m_text_to_mesh(params: dict[str, Any], responder: Responder) -> dict[str, Any]:
    """A prompt to an image to a raw mesh. **ComfyUI has to be running.**"""

    def make_image(client: ComfyUIClient, run_dir: Path) -> Path:
        responder.progress("image", "generating an image from the prompt")
        return imagegen.text_to_image(client, run_dir, str(params["prompt"]), **_image_args(params))

    return _image_then_mesh(params, responder, _IMAGE_KEYS, make_image)


def m_sketch_to_mesh(params: dict[str, Any], responder: Responder) -> dict[str, Any]:
    """A sketch plus a prompt, to an image through ControlNet, to a raw mesh."""

    def make_image(client: ComfyUIClient, run_dir: Path) -> Path:
        responder.progress("image", "generating an image from the sketch")
        return imagegen.sketch_to_image(
            client,
            run_dir,
            Path(str(params["sketch_path"])),
            str(params["prompt"]),
            strength=float(params.get("strength", 0.8)),
            **_image_args(params),
        )

    return _image_then_mesh(
        params, responder, (*_IMAGE_KEYS, "sketch_path", "strength"), make_image
    )


def _free_comfy(responder: Responder) -> None:
    """If ComfyUI is up, ask it to free its VRAM. Best effort.

    **Always call this before loading a 3D model.** They share one GPU, and an
    image model left resident does not leave room for a 3D one.
    """
    client = ComfyUIClient()
    if client.is_alive():
        responder.progress("free_vram", "asking ComfyUI to release its models")
        client.free_models()


METHODS = {
    "ping": m_ping,
    "status": m_status,
    "load": m_load,
    "unload": m_unload,
    # Making an image, and stopping there
    "text_to_image": m_text_to_image,
    "image_to_image": m_image_to_image,
    "sketch_to_image": m_sketch_to_image,
    # An image to a mesh
    "image_to_mesh": m_image_to_mesh,
    # An existing mesh to a textured one
    "texture_mesh": m_texture_mesh,
    # The shortcut over the two stages
    "text_to_mesh": m_text_to_mesh,
    "sketch_to_mesh": m_sketch_to_mesh,
}


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
    except Exception as exc:  # noqa: BLE001 - whatever happens, still answer
        import traceback

        traceback.print_exc()
        responder.error(exc)
