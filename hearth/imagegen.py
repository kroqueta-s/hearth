# SPDX-License-Identifier: MIT
"""Making the input image, through ComfyUI. **Images only; nothing here touches 3D.**

This lives in hearth rather than in a runner because **image generation does not
depend on which 3D model will be used**.

**The workflow JSON lives in `hearth/workflows/` and is meant to be edited.**
Values are injected by node id (`apply_overrides`), so a replacement workflow
has to keep the same ids and input keys. A mismatch raises `KeyError` rather
than being quietly ignored.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

from PIL import Image

from . import config
from .comfy import ComfyUIClient


def apply_overrides(
    workflow: dict[str, Any], overrides: list[tuple[str, str, Any]]
) -> dict[str, Any]:
    """Return a copy of an API-format workflow with values injected into it.

    Args:
        workflow: Node id to node dict; every node dict has ``"inputs"``.
        overrides: A sequence of ``(node_id, input_key, value)``.

    Returns:
        A deep copy. The argument is not modified.

    Raises:
        KeyError: If the node is missing, or the input key is not already there.
            **No new input keys are created**: existing values are replaced.
    """
    result = copy.deepcopy(workflow)
    for node_id, input_key, value in overrides:
        if node_id not in result:
            raise KeyError(f"node {node_id!r} is not in the workflow")
        inputs = result[node_id].get("inputs")
        if not isinstance(inputs, dict) or input_key not in inputs:
            raise KeyError(f"node {node_id!r} has no input {input_key!r}")
        inputs[input_key] = value
    return result


def load_workflow(name: str) -> dict[str, Any]:
    """Read a workflow JSON out of `hearth/workflows/`.

    Args:
        name: The file name, such as ``sdxl_txt2img.json``.

    Returns:
        The workflow dict.

    Raises:
        FileNotFoundError: If it is not there.
    """
    path = config.WORKFLOW_DIR / name
    if not path.is_file():
        raise FileNotFoundError(f"workflow not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def snap_to_sdxl(width: int, height: int, *, target: int = 1024) -> tuple[int, int]:
    """Scale so the long side lands near `target`, and round both sides to a multiple of 8.

    The VAE cannot take a side that is not a multiple of 8.

    Args:
        width: The original width.
        height: The original height.
        target: What the long side should be near.

    Returns:
        (width, height), both multiples of 8 and at least 8.

    Raises:
        ValueError: If either input is zero or negative.
    """
    if width <= 0 or height <= 0:
        raise ValueError(f"invalid image size: {width}x{height}")
    scale = target / float(max(width, height))
    out = []
    for value in (width, height):
        snapped = int(round(value * scale / 8.0)) * 8
        out.append(max(8, snapped))
    return (out[0], out[1])


def require_alive(client: ComfyUIClient) -> None:
    """Check that ComfyUI answers.

    Args:
        client: The ComfyUI client.

    Raises:
        RuntimeError: If it does not. **Starting it is not hearth's business**,
            so this only says so.
    """
    if not client.is_alive():
        raise RuntimeError(
            f"ComfyUI ({config.COMFY_BASE_URL}) does not answer. Start it first."
        )


def _run(client: ComfyUIClient, prompt_id: str, out_dir: Path) -> list[Path]:
    """Wait for a submitted workflow and save the images it produced into out_dir.

    Args:
        client: The ComfyUI client.
        prompt_id: What `queue_prompt` returned.
        out_dir: Where to save.

    Returns:
        The saved paths, in the order they were produced.

    Raises:
        RuntimeError: If no image came out at all.
    """
    entry = client.wait_for(prompt_id, timeout_sec=float(config.COMFY_TIMEOUT_SEC))
    out_dir.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []
    for out in client.collect_outputs(entry):
        if Path(out.filename).suffix.lower() not in (".png", ".jpg", ".jpeg", ".webp"):
            continue
        dest = out_dir / out.filename
        dest.write_bytes(client.download(out))
        saved.append(dest)
    if not saved:
        raise RuntimeError("no image was produced (check the workflow SaveImage node)")
    return saved


def _spec(image_model: str | None, route: str) -> tuple[dict[str, str], str]:
    """Return an image model's settings and the workflow for one route.

    **Each model has its own workflow file, but the node ids are kept aligned**,
    so one list of `(node_id, key, value)` serves every model on a given route.

    Args:
        image_model: The model name, or None for the default in `.env`.
        route: One of `txt2img`, `img2img`, `controlnet`.

    Returns:
        (settings, workflow file name).

    Raises:
        RuntimeError: If the model does not support the route.
    """
    name = image_model or config.DEFAULT_IMAGE_MODEL
    spec = config.image_model_spec(name)
    workflow = spec[route]
    if not workflow:
        raise RuntimeError(
            f"image model {name} does not support {route} "
            f"(HEARTH_IMAGE_MODEL_{name.upper()}_{route.upper()} is empty in .env)"
        )
    return spec, workflow


def text_to_image(
    client: ComfyUIClient,
    out_dir: Path,
    prompt: str,
    *,
    negative: str = "",
    seed: int = 0,
    steps: int = 25,
    width: int = 1024,
    height: int = 1024,
    image_model: str | None = None,
) -> Path:
    """Generate one image from a text prompt, save it, and return its path."""
    spec, workflow = _spec(image_model, "txt2img")
    wf = apply_overrides(
        load_workflow(workflow),
        [
            ("4", "ckpt_name", spec["checkpoint"]),
            ("6", "text", prompt),
            ("7", "text", negative),
            ("3", "seed", seed),
            ("3", "steps", steps),
            ("5", "width", width),
            ("5", "height", height),
        ],
    )
    return _run(client, client.queue_prompt(wf), out_dir)[0]


def sketch_to_image(
    client: ComfyUIClient,
    out_dir: Path,
    sketch_path: Path,
    prompt: str,
    *,
    negative: str = "",
    seed: int = 0,
    steps: int = 25,
    strength: float = 0.8,
    max_dim: int = 1024,
    image_model: str | None = None,
) -> Path:
    """Generate one image from a sketch plus a prompt, save it, and return its path."""
    spec, workflow = _spec(image_model, "controlnet")
    width, height, staged = _stage_input(sketch_path, out_dir, "sketch_input.png", max_dim)
    ref = client.upload_image(staged.read_bytes(), "hearth_sketch.png")
    wf = apply_overrides(
        load_workflow(workflow),
        [
            ("4", "ckpt_name", spec["checkpoint"]),
            ("12", "control_net_name", config.CONTROLNET_MODEL),
            ("10", "image", ref),
            ("6", "text", prompt),
            ("7", "text", negative),
            ("5", "width", width),
            ("5", "height", height),
            ("3", "seed", seed),
            ("3", "steps", steps),
            ("13", "strength", strength),
        ],
    )
    return _run(client, client.queue_prompt(wf), out_dir)[0]


def image_to_image(
    client: ComfyUIClient,
    out_dir: Path,
    image_path: Path,
    prompt: str,
    *,
    negative: str = "",
    seed: int = 0,
    steps: int = 25,
    denoise: float = 0.6,
    max_dim: int = 1024,
    image_model: str | None = None,
) -> Path:
    """Generate one image from an input image plus a prompt, save it, and return its path."""
    spec, workflow = _spec(image_model, "img2img")
    _, _, staged = _stage_input(image_path, out_dir, "img2img_input.png", max_dim)
    ref = client.upload_image(staged.read_bytes(), "hearth_img2img.png")
    wf = apply_overrides(
        load_workflow(workflow),
        [
            ("4", "ckpt_name", spec["checkpoint"]),
            ("10", "image", ref),
            ("6", "text", prompt),
            ("7", "text", negative),
            ("3", "seed", seed),
            ("3", "steps", steps),
            ("3", "denoise", denoise),
        ],
    )
    return _run(client, client.queue_prompt(wf), out_dir)[0]


def _stage_input(src: Path, out_dir: Path, name: str, max_dim: int) -> tuple[int, int, Path]:
    """Resize an input image to something the model can take, and put it in out_dir.

    Args:
        src: The original image.
        out_dir: Where to put it.
        name: What to call it.
        max_dim: What the long side should be near.

    Returns:
        (width, height, the path it was written to).
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    with Image.open(src) as im:
        rgb = im.convert("RGB")
        width, height = snap_to_sdxl(rgb.width, rgb.height, target=max_dim)
        prepared = rgb.resize((width, height), Image.Resampling.LANCZOS)
    dest = out_dir / name
    prepared.save(dest)
    return (width, height, dest)
