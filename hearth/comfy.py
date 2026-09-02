# SPDX-License-Identifier: MIT
"""A blocking HTTP client for ComfyUI, built on httpx.

**ComfyUI is a separate application and hearth never starts or stops it.**
Workflows are submitted to whatever is already listening at
`HEARTH_COMFY_BASE_URL` and the results are fetched back.
**Nothing is ever installed into ComfyUI's virtual environment**: the only
contact is over HTTP.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any

import httpx

from . import config


class ComfyUINotReadyError(RuntimeError):
    """ComfyUI cannot be reached, or is still starting."""


class ComfyUIExecutionError(RuntimeError):
    """The workflow ended with an error."""


@dataclass
class OutputFile:
    """A reference to one file ComfyUI produced, fetchable through /view."""

    filename: str
    subfolder: str
    type: str  # "output" | "temp" | "input"
    node_id: str
    kind: str  # A key of the outputs dict: "images", "gifs", "3d", "meshes", ...


@dataclass
class ComfyUIClient:
    base_url: str = config.COMFY_BASE_URL
    client_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    timeout_sec: float = 30.0

    def _url(self, path: str) -> str:
        return f"{self.base_url}/{path.lstrip('/')}"

    # --- Reachability --------------------------------------------------------
    def is_alive(self) -> bool:
        try:
            r = httpx.get(self._url("system_stats"), timeout=3.0)
            return r.status_code == 200
        except httpx.HTTPError:
            return False

    def system_stats(self) -> dict[str, Any]:
        r = httpx.get(self._url("system_stats"), timeout=self.timeout_sec)
        r.raise_for_status()
        return r.json()

    def free_models(self, *, unload_models: bool = True, free_memory: bool = True) -> bool:
        """Ask ComfyUI to release its models and give the VRAM back.

        A run generates an image in ComfyUI and then loads a 3D model into the
        same GPU, in another process. Holding both at once does not fit, so this
        is called between the two stages.

        **Best effort: it never raises.** An older ComfyUI has no `/free` route,
        and failing to free memory is not a reason to abandon a run that might
        still succeed.

        Args:
            unload_models: Whether to unload the loaded models.
            free_memory: Whether to drop the caches.

        Returns:
            True if the request was accepted.
        """
        payload = {"unload_models": unload_models, "free_memory": free_memory}
        try:
            response = httpx.post(self._url("free"), json=payload, timeout=30.0)
            return response.status_code < 400
        except httpx.HTTPError:
            return False

    # --- Uploading an input image (for img2img and image-to-3D) --------------
    def upload_image(self, image_bytes: bytes, filename: str, overwrite: bool = True) -> str:
        """Upload an image into ComfyUI's input/ and return the name to refer to it by."""
        files = {"image": (filename, image_bytes, "application/octet-stream")}
        data = {"overwrite": "true" if overwrite else "false"}
        r = httpx.post(self._url("upload/image"), files=files, data=data, timeout=self.timeout_sec)
        r.raise_for_status()
        body = r.json()
        name = body.get("name", filename)
        sub = body.get("subfolder", "")
        return f"{sub}/{name}" if sub else name

    # --- Submitting a workflow and waiting for it ---------------------------
    def queue_prompt(self, workflow: dict[str, Any]) -> str:
        """Submit an API-format workflow and return its prompt_id."""
        payload = {"prompt": workflow, "client_id": self.client_id}
        try:
            r = httpx.post(self._url("prompt"), json=payload, timeout=self.timeout_sec)
        except httpx.HTTPError as exc:  # noqa: BLE001 - turn unreachable into our own type
            raise ComfyUINotReadyError(f"cannot reach ComfyUI: {exc}") from exc
        if r.status_code == 400:
            raise ComfyUIExecutionError(f"the workflow was rejected: {r.text}")
        r.raise_for_status()
        return r.json()["prompt_id"]

    def wait_for(self, prompt_id: str, timeout_sec: float, poll_sec: float = 1.5) -> dict[str, Any]:
        """Poll until the prompt_id appears in the history, and return that entry."""
        deadline = time.monotonic() + timeout_sec
        while time.monotonic() < deadline:
            r = httpx.get(self._url(f"history/{prompt_id}"), timeout=self.timeout_sec)
            r.raise_for_status()
            entry = r.json().get(prompt_id)
            if entry is not None:
                status = entry.get("status", {})
                # An execution error stacked into `messages` means it failed.
                for msg in status.get("messages", []):
                    if (
                        isinstance(msg, (list, tuple))
                        and len(msg) == 2
                        and msg[0] == "execution_error"
                    ):
                        raise ComfyUIExecutionError(str(msg[1]))
                if status.get("status_str") == "error":
                    raise ComfyUIExecutionError(f"prompt {prompt_id} ended with an error")
                # Present and completed means success; a missing flag counts as success.
                if status.get("completed", True):
                    return entry
            time.sleep(poll_sec)
        raise TimeoutError(f"prompt {prompt_id} did not finish within {timeout_sec} seconds")

    # --- The resulting files -------------------------------------------------
    @staticmethod
    def collect_outputs(history_entry: dict[str, Any]) -> list[OutputFile]:
        """Flatten a history entry's outputs into a list of file references."""
        result: list[OutputFile] = []
        for node_id, node_out in history_entry.get("outputs", {}).items():
            for kind, items in node_out.items():
                if not isinstance(items, list):
                    continue
                for item in items:
                    if not isinstance(item, dict) or "filename" not in item:
                        continue
                    result.append(
                        OutputFile(
                            filename=item["filename"],
                            subfolder=item.get("subfolder", ""),
                            type=item.get("type", "output"),
                            node_id=node_id,
                            kind=kind,
                        )
                    )
        return result

    def download(self, out: OutputFile) -> bytes:
        params = {"filename": out.filename, "subfolder": out.subfolder, "type": out.type}
        r = httpx.get(self._url("view"), params=params, timeout=self.timeout_sec)
        r.raise_for_status()
        return r.content
