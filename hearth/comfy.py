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


class Interrupted(RuntimeError):
    """The caller asked for this prompt to stop.

    A type of its own so that `worker` can tell it from a real failure and
    report it as `CanceledError` (`docs/protocol.md` §6). It lives here rather
    than in `manager` because importing that from here would be a cycle.
    """


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

    def cancel_prompt(self, prompt_id: str) -> bool:
        """Take one prompt out of ComfyUI's queue, and **only that one**.

        ComfyUI is another application with other clients. `POST /interrupt`
        stops **whatever is running now**, which is not necessarily ours: if our
        prompt is still queued behind somebody else's, interrupting would kill
        their work and leave ours to run. hearth does not kill anything on a
        user's behalf (`docs/protocol.md` §6), and that holds for other people's
        jobs as much as for other processes.

        So the queue is read first:

        - running, and it is ours: `interrupt`;
        - still pending: delete it from the queue by id;
        - neither: it has already finished, and there is nothing to cancel.

        Args:
            prompt_id: What `queue_prompt` returned.

        Returns:
            Whether anything was actually taken out.
        """
        try:
            queue = httpx.get(self._url("queue"), timeout=self.timeout_sec).json()
        except (httpx.HTTPError, ValueError):
            return False
        if any(str(prompt_id) in str(entry) for entry in queue.get("queue_running", [])):
            try:
                httpx.post(self._url("interrupt"), timeout=self.timeout_sec)
            except httpx.HTTPError:
                return False
            return True
        if any(str(prompt_id) in str(entry) for entry in queue.get("queue_pending", [])):
            try:
                httpx.post(
                    self._url("queue"), json={"delete": [prompt_id]}, timeout=self.timeout_sec
                )
            except httpx.HTTPError:
                return False
            return True
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

    def wait_for(
        self,
        prompt_id: str,
        timeout_sec: float,
        poll_sec: float = 1.5,
        relay: Any | None = None,
        heartbeat_sec: float = 5.0,
        should_stop: Any | None = None,
    ) -> dict[str, Any]:
        """Poll until the prompt_id appears in the history, and return that entry.

        **The progress relayed from here is a heartbeat and nothing more.** It
        proves the image is still being worked on; it says nothing about how far
        along it is, because this route has no count to report - ComfyUI's
        history says "finished" or "not yet", and **inventing a fraction from
        elapsed time is exactly the estimate the contract forbids**
        (`docs/runner_contract.md` §8). A per-step count needs ComfyUI's
        WebSocket, which is a dependency this has not taken.

        Args:
            prompt_id: What `queue_prompt` returned.
            timeout_sec: When to give up.
            poll_sec: How often to ask.
            relay: Where heartbeats go. `(stage, message)`.
            heartbeat_sec: How often to send one.
            should_stop: Asked between polls. **This is what makes an image
                cancellable**: the prompt has already been taken out of
                ComfyUI's queue, so continuing to wait for it would be waiting
                for something that will never arrive.

        Raises:
            Interrupted: If `should_stop` says so.
            ComfyUIExecutionError: If the workflow failed.
            TimeoutError: If it never finished.
        """
        deadline = time.monotonic() + timeout_sec
        started = time.monotonic()
        last_beat = started
        while time.monotonic() < deadline:
            if should_stop is not None and should_stop():
                raise Interrupted(f"prompt {prompt_id} was cancelled")
            if relay is not None and time.monotonic() - last_beat >= heartbeat_sec:
                last_beat = time.monotonic()
                relay("image", f"ComfyUI is working ({int(last_beat - started)}s elapsed)")
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
