# The hearth protocol

**This document is the specification for talking to hearth.** It is what an
application implements: a Blender add-on, an editor plugin, a script.
[`docs/runner_contract.md`](runner_contract.md) is the other side of hearth and
is only of interest if you are **writing a model runner**.

There is a working client in [`client/hearth_client.py`](../client/hearth_client.py):
one file, no dependencies, MIT. **Copy it into your application** rather than
implementing the below by hand.

---

## 1. The wire

**hearth is a child process, not a server.** It has no port, no HTTP and no
authentication, because it never listens: you start it and talk over its stdin
and stdout.

```
your application  --stdin/stdout-->  hearth  --stdin/stdout-->  a model runner
```

```powershell
& C:\path\to\hearth\.venv\Scripts\python.exe -m hearth
```

| # | Rule | Why |
|---|---|---|
| 1 | **One message is one line of JSON**, UTF-8, separated by `\n` | Lines are the framing, so newlines inside a message stay JSON-escaped |
| 2 | **Read stderr and throw it away** | It is diagnostics, never protocol. **Left unread the pipe fills and hearth stops** |
| 3 | **Pass absolute paths, never bytes** | Everything is on one machine, so copying is waste |
| 4 | **A leading BOM is stripped** on the way in | Windows tools add one. Writing one is still a bug in the caller |
| 5 | **Replies are matched by `id`, not by order** | §3: replies interleave |

A request:

```json
{"id": 1, "method": "image_to_mesh", "params": {"model": "trellis", "image_path": "C:/in.png"}}
```

Its replies. **Zero or more `progress`, then exactly one `result` or `error`**:

```json
{"id": 1, "event": "progress", "stage": "shape", "message": "denoising", "step": 6, "total": 30}
{"id": 1, "event": "result", "result": {"mesh_path": "C:/out/raw.ply"}}
{"id": 1, "event": "error", "error": {"type": "RunnerError", "message": "..."}}
```

**A request that cannot be parsed is answered with nothing at all**, because
there is no `id` to answer to. It is written to stderr instead. Send valid JSON.

## 2. Two classes of method

**This is the part that surprises people.** hearth does not answer strictly in
order, and it must not.

| Class | Methods | Behaviour |
|---|---|---|
| **GPU** | `load`, `unload`, every generating method, `selftest_long_job` | **Strictly serial.** Queued and run one at a time, in the order they arrived. There is one GPU |
| **Control** | `ping`, `status`, `capabilities`, `cancel`, `shutdown` | **Answered immediately**, even while a generation is running |

So a `ping` sent during an eight-minute generation is answered in milliseconds,
**before** the generation it was sent after. Two consequences for the caller:

1. **Match replies by `id`.** A caller that assumes the next reply belongs to the
   last request will attribute a `ping` result to a generation and be wrong.
2. **The control methods are what make a responsive user interface possible.**
   Asking what is loaded and cancelling both work while the GPU is busy — which
   is precisely when a user wants them.

**Control methods never touch the GPU.** That is what makes answering them next
to a generation safe, and it is why nothing that loads weights is one of them.

## 3. Methods

### Control

| Method | Arguments | Returns |
|---|---|---|
| `ping` | — | `ok`, `pid`, `python`, `role`, `protocol` (§6). **Starts nothing**, so it is instant |
| `status` | — | §4. **Starts no runner**: inventory and state only |
| `capabilities` | `model` (optional) | One runner's capability table ([contract §3](runner_contract.md#3-what-capabilities-looks-like)), or every runner's when `model` is omitted. **Starting a runner to ask is cheap, but it is not free** — see §4 |
| `cancel` | — | `{"canceled": bool, "was": name}`, or `why` when there was nothing to cancel. §5 |
| `shutdown` | — | `{"bye": true}`, then hearth exits |

### GPU

| Method | Arguments | Gives |
|---|---|---|
| `load` | `model` | Switch to a model and load its weights. Reports `spawn_sec` apart from the load itself, and `already: true` when it was the one already there |
| `unload` | — | Free the GPU. Reports `vram_used_gb` as the runner measured it, `was` (the model) and `stop_sec` |
| `text_to_image` | `prompt`, §3.1 | An image |
| `image_to_image` | `image_path`, `prompt`, `denoise`, §3.1 | A reworked image. **The result's `image_path` is the new one**; the input comes back as `source_path` |
| `sketch_to_image` | `sketch_path`, `prompt`, `strength`, §3.1 | An image following a sketch. The sketch comes back as `source_path`, with `source_argument` naming which argument it was |
| `image_to_mesh` | `model`, `image_path` | A raw mesh |
| `multi_image_to_mesh` | `model`, `image_paths` | A raw mesh from several views |
| `texture_mesh` | `model`, `mesh_path`, `image_path` | A texture on a mesh you already have. **Its settings are the runner's `method_params.texture_mesh`**, not the ones `image_to_mesh` takes ([contract §3](runner_contract.md)) |
| `selftest_long_job` | `seconds`, `interval` | Nothing. **It occupies the GPU queue and reports progress**, which is how you test that your UI survives a long job without owning a GPU |

Every method takes an optional **`out_dir`**: an absolute path to write into.
Without one hearth makes a fresh directory per run under `HEARTH_OUTPUT_DIR`.
**Pass the same `out_dir` through a multi-step flow** to keep one piece of work
in one place instead of scattered across timestamps.

**hearth does not validate a model's own parameters.** Anything beyond the
arguments above is passed to the runner untouched, and **an argument the runner
never declared is rejected by the runner, with a reason**
([contract §3](runner_contract.md#3-what-capabilities-looks-like)).

#### 3.1 Image arguments

`prompt`, `negative`, `image_seed`, `image_steps`, `image_model`, and per route
`denoise` (`image_to_image`) or `strength` (`sketch_to_image`). `image_model` is
one of the names in `status.image_models`; omitting it uses the default from
`.env`. **The image routes need ComfyUI running**; the mesh routes do not.

#### 3.2 There is no "do the whole flow" method

Chaining belongs to the caller, and this is deliberate. A pipeline method inside
hearth would have to decide when to stop, how to preview an intermediate image,
and what to do when a user wants to redo one step of it — decisions that belong
to whatever is showing the result to a person. **Send one step, look at what came
back, send the next**; the client library has a helper for exactly this.

## 4. `status`

```json
{
  "loaded": "trellis",
  "busy": null,
  "available": ["hunyuan3d", "trellis", "hi3dgen"],
  "known": {"trellis": {"name": "trellis", "capabilities": {}, "params": {}}},
  "image_models": {"sdxl": {"capabilities": {}, "params": {}}},
  "default_image_model": "sdxl",
  "comfy_alive": true,
  "gpu_busy": false,
  "output_dir": "C:/.../output",
  "protocol": 1
}
```

**`status` starts nothing.** `known` holds the capability tables hearth has
already asked for; a runner that has not been asked yet is simply absent from
it. Ask for one with `capabilities` when the user selects that model, not for all
of them at startup: **each answer costs starting that runner's python**, and
doing three of those while a window is opening is felt.

`busy` names the model whose runner is generating right now, or `null`.

**`image_models` describes the image side in the same shape as the mesh side**,
so one piece of code can build a form for both. A route the model does not have
is `false` in its capability table (FLUX has no ControlNet here, for instance).

## 5. Long jobs: cancelling, and telling work from a hang

**Switching models costs a load** — tens of seconds, on the machine this was
written for, and that is a floor rather than something to design around. Loading
is dominated by reading the weights and putting them on the card, and neither
can happen while another model holds it. **So a flow that alternates between two
models pays for every switch**, and the way to spend less is to order the steps
so there are fewer of them, not to overlap them.

**`cancel`** ends the running generation by **ending the runner's process**. This
is the only thing that stops a `torch` loop reliably, so the price is fixed and
worth stating in your UI: the cancelled request fails with `CanceledError`, and
**the next generation with that model pays a full load again**. Between steps you
do not need it at all — just do not send the next one.

**Cancelling an image is a different thing, and costs less.** ComfyUI is another
application, so nothing of its is killed: hearth reads its queue and takes out
**its own prompt** — interrupting it if it is the one running, deleting it if it
is still waiting, and doing nothing at all if it has already finished. Somebody
else's job on that ComfyUI is never touched. The reply carries
`image_model_reload: false`, which means what it says and no more: the *image*
model is still loaded. **The 3D model is not** — it was unloaded to make room
before the image began — so the next mesh step still pays a load.

**A shutdown during a generation kills the runner first.** Asking it to unload
would wait for the generation to finish, which looks like a hang; and the usual
answer to a hang is to kill hearth, which on Windows leaves the runner alive with
the card. Requests still queued when a shutdown starts are answered with an
error rather than run.

**Progress is counted, never estimated.** A `progress` carries `stage` and
`message` always, `step` when it is a counted step, and `total` only when the
length is known. **Never show a percentage without a `total`**, and never derive
an ETA: on this class of hardware the first run of a loop can be an order of
magnitude slower than the ones after it, so a prediction is worst exactly when it
is most wanted. A `progress` with no `step` is a heartbeat: it proves the work is
alive and says nothing about how far along it is.
[Contract §8](runner_contract.md#8-progress) is the full rule.

## 6. Errors, death, and versions

An `error` carries a `type` and a `message`. The type is the exception's name and
is worth branching on; the message is for a person.

| `type` | Means | What a caller should do |
|---|---|---|
| `CanceledError` | You cancelled it | Say so. Not a failure |
| `GpuBusyError` | Another process holds the GPU | Say which port said so. **Do not kill anything on the user's behalf** |
| `RunnerError` | A runner would not start, died, or refused | Show the message: it carries the tail of that runner's stderr |
| `FileNotFoundError` | An input, a weight, or a repository is missing | Show the path |
| `ValueError` | An unknown argument or a value out of range | Show it verbatim; it names the argument |
| `RuntimeError` | Generation failed, or ComfyUI is not running | Show the message |

**If hearth dies, every outstanding request dies with it.** Nothing will answer
them, so a caller must notice the process is gone and fail them itself, with the
tail of stderr attached. **A user interface that waits forever is the failure
this rule exists to prevent.**

`ping` and `status` report `protocol`, an integer that changes when this document
changes in a way a caller can notice. **A caller should check it once at startup
and say something useful when it is newer than it understands** rather than
failing in the middle of a generation later. Runners carry their own version of
[the runner contract](runner_contract.md), reported as `contract` inside a
capability table.
