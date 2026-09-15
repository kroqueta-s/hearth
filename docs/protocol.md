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

A generating request whose runner died without answering is tried once more
(`HEARTH_GENERATE_RETRIES`). It looks like this, and the `attempts` in the
result is there so that a caller timing the call can explain the extra load:

```json
{"id": 1, "event": "progress", "stage": "retry", "message": "trellis2 died without answering; ..."}
{"id": 1, "event": "result", "result": {"mesh_path": "C:/out/raw.ply", "attempts": 2, "runner_logs": ["C:/out/runner_stderr_1.txt"]}}
```

**Every death leaves its stderr beside the output**, as
`runner_stderr_<attempt>.txt` in the request's `out_dir`: which runner and method,
the attempt, the exit code, and every line the process wrote (up to the last
400). A result that took more than one attempt names those files in
`runner_logs`; an error after the retries are spent names them in its message.
**Keep them when a run fails.** A driver fault that happens one run in several
can only be studied from the runs that recorded it, and a retry that succeeded
is one of those.

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
`comfy_start` is not an exception: it starts a process and returns, and the
weights are loaded by ComfyUI in its own process, watched through `status`.

## 3. Methods

### Control

| Method | Arguments | Returns |
|---|---|---|
| `ping` | — | `ok`, `pid`, `python`, `role`, `protocol` (§6). **Starts nothing**, so it is instant |
| `status` | — | §4. **Starts no runner**: inventory and state only |
| `capabilities` | `model` (optional) | One runner's capability table ([contract §3](runner_contract.md#3-what-capabilities-looks-like)), or every runner's when `model` is omitted. **Starting a runner to ask is cheap, but it is not free** — see §4 |
| `cancel` | — | `{"canceled": bool, "was": name}`, or `why` when there was nothing to cancel. §5 |
| `comfy_start` | — | `{"state": "starting"|"ready", "owned": bool, "pid": int|null}`. **Returns at once**: watch `status.comfy` for the rest. §4a |
| `comfy_stop` | — | `{"stopped": bool, "why": str|null}`. **Only the ComfyUI hearth started.** §4a |
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
| `segment_mesh` | `model`, `mesh_path` | Which part of a thing each face belongs to, for **every** number of parts at once ([contract §5a](runner_contract.md)). **It makes no mesh**: the answer is an `.npz` of labels plus the hash of the faces they are on, so a caller can refuse a labelling of some other mesh |
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
`.env`. **The image routes need ComfyUI running**; the mesh routes do not. A
route asked for while ComfyUI is `starting` waits for it (§4a) rather than
failing a race a person cannot win.

#### 3.1a What comes back, and how to read it

Every mesh route answers with the same shape ([contract
§5](runner_contract.md#5-image_to_mesh-results)). Four of its fields decide what
a caller can do, and three of them are easy to miss:

| Field | What to do with it |
|---|---|
| `mesh_path` | The mesh the method is named for. **Always there** |
| `extra` | Other files the same run produced, by name. **Absent, empty, or full** |
| `up_axis`, `forward_axis` | Which way the mesh is oriented, or `null` |
| `params_used` | What the runner actually ran with, for repeating a generation |

**`extra` is where a second file lands.** A runner that produces more than one
thing in a single call puts the rest here rather than inventing a method: a
background-removed image under `foreground`, a textured copy under
`textured_glb`. The keys belong to the runner, not to hearth, and hearth passes
the dictionary through untouched.

So a caller **looks for a key and uses it if it is there**. It does not require
one, and it does not decide which keys to expect from the model's name -
the same rule as `capabilities` (§3). A runner that gains a second output
becomes a caller that can offer it, with no change here.

**A file in `extra` may be a format the caller does not import, and may not be
the same kind of thing as `mesh_path`.** A runner may answer with a printable mesh
(`.ply`, perhaps carrying vertex colours) and a textured copy (`.glb`, carrying
UVs with colour, metalness and roughness). Those are two representations of one
generation, not a mesh and an improvement on it: **which one a caller wants
depends on what the caller is for**, a printable solid or something to look at.
A caller that imports only `mesh_path` is not broken; it simply has not been
given the choice.

**`textured_glb` is the one key with a fixed meaning** ([contract
§5](runner_contract.md#5-image_to_mesh-results), decided 2026-09-15): a binary
glTF of the same geometry, in the same frame and scale as `mesh_path`. A caller
may look for it by name - still only when it is there, and **never because of
which model answered**. Every other key belongs to the runner: read the
dictionary that came back, take what you can use, and leave the rest.

**Which format for what**, so that a caller knows what each file is for:
geometry that anything is measured or cut from is **PLY** (`mesh_path`);
appearance is **GLB** (`extra.textured_glb`). A glTF importer is free to convert
axes and units and to drop duplicate faces, which is harmless for a copy to look
at and wrong for a mesh whose faces are counted.

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
  "comfy": {"state": "ready", "owned": true, "pid": 27652, "url": "http://127.0.0.1:8200"},
  "vram": {"dedicated_used_gb": 29.1, "dedicated_total_gb": 32.0, "shared_used_gb": 0.1,
           "shared_abort_gb": 1.0,
           "by_pid": {"27652": {"dedicated_gb": 28.4, "shared_gb": 0.1, "what": "comfyui"}},
           "sampled_at": 1757170000.0},
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

**Each image model also says how it wants its prompt written**, as
`prompt_format`, for a caller that writes one for a person:

```json
"prompt_format": {"negative": false, "negative_why": "cfg 1", "style": "natural", "max_words": 80}
```

| Field | Means | Where it comes from |
|---|---|---|
| `negative` | **Whether the sampler reads the negative prompt at all.** Always present | **Read out of the model's workflow**, not declared: false when the sampler's `cfg` is 1.0 or less, or when the negative input passes through `ConditioningZeroOut`. A workflow hearth cannot follow counts as true, which is what it did before anything looked |
| `negative_why` | Why not, when `negative` is false | The same reading |
| `style` | `tags` (comma-separated short phrases) or `natural` (plain sentences). **Absent when nothing declared it**: say that a style is being assumed rather than guessing one | `HEARTH_IMAGE_MODEL_<KEY>_PROMPT_STYLE` |
| `max_words` | A word limit to write within. Absent when not declared | `HEARTH_IMAGE_MODEL_<KEY>_PROMPT_MAX_WORDS` |

**A negative prompt the model does not read is still accepted**, so a caller that
sends one is not refused; it simply changes nothing, and a form should say so
rather than offer the field as if it did. **The default of `negative` in
`params` is the text the workflow already carries**, so a default written into a
workflow is the one that runs when a caller sends none. `prompt_format` sits
beside `params` rather than inside it, so adding it does not change the table a
caller's form was built from.

### 4a. ComfyUI, and how full the card is

**hearth starts ComfyUI and stops the one it started.** ComfyUI is still a
separate application - nothing is installed into its virtual environment and its
code is never touched - but the launch arguments decide how much of the card it
takes, and sharing one GPU is already hearth's job. `comfy.state` is one of:

| `state` | Means |
|---|---|
| `absent` | Nothing is listening. The image routes fail with a reason |
| `starting` | hearth launched it and it has not answered yet. **An image route waits**, reporting `comfy` progress |
| `ready` | `/system_stats` answers |
| `failed` | It exited, or never answered within `HEARTH_COMFY_START_TIMEOUT_SEC`. `why` says which |

**`owned` is the difference between "hearth started it" and "hearth found it".**
There is one port, so a ComfyUI already running is adopted rather than
duplicated: `owned: false`, and `comfy_stop` leaves it alone and says so in
`why`. Show that in an interface — a Stop button that silently does nothing is
worse than one that explains itself. `pid` is the process actually listening,
which on Windows is a child of the one hearth launched.

**`vram` is read from Windows' own performance counters, not from a GPU
library.** This matters more than it sounds: on the machine hearth was written
for, `torch.cuda.mem_get_info` reports 43.87 GB of VRAM for a 32 GB card,
because it counts the shared pool — system RAM the driver spills into. An
application that believes it keeps loading, the driver pages, and **nothing
fails**: it just becomes several times slower. `dedicated_total_gb` comes from
`HEARTH_VRAM_DEDICATED_GB`, which is a measurement of the machine.

`shared_used_gb` rising **is** the spill. `by_pid` gives the same two numbers per
process for the ones hearth is watching — ComfyUI and any runner that is
generating — so the one that is paging is named rather than guessed at.
`sampled_at` is a unix time: the numbers come from a background sample, so
`status` stays instant however often it is asked. **`vram` is `null` where those
counters do not exist** (anywhere but Windows), and a caller must show that as
unknown rather than as zero.

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

Two things about that reply are worth reading precisely.

- **`canceled: true` means the request ends, not that ComfyUI was still holding
  the prompt.** The waiting side is asked to stop between polls, so it stops
  either way and the caller gets `CanceledError`. `dropped_from_queue` says
  whether the work itself was taken out; when it is false, `why` says why —
  usually that ComfyUI had already finished it, in which case an image was
  written and then abandoned.
- **A cancel can arrive before there is a prompt to cancel.** hearth marks
  itself busy before submitting the workflow — indeed before it unloads the 3D
  model to make room, which takes seconds — because a caller asking `status` in
  that time deserves an answer. A cancel in that window is remembered and the
  prompt is dropped the moment it exists; `why` says so.
- **A request still waiting in the GPU queue cannot be cancelled at all.**
  Nothing is busy until the queue reaches it, so `cancel` truthfully answers
  `nothing is generating`. Measured on 2026-09-03: the window is under 50 ms for
  an image, and a cancel at 250 ms or later always lands. A caller that sends
  cancel programmatically the instant it sends a request — `stop()` does — may
  fall inside it, and should treat `canceled: false` as "it may not have started
  yet" rather than "there was nothing to stop".
- **A cancel waits five seconds on ComfyUI, not thirty.** It is answered on the
  thread that reads stdin, so a wedged ComfyUI would otherwise hold up every
  control method queued behind it — including the `shutdown` a caller usually
  sends next.

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
| `VramOverError` | The work spilled out of the card into system memory | Say what to change. It carries `shared_gb`, `dedicated_gb` and `pid` |
| `VramShortError` | **The generation was not started**: other processes hold the room its runner declared it needs | Name the holders. It carries `need_gb`, `others_gb`, `usable_gb` and `holders` (`pid`, `name`, `dedicated_gb`, largest first) |
| `RuntimeError` | Generation failed, or ComfyUI is not running | Show the message |

**`VramOverError` is a failure that would otherwise not have been one.** Going
over the card does not raise anywhere: the driver falls back to shared memory -
system RAM - and the work carries on. Measured on 2026-09-06, FLUX at 2048x2048
put ComfyUI at 29.0 GB of dedicated VRAM and 1.1 GB of shared on a 32 GB card.
So hearth watches the shared usage of ComfyUI and of the running runner, and past
`HEARTH_VRAM_SHARED_ABORT_GB` it ends the work rather than letting it crawl —
taking the prompt out of ComfyUI's queue, or ending the runner's process. **The
numbers on the error are the argument for the advice**: a smaller image, fewer
steps, or a model that fits.

**`VramShortError` is the same limit met before the work instead of during it.**
When every process together reaches what the card can hold, the driver does not
only spill: measured on 2026-09-14, it also fails a runner's command submit with
PAL `ErrorOutOfGpuMemory`, which ends that process minutes into a generation. So
before each attempt of a generating method, hearth asks ComfyUI to free its
models, reads the card, and compares **what the other processes hold plus the
runner's declared `vram_peak_gb`** with `HEARTH_VRAM_USABLE_GB`. It waits up to
`HEARTH_VRAM_HEADROOM_WAIT_SEC` for memory to come back (reporting `vram_wait`
progress), and then refuses, naming the processes that hold the most. A runner
that declares nothing, or a card that cannot be read, is never refused.

**One hearth at a time, and it says so rather than fighting.** hearth holds a
local port while it owns the card - `HEARTH_LOCK_PORT`, **8011 by default** - and
a second hearth started against the same card answers `GpuBusyError` naming that
port instead of loading a model beside the first one. Two model loads on one card
do not fail; they fall back to shared memory and run several times slower, which
is the kind of failure nobody attributes to its cause. **Set it to 0 to disable
the lock**, which is what a test does so it can run while the operator's own
hearth is up.

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
