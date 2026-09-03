# The runner contract

**This document is the specification.** A runner that follows it can be driven
by hearth without hearth knowing anything about the model inside it, and
**adding a fourth model becomes configuration rather than a patch**.

If you are writing a runner, start from [`templates/runner/`](../templates/runner)
— it implements everything below and passes `tests/test_template_runner.py`.

---

## 0. Why runners are separate processes

**Because the models' dependencies genuinely conflict.** This is a constraint,
not a preference:

- different builds and versions of torch,
- hand-built extensions that exist for one model and not another,
- libraries pinned to incompatible versions by different upstreams.

**One virtual environment per model.** hearth starts a runner as a child
process, and **only one model is loaded at a time**, because there is one GPU.

## 1. The wire

One JSON object per line, UTF-8, over stdin and stdout.

| # | Rule | Why |
|---|---|---|
| 1 | **One message is one line of JSON**, separated by `\n` | Lines are the framing, so newlines inside a message stay JSON-escaped |
| 2 | **Nothing but the protocol may write to stdout** | Model code prints to stdout as a matter of course. A runner duplicates the real stdout at startup, hides it, and points `sys.stdout` **and file descriptor 1 itself** at stderr, so a C extension cannot reach the protocol either |
| 3 | **stderr is never parsed, but must be drained** | Left unread, the pipe fills and the runner stops |
| 4 | **No bytes on the wire: pass absolute paths** | Both processes are on the same machine, so copying is waste |
| 5 | **Requests are serial** | There is one GPU |

## 2. Required methods

| Method | Arguments | Returns | Notes |
|---|---|---|---|
| `capabilities` | — | §3 | **Answer without loading the model.** hearth calls it when a caller chooses this model, not at startup ([protocol §4](protocol.md)) |
| `load` | — | `{"loaded": true, "elapsed_sec": float}` | Load the weights |
| `unload` | — | `{"unloaded": bool, "vram_used_gb": float}` | **Give the VRAM back.** hearth calls it before switching models |
| `image_to_mesh` | `image_path`, `out_dir`, plus §4 | §5 | **One image to a raw mesh.** Preprocessing is **the runner's job** |
| `shutdown` | — | `{"bye": true}` | Exit |

**Optional methods**, implemented only if `capabilities` declares them:
`text_to_mesh`, `multi_image_to_mesh`, `texture_mesh`.

**A runner is never asked to cancel anything** (§9).

## 3. What `capabilities` looks like

**Capabilities are data. Nothing branches on a model's name.**

```json
{
  "name": "hunyuan3d",
  "version": "2.1",
  "contract": 3,
  "capabilities": {
    "image_to_mesh": true,
    "text_to_mesh": false,
    "multi_image_to_mesh": false,
    "texture": true,
    "texture_mesh": true
  },
  "params": {
    "steps": {"type": "int", "default": 30, "min": 1, "max": 200},
    "octree_resolution": {"type": "int", "default": 384, "min": 64, "max": 768},
    "guidance_scale": {"type": "float", "default": 5.0, "min": 0.0, "max": 20.0},
    "seed": {"type": "int", "default": 0, "min": 0}
  },
  "method_params": {
    "texture_mesh": {"rembg": {"type": "bool", "default": true},
                     "save_glb": {"type": "bool", "default": false}}
  },
  "notes": "Free text. Anything a caller should know that the fields cannot say."
}
```

- **The moment a caller writes `if model == "..."`, this has failed.** Every new
  model would then mean touching the caller. **Branch on the capability table
  and the caller never changes.**
- **`contract` is the version of this document the runner was written against**,
  an integer. A runner that omits it is read as `1`. hearth passes it on and
  **never refuses a runner over it**: an old runner missing a new optional
  method is exactly the case capabilities already describes. It exists so a
  caller can say *why* something is unavailable rather than guessing.
- **A capability that is absent is false.** Adding a name to this table never
  breaks an older runner.
- `params` declares the model's own settings **for `image_to_mesh`**. hearth does
  not interpret them; it passes them through, and a user interface can build a
  form from this table.
- **`method_params` declares the settings of every *other* method**, by name.
  `texture_mesh` is not `image_to_mesh` with a flag: it takes a mesh from
  anywhere, it may be asked of a different model, and its settings are its own.
  Sending a mesh model's `steps` to it was accepted only because one runner
  happened to ignore what it did not recognise, and that is the kind of coupling
  a contract exists to prevent.
- **A method with no entry in `method_params` takes no settings**, only the
  arguments §4 names. At `contract` 1 or 2 there was no such table, so a caller
  may fall back to `params` for those runners - **and hearth says on stderr that
  it did**. The fallback is for old runners, not a default.
- **hearth does not validate either table.** Only the runner knows what its
  values mean, so checking them is the runner's job.

## 4. `image_to_mesh` arguments

| Argument | Required | Meaning |
|---|:--:|---|
| `image_path` | yes | **Absolute path** to the input image |
| `out_dir` | yes | **Absolute path** to write results into. hearth makes one per run |
| anything in `params` | no | The model's own settings. **An argument that was never declared is rejected with a reason** |

**Preprocessing belongs to the runner.** Whether background removal is needed,
and which method works, differs per model, and getting it wrong is a silent
quality loss rather than an error. **hearth does no preprocessing at all.**

## 5. `image_to_mesh` results

```json
{
  "mesh_path": "C:/.../out/raw.ply",
  "n_vertices": 637857,
  "n_faces": 1275718,
  "up_axis": "z",
  "forward_axis": "y",
  "params_used": {"steps": 30, "octree_resolution": 384, "guidance_scale": 5.0, "seed": 4711},
  "extra": {"foreground": "C:/.../out/foreground.png"},
  "metrics": {"load_sec": 40.2, "gen_sec": 87.9, "vram_peak_gb": 14.18}
}
```

**`up_axis` and `forward_axis` say which way the mesh is oriented**, as one of
`x`, `y`, `z`, `-x`, `-y`, `-z`. They are required at `contract` 3.

**Report `null` if it has not been measured.** That is not a formality: a mesh
imported on the wrong axis renders perfectly correctly, so nobody finds the
mistake by looking - a mirrored joint is the first sign, and by then it has been
printed. `null` travels downstream and the caller says "assumed, unverified"
where it would otherwise have said nothing at all. **hearth never fills these in**,
and neither should anything else: a guess here is indistinguishable from a
measurement, which is precisely what makes it expensive.

**`params_used` is every declared parameter with the value that was actually
used**, defaults filled in. It is what makes "run that again" and "same, but one
setting different" possible: a caller that only kept what it sent cannot
reproduce a result whose seed was drawn or whose default moved between versions.
**Report the value the model ran with**, not the one that arrived — if a value
was clamped, the clamped one is the true answer.

- **`mesh_path` is a PLY.** glTF splits and reorders vertices, which breaks any
  index the caller was given.
- **Normalized scale is fine.** Scaling to real-world units is downstream work.
- **Never use `metrics.gen_sec` as a pass/fail signal.** It varies by several
  times for identical settings, and the first run on a machine can be an order
  of magnitude slower while kernels are tuned.
- `extra` holds whatever intermediate files the model produced. hearth passes it
  through untouched.

## 6. Failure

| Type | Meaning |
|---|---|
| `FileNotFoundError` | An input, a weight file, or a repository is missing |
| `ValueError` | An unknown argument, or a value out of range |
| `RuntimeError` | Inference failed |
| `OSError` | A dependency would not load (a blocked DLL arrives here) |

**A runner never lets an exception escape.** One that dies leaves hearth waiting.
When hearth notices a runner has died, it fails the outstanding request and
attaches the tail of that runner's stderr.

## 7. Registering a runner

hearth learns about runners from `.env`. **No model is ever named in code.**

```
HEARTH_RUNNERS=hunyuan3d,trellis,hi3dgen
HEARTH_RUNNER_HUNYUAN3D_PYTHON=C:\path\to\its\.venv\Scripts\python.exe
HEARTH_RUNNER_HUNYUAN3D_MODULE=runners.hunyuan3d
HEARTH_RUNNER_HUNYUAN3D_CWD=C:\path\to\hunyuan3d-strix-halo
```

`install.ps1 -Runner <name>=<path>` and `tools/add_runner.ps1` write these
entries for you.

**`CWD` and `MODULE` are separate so that a runner can live in its own
repository.** Point `CWD` at the clone and nothing else changes; the runner
reads its own `.env` from there for the paths to its weights.

## 8. Progress

**Report what you counted. Never report an estimate.**

A runner may send `progress` for the request it is handling, as often as it
likes:

```json
{"id": 1, "event": "progress", "stage": "texture", "message": "multi-view denoising",
 "step": 7, "total": 15}
```

| Field | Required | Meaning |
|---|:--:|---|
| `stage` | yes | The stage's name. **An identifier for machines**, so keep it stable |
| `message` | yes | Free text for a person |
| `step` | no | The **counted** step, from 1. Only when it is counted |
| `total` | no | How many steps there are. **Only when the length is known** |

**Three rules.**

1. **No percentage without a `total`.** With only `step`, say "step 7". A
   receiver must not invent a denominator.
2. **Never send an ETA or an overall percentage.** The first run of a loop can
   be an order of magnitude slower than every run after it, so a prediction
   built from a stored constant is worst exactly when it is most wanted.
3. **Always send the first and last step.** Thin out the middle if the loop is
   fast, but never the two that say it started and finished.

`heartbeat` is a `progress` with no `step`. **It proves the runner is alive and
says nothing about how far along it is**, so it does not replace a count.

**Get the count from upstream rather than keeping your own.** For a diffusers
pipeline, the scheduler's `set_timesteps` fixes the total and its `step`
advances the count; for a hand-written loop, replace the `tqdm` in the module
that holds it. Both work without modifying the model's code, which matters
because that code gets replaced wholesale on the next update.

## 9. Cancelling

**A runner implements nothing for this.** There is no `cancel` method, and a
request that has started always runs to its end from the runner's point of view.

**hearth cancels by ending the process.** It is the only method that works
against a `torch` loop that does not check for anything, and it is the only one
that reliably gives the VRAM back. The consequences are the caller's to accept:

- the request being cancelled fails with `CanceledError`,
- **the weights are gone**, so the next generation pays a full load again.

A runner does not need to do anything to support this, but it must not make it
worse: **do not write a mesh file in place under its final name until it is
complete**, or a cancelled run leaves a truncated file that looks finished.
Write beside it and rename when it is whole - `os.replace` is atomic. A format
that writes several files at once (`.obj` with its `.mtl` and its textures) is
written into a directory of its own and **the directory** is renamed: renaming
only the mesh leaves the references inside it pointing at nothing.

## 10. Ending when hearth is gone

**A runner watches the process that started it and ends itself when it goes.**
hearth stops its runners when it shuts down, and a caller that kills hearth kills
the whole tree - but a hearth that *crashes* does neither. On Windows the child
carries on with the entire card, and **nothing anywhere errors**: everything
afterwards is several times slower for a reason nobody can see. This is the only
defence against that, and it is fifteen lines.

Copy `watch_parent()` from the template and call it at the top of `main()`,
before any weights are loaded. Three things about it are measured rather than
assumed, and each one is a way of getting it wrong:

- **Watch the process `HEARTH_PARENT_PID` names**, not this process's parent.
  hearth sets that variable when it starts a runner. A venv's `python.exe` may
  re-execute the base interpreter, which makes the runner a grandchild of a
  launcher that outlives hearth by design - watching it would never fire.
- **`os.getppid()` cannot detect a dead parent on Windows.** A process whose
  parent dies is not reparented there, so the field keeps naming the dead one.
  Open a handle at startup and wait on it: the handle stays valid afterwards,
  and a reused process id cannot fool it.
- **Leaving must not depend on being able to say so.** stderr is a pipe to the
  process that just died, so the message raises; an exception there kills the
  watching thread and leaves the runner holding the card, which is the whole
  failure being prevented.

Reporting progress and reading stdin are not substitutes. Both notice when the
caller's pipes close, which covers most of a run - but not the middle of a long
kernel, which is exactly when there is most to lose.

## Future

**Not implemented, and recorded so that it is not rediscovered.**

`capabilities` says which methods a runner has, but not **what they take and
give**. A caller that joins steps together therefore keeps its own table of
"`image_to_mesh` turns an image into a mesh", and adding a GPU model that does
something new - splitting a mesh into parts, say, or rigging one - means editing
that table in the caller as well as writing the runner.

The fix is to let the table say it:

```json
"kinds": {"image_to_mesh": {"takes": ["image"], "gives": "mesh"}}
```

**It is not worth doing yet.** Everything installed here is image-to-mesh, so the
table has one shape in it and the caller's copy is not wrong. It becomes worth
doing on the day a runner does something else - and on that day this note is what
stops it being designed twice.
