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
`text_to_mesh`, `multi_image_to_mesh`, `texture_mesh`, `segment_mesh` (§5a).

**`image_to_mesh` is required of a runner that generates meshes**, which is not
every runner. One that only answers questions about a mesh it is given declares
`image_to_mesh: false` and implements the optional method it does have; hearth
never calls a method a capability table did not claim, so nothing else changes.

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
    "texture_mesh": true,
    "segment_mesh": false
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
  "vram_peak_gb": {"image_to_mesh": 18.0, "texture_mesh": 11.5},
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
- **`vram_peak_gb` is optional**: per method, the most dedicated VRAM the
  runner's process family reaches **at its default settings, weights included,
  measured** - not estimated. hearth uses it to refuse a generation while other
  processes hold that room (`VramShortError`, [protocol §6](protocol.md)),
  because a runner started without it can have its process aborted by the
  driver minutes in. **Round a measured peak up**: a sample taken every second
  misses the top of a short surge. A runner that does not declare it is never
  refused, and one whose settings can raise the peak well above the default
  should say so in `notes`.

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
- `extra` holds whatever else the run produced, by name. hearth passes it
  through untouched. **It is not only for intermediates**: a runner that makes
  two representations of one generation - a printable mesh and a textured copy
  of it, say - puts the second here rather than inventing a method for it. How a
  caller is meant to read that is in
  [`docs/protocol.md` §3.1a](protocol.md), and the short of it is that a key is
  used when it is there and never expected because of which model answered.
- **A textured copy goes under `extra.textured_glb`, as a binary glTF.** One
  file: the same geometry as `mesh_path`, in the same frame and scale, with its
  UVs, the colour texture, and - when the model makes them - metalness and
  roughness packed into a glTF `metallicRoughnessTexture` (blue and green).
  `mesh_path` stays the PLY. This is the one `extra` key with a fixed meaning,
  decided 2026-09-15 across hearth and its callers: **geometry travels as PLY,
  appearance as GLB**, because the face order and the absence of any conversion
  matter only to the first. Write it without `bpy` (trimesh is enough), and
  write it beside its name then rename it (§9).

## 5a. `segment_mesh`: a mesh in, a label per face out

**Not every runner generates.** A runner may instead answer a question about a
mesh it is handed, and the first of those is: *where do the parts of this thing
meet?* A runner that can answer it declares `segment_mesh: true` and takes:

| Argument | Required | Meaning |
|---|:--:|---|
| `mesh_path` | yes | **Absolute path** to a PLY. Not glTF and not `.obj` |
| `out_dir` | yes | **Absolute path** to write into. hearth makes one per run |
| anything in `method_params.segment_mesh` | no | The runner's own settings for this method |

and answers:

```json
{
  "segments_path": "C:/.../out/segments.npz",
  "faces": 395712,
  "k_values": [2, 3, 4, "...", 20],
  "faces_sha256": "1f0c…",
  "elapsed_sec": 301.5,
  "peak_rss_mb": 6685.0,
  "params_used": {"n_point_per_face": 100}
}
```

- **`segments_path` is an `.npz` holding `k_values` `(K,)` and `labels_by_k`
  `(K, F)`, both `int16`.** Every number of segments comes back at once, because
  the expensive part is the field the labels are cut out of and not the cutting:
  a caller's slider over K is then an array lookup rather than another run.
  Row *i* is the labelling for `k_values[i]`, and its labels are `0..k-1`.
- **`faces_sha256` is the hash of the input's face array**, and it is what makes
  the answer usable at all. A label array means nothing unless face *i* on the
  way out is face *i* on the way in, and **nothing about a mislabelled mesh
  looks wrong**: it renders perfectly and prints as the wrong part. So the
  runner hashes the faces it read and the caller compares that against its own
  copy before believing a single label. The canonical form is fixed so that two
  libraries reading one file agree: **little-endian `int32`, C order, `(F, 3)`,
  SHA-256 of those bytes.**
- **There is no `mesh_path`, and no `up_axis`.** Nothing was generated and
  nothing was moved, so there is nothing to orient - and hearth does not ask for
  one here (§5's axes are about a mesh result).
- `peak_rss_mb` is worth reporting when the method is expensive in memory rather
  than in VRAM, which is the case for anything running on the CPU.

**The face order is the runner's to preserve, and its to check.** Reading a mesh
with a library that merges duplicate vertices by default, or writing it out
through a format that splits them, renumbers every face after the first change.
A runner that cannot preserve the order must fail rather than answer.

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

**A death is not always the runner's fault, so a generating call is tried once
more** (`HEARTH_GENERATE_RETRIES`, default 1). A driver can take the process
away mid-decode - on gfx1151 a large one hits `PAL failed to submit CMD!` and
torch's abort handler ends the process - and there is nothing for a runner to
catch, because the runner is gone. The retry costs a full load of the weights,
so the result carries `attempts` when it took more than one, and `progress`
reports a `retry` stage while it happens.

**Only a generating call, and only a death.** A cancel ends the process on
purpose (§9) and is never retried; neither is an error a runner *answered*
with, because the runner is still running and will say the same thing again.

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

**That day has arrived, and it was cheaper than expected.** `segment_mesh` (§5a)
is a runner that does something else: a mesh in, labels out. Adding it cost the
caller one line in its own table and cost this document a section - **and no
caller branched on a name**, because the capability table already said which
runners could be asked. So the shape of the answer stayed a document rather than
becoming data.

**Still not worth doing, for a different reason than before.** The argument for
`kinds` was that a caller would otherwise edit a table per new method; the
measured cost of doing exactly that, once, was one line. It becomes worth doing
when a caller has to join steps it was not written for - a saved flow naming a
method that did not exist when the flow was written - and not before.
