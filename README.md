# hearth

**Run several local image-to-3D models behind one interface, one at a time.**

Local 3D generation models each want their own build of torch, their own
hand-compiled extensions, and their own pinned libraries. **They cannot share a
virtual environment**, and on one GPU they cannot share the VRAM either. So each
model lives in its own repository with its own environment, and hearth starts
them as child processes, keeps exactly one loaded, and passes requests through.

```
your application
      |  one JSON object per line, over stdin/stdout
   hearth  ------------------------------------------ HTTP -> ComfyUI (optional)
      |
      +-- child process -> model A runner   (its own venv, its own torch)
      +-- child process -> model B runner   (its own venv, its own torch)
      +-- child process -> model C runner   (its own venv, its own torch)
```

**hearth holds no torch and no model.** It is a few hundred kilobytes of python.
Everything heavy is behind a process boundary, which is what lets models with
incompatible dependencies coexist at all.

## Why it might be useful

- **Adding a model is configuration, not a patch.** Runners are declared in
  `.env`, capabilities are returned as data, and nothing anywhere branches on a
  model's name. A caller written for one model works with the next one.
- **One thing holds the GPU.** Switching models unloads before it loads, and
  reports whether the memory actually came back.
- **Long jobs are visible.** Runners report counted progress, so a caller can
  tell work from a hang. **Nothing is ever estimated** — see below.
- **Starting from a template.** [`templates/runner/`](templates/runner) is a
  working runner that implements the whole contract and generates nothing;
  filling in one file makes it yours.

## Requirements

- Windows (the installer is PowerShell; the python is portable)
- **Python 3.12**
- Model runners, each installed separately. hearth only needs their paths.
- ComfyUI, **only** if you want hearth to generate the input image too.

## Install

```powershell
git clone https://github.com/kroqueta-s/hearth
cd hearth
.\install.ps1
```

That creates a virtual environment, installs three pure-python dependencies,
writes `.env`, and checks that hearth starts. If PowerShell refuses to run the
script, use `powershell -ExecutionPolicy Bypass -File .\install.ps1`.

Register the models you have, either during the install or afterwards:

```powershell
.\install.ps1 -Runner mymodel=C:\path\to\mymodel-repo

.\tools\add_runner.ps1 -Name mymodel -Path C:\path\to\mymodel-repo
.\tools\add_runner.ps1 -Name mymodel -Remove
```

`add_runner.ps1` works out the module and the python from the repository,
**starts the runner to check that it answers**, and only then writes the entry.
An entry that has never answered is worse than no entry: the failure would
otherwise surface in the middle of a generation.

## Quickstart

```powershell
.venv\Scripts\python.exe tools\rpc_call.py status
```

`status` lists the installed models and what each one can do. To generate,
put the arguments in a file and pass it:

```powershell
'{"model":"mymodel","image_path":"C:\\in.png"}' | Set-Content args.json
.venv\Scripts\python.exe tools\rpc_call.py image_to_mesh --params-file args.json
```

**Do not pass JSON to `--params` from PowerShell**: it strips the double quotes
on the way to a native executable.

## Methods

| Method | Takes | Gives |
|---|---|---|
| `status` | — | What is installed, what is loaded, what each model can do |
| `load` / `unload` | `model` | Switch models, or free the GPU |
| `image_to_mesh` | `model`, `image_path` | A mesh |
| `texture_mesh` | `model`, `mesh_path`, `image_path` | A texture on a mesh you already have |
| `text_to_image` | `prompt` | An image |
| `image_to_image` | `image_path`, `prompt`, `denoise` | A reworked image |
| `sketch_to_image` | `sketch_path`, `prompt`, `strength` | An image following a sketch |
| `text_to_mesh` / `sketch_to_mesh` | both stages at once | A mesh |

**Making the image and making the mesh are separate on purpose.** Spending a
minute turning an image you dislike into a mesh helps nobody, so the usual path
is `*_to_image`, look at it, then `image_to_mesh`. The combined methods are a
shortcut over exactly those two steps and do nothing else.

The image methods need ComfyUI. `image_to_mesh` and `texture_mesh` do not.

**What a model supports is data, not a name.** Read `capabilities` from `status`
rather than checking which model it is; that is what keeps a caller working when
a new model arrives.

## Progress: counted, never estimated

Runners report which stage they are in, and for loops they can count, how far
through they are. `tools/rpc_call.py` draws it:

```
[   48.6s] shape      [#####-------------------]  20%  (6/30)
[  318.6s] texture    [###########-------------]  46%  (7/15)
```

**There is no ETA and no overall percentage, deliberately.** The first run of a
loop can be an order of magnitude slower than every run after it, because
kernels are tuned once per machine; a prediction built on a stored constant is
therefore worst exactly when it is most wanted. A stage whose length is unknown
reports a step number and nothing more, and a percentage is never shown without
a real denominator. The rules are in
[§8 of the contract](docs/runner_contract.md#8-progress).

## Writing a runner

**[`docs/runner_contract.md`](docs/runner_contract.md) is the specification**,
and [`templates/runner/`](templates/runner) is a working implementation of it
that generates nothing. Copy the template, fill in `pipeline.py`, and register
it with `add_runner.ps1`.

The template is not pseudocode: `tests/test_template_runner.py` starts it and
holds a real conversation with it, so **the build fails if the template stops
matching the contract**.

## Tests

`pytest` is not used. Each file under `tests/` is a script:

```powershell
.venv\Scripts\python.exe tests\test_config.py            # the rules that keep hearth a relay
.venv\Scripts\python.exe tests\test_template_runner.py   # the template obeys the contract
.venv\Scripts\python.exe tests\test_model_switch.py      # switching models, on real hardware
```

The first two need nothing but the install. **The third uses the GPU** and needs
runners installed; point `HEARTH_TEST_IMAGE` at an input image first.

## Troubleshooting

- **`status` shows a model with an `error`.** hearth started that runner and it
  did not answer. Run the runner's own tests in its repository: the fault is
  there, not here.
- **A generation fails and you cannot tell where.** Send the same request with
  `tools/rpc_call.py`. It cuts your application out of the picture and shows the
  runner's own progress and error.
- **hearth refuses to load anything, saying the GPU is taken.** Something is
  listening on `HEARTH_GPU_BUSY_PORT`. Stop it, or set the port to 0 to disable
  the check.
- **`Unexpected UTF-8 BOM` from `--params-file`.** PowerShell's
  `Set-Content -Encoding utf8` always writes a BOM. hearth reads argument files
  as `utf-8-sig` so this should not happen; if you see it elsewhere, that is the
  cause.
- **Nothing is reported for minutes.** Check whether the runner is emitting a
  `heartbeat`. If it is, it is working; if it is not, `tests/harness.py` is the
  tool that gives up and collects a diagnosis instead of waiting.

## License

**MIT** ([`LICENSE`](LICENSE)). Models and their weights carry their own
licences, which are usually not MIT — read them before commercial use. This
repository contains no model code and no weights.
