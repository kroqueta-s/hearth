# A runner template

A working runner that implements [the contract](../../docs/runner_contract.md)
and generates nothing. **Copy it, replace one function, and you have a runner.**

It is not pseudocode: `tests/test_template_runner.py` starts this template and
holds a real conversation with it, so **if the contract changes and the template
stops matching, the build fails** rather than the template quietly rotting.

## What to do

1. Copy `templates/runner/` into a new repository, and rename the package
   directory `runners/example/` to `runners/<your model>/`.
2. Fill in `pipeline.py`. **It is the only file with anything to do**: load the
   model, and turn one image into one mesh.
3. Adjust `capabilities()` in `__main__.py` so it declares what your model can
   actually do and which settings it takes.
4. Register it:

   ```powershell
   .\tools\add_runner.ps1 -Name <your model> -Path <your repository>
   ```

## What is already done for you

- **The protocol loop**: one JSON object per line, one answer per request,
  errors returned rather than raised out of the process.
- **The stdout guard.** Model code prints to stdout, and a single stray line
  breaks the protocol. The real stdout is duplicated and hidden at startup.
- **`capabilities` without loading anything**, which is what lets a caller show
  what is available without paying for it.
- **Progress that is counted, never estimated**, including the helper that reads
  the count out of a diffusers scheduler or a `tqdm` loop without modifying the
  model's code.
- **Configuration from `.env`**, so no path and no model name is ever in the
  source.

## What to keep

- **Never let an exception escape.** A runner that dies leaves its caller
  waiting; a runner that answers with an error lets the caller say why.
- **Preprocessing is yours.** Whether the image needs its background removed,
  and how, differs per model and hearth will not do it for you.
- **Only one model is loaded at a time.** Release the VRAM in `unload` and mean
  it.
- **Do not report an ETA or an overall percentage.** See §8 of the contract.
