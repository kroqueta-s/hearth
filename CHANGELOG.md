# Changelog

What changed, and **why it mattered**. A line here earns its place by being
something a caller would otherwise have to discover: a contract that moved, a
promise that turned out to be false, a failure that had no symptom.

Dates are when the change was made. There are no releases yet; the contract
version in [`docs/runner_contract.md`](docs/runner_contract.md) is what a runner
should track.

## Unreleased

### Runner contract 3

- **A result names its settings `params_used`**, not `params` (§5). Under the
  old name a caller found nothing and nothing errored, so the settings needed to
  repeat a generation were simply gone. hearth promotes `params` from an older
  runner and says so on stderr.
- **`capabilities` reports the `contract` version it was written against** (§3),
  so a caller can tell a missing field from an old runner. hearth never refuses
  a runner over it.
- **`method_params` declares the settings of every method that is not
  `image_to_mesh`** (§3). A texture stage has no use for an octree resolution;
  one runner accepted those and threw them away, so changing a setting produced
  no error and no difference.
- **`up_axis` and `forward_axis` are on the result** (§5), `null` where they were
  not measured. **hearth never fills one in.** A mesh imported on the wrong axis
  renders perfectly correctly and prints mirrored, so a plausible guess turns
  "nobody looked" into "known", and deletes the only true thing anyone can say.
- **Nothing is written under its final name until it is complete** (§9). A
  cancel ends a runner outright, and a truncated file wearing the finished name
  is read by the next step without complaint. A format that writes several files
  at once is written into a directory and **the directory** is renamed.
- **A runner ends itself when the process that started it is gone** (§10). See
  below; this is the one that had no symptom at all.

### Nothing is left holding the card

- **`shutdown` during a generation kills the runner first**, rather than asking
  it to unload and queueing behind the work. It used to look like a hang, and
  the answer to a hang is to kill hearth - which on Windows leaves the runner
  alive with the whole card. Nothing errored; everything afterwards was several
  times slower.
- **Requests still queued when a shutdown starts are answered with an error**
  instead of run. One arriving a moment too late started a runner after
  shutdown had already taken its list of them.
- **A runner whose hearth crashes ends itself.** hearth passes
  `HEARTH_PARENT_PID` and the runner waits on that process. Three details of
  that are not optional and are written down in contract §10, because each looks
  fine and none of them works: watching its own parent (a venv `python.exe` may
  re-execute the base interpreter, so the parent is a launcher that outlives
  hearth), `os.getppid()` on Windows (no reparenting, so it names the dead one
  forever), and letting the goodbye message decide whether it exits (stderr is a
  pipe to the process that just died).
- `HEARTH_LOCK_PORT` defaults to **8011** rather than 0. Two model loads on one
  card do not fail, they fall back to shared memory and run several times
  slower. Set it to 0 to disable the lock, which is what the tests do.

### Cancelling an image actually stops it

- **The wait on ComfyUI is interruptible.** It polls the history for up to
  `COMFY_TIMEOUT_SEC` - half an hour by default - and had no way in: `cancel`
  answered `{"canceled": false, "why": "the prompt had already finished"}`,
  which was untrue in every particular, while the image went on being made.
- **`cancel` learns which prompt to drop.** The id does not exist when the work
  starts, so it is recorded after the workflow is submitted - without forgetting
  a cancel that arrived in between, and dropping the prompt at once if one did.
- `canceled: true` now means the request ends. `dropped_from_queue` says whether
  ComfyUI was still holding the work, and `why` explains a false one.

### Other

- **A reworked image reports the image it made.** `image_to_image` echoed its
  input under `image_path` - the key the output uses - so the new image was
  written and its location never reported. Nothing errored, because the path
  left behind was a real file. The input comes back as `source_path` now, with
  `source_argument` naming which argument it was.
- **Images are written beside their final name and renamed**, like everything
  else (§9). A cancel can land mid-download.
- **`sys.stdout` is not the only thing redirected**: file descriptor 1 itself
  points at stderr, so a C extension writing straight to it cannot corrupt the
  protocol. Measured in a sibling repository: `pymeshfix` emitted hundreds of
  lines that way.
- `RunnerProcess.start()` is guarded by its own lock. `capabilities` is answered
  on the control thread while the GPU thread generates, and both could find the
  process stopped and each start one.
- ComfyUI's reachability check is cached for ten seconds. It cost three seconds
  on the thread that reads stdin, which is the thread a cancel arrives on.
