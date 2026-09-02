# CLAUDE.md — working on hearth

Read this first, then [`README.md`](README.md) and
**[`docs/runner_contract.md`](docs/runner_contract.md), which is the
specification everything else serves**.

**If `docs/local/` exists, read `docs/local/00_operator_notes.md` too.** It holds
this machine's setup, the environment-specific traps, and how the operator wants
to be told things. It is deliberately not tracked (see below).

---

## Language: **what ships is English, what explains is not**

This repository may become public, and so may others in this family. **Assume
everything tracked by git will be read by a stranger.**

| Where | Language | Why |
|---|---|---|
| **Everything tracked by git** — code, comments, docstrings, README, the contract, commit messages, `.env.example` | **English** | It is published, or may be. A reader who does not share the author's language should not be shut out of the part they have to use. |
| **`docs/local/`** — design notes, verification reports, the record of what was tried and rejected | The author's language | It is **not tracked**, and never will be. It carries the operator's environment, dead ends, and reasoning that would only mislead a stranger reading it as documentation. |

Two consequences worth stating plainly:

- **Do not translate the internal documents.** They are not drafts of the public
  ones; they are a different kind of document with a different reader.
- **Do not put internal reasoning into a tracked file** to keep it together with
  the code. If it explains *why this machine*, it belongs in `docs/local/`.

**A repository whose publication is undecided is treated as public.** Deciding
later is cheap; discovering later that the history is full of internal notes is
not.

## Architecture: **break these and the design stops working**

1. **hearth never imports torch.** Only a runner holds the GPU. This is what
   allows runners to have dependencies that contradict each other.
2. **Nothing here imports `bpy`.** Blender's python is GPL and would end the MIT
   licence.
3. **A runner never imports hearth.** Each one has to remain shippable as its own
   repository.
4. **One model per virtual environment, one loaded at a time.** There is one GPU.
5. **Capabilities are data. Nothing branches on a model's name** (contract §3).
6. **hearth does not validate a runner's arguments.** Only the runner knows what
   its values mean (contract §3).
7. **No path, port or model name is written in code.** `.env` is the authority.
8. **Nothing writes to the protocol's stdout.** Model code prints; replacing
   `sys.stdout` is not enough, because compiled extensions write to file
   descriptor 1 directly (contract §1).
9. **stderr is always drained.** Left unread, the pipe fills and the runner stops.
10. **Report what you counted, never an estimate** (contract §8).
11. **Generating is queued and serial; control methods are answered while it
    runs** (`docs/protocol.md` §2). Making everything serial again would be a
    one-line simplification that silently removes the only thing keeping an
    interface usable during a generation - and `warm` and `cancel` with it.
12. **No method runs a whole flow.** Joining `text_to_image` to `image_to_mesh`
    belongs to the caller, because only the caller knows whether a person is
    about to look at the image (§3.2). A convenience method that chains them
    looks helpful and takes that choice away.

`tests/test_config.py` enforces 1, 2 and 7 as far as they can be checked
statically, and `tests/test_protocol.py` holds hearth to 11 - **without a GPU**,
through `selftest_long_job`.

## Working on model code

**Never modify a model's own source.** It gets replaced wholesale on the next
update, and a local edit is silently lost. Everything a model needs — a
replacement for a missing extension, a shim for a renamed API, a count of its
progress — is installed at launch time, from the runner. This is a hard rule and
it has never needed an exception.

## Style

- Python: `ruff` and `black`, line length 100, **type hints everywhere**.
- **Comments say why, not what.** The code already says what.
- **A default that came from a measurement should say so**, and one that did not
  should say that too.
- Tests are hand-written scripts under `tests/`. **Do not add `pytest`.**
- **Run the tests after changing anything they cover.**

## Do not

- Add torch, or any GPU dependency, to hearth.
- Import `bpy` anywhere.
- Write a model's name into code.
- Modify a model's own source (see above).
- Publish, or track in git, anything from `docs/local/`.
