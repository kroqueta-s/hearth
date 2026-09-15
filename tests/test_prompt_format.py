# SPDX-License-Identifier: MIT
"""An image model says how it wants its prompt written, and the workflow decides half of it.

Two things were true before this and nobody could see either:

- **FLUX's negative prompt did nothing.** Its workflow encodes the text and then
  routes it through `ConditioningZeroOut`, at a `cfg` of 1.0, so whatever a person
  typed into the negative field was thrown away - and the field was offered all
  the same.
- **SDXL's workflow carries a negative prompt that never ran.** `negative`
  defaulted to "" and was always written over the workflow's own text, so leaving
  the field empty sent no negative prompt at all.

So `read_negative` reads the workflow, `capabilities` reports it as
`prompt_format` and uses the workflow's text as the default, and
`effective_params` fills that default in. **No ComfyUI, no GPU.**

Run it with hearth's own virtual environment::

    .venv\\Scripts\\python.exe .\\tests\\test_prompt_format.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from hearth import config, imagegen  # noqa: E402

# Two image models under names no real one has, pointing at the workflows that
# ship with hearth. **The environment is restored afterwards**, so this runs next
# to a real `.env` without changing what it declares.
_ENV = {
    "HEARTH_IMAGE_MODELS": "tagger,sentencer",
    "HEARTH_IMAGE_MODEL_TAGGER_CHECKPOINT": "a.safetensors",
    "HEARTH_IMAGE_MODEL_TAGGER_TXT2IMG": "sdxl_txt2img.json",
    "HEARTH_IMAGE_MODEL_TAGGER_IMG2IMG": "sdxl_img2img.json",
    "HEARTH_IMAGE_MODEL_TAGGER_CONTROLNET": "",
    "HEARTH_IMAGE_MODEL_TAGGER_PROMPT_STYLE": "tags",
    "HEARTH_IMAGE_MODEL_TAGGER_PROMPT_MAX_WORDS": "60",
    "HEARTH_IMAGE_MODEL_SENTENCER_CHECKPOINT": "b.safetensors",
    "HEARTH_IMAGE_MODEL_SENTENCER_TXT2IMG": "flux_txt2img.json",
    "HEARTH_IMAGE_MODEL_SENTENCER_IMG2IMG": "flux_img2img.json",
    "HEARTH_IMAGE_MODEL_SENTENCER_CONTROLNET": "",
    "HEARTH_IMAGE_MODEL_SENTENCER_PROMPT_STYLE": "natural",
    "HEARTH_IMAGE_MODEL_SENTENCER_PROMPT_MAX_WORDS": "",
}


class _Env:
    """Set some variables for the length of a block, then put them back."""

    def __init__(self, values: dict[str, str]) -> None:
        self.values = values
        self.saved: dict[str, str | None] = {}

    def __enter__(self) -> None:
        for key, value in self.values.items():
            self.saved[key] = os.environ.get(key)
            os.environ[key] = value

    def __exit__(self, *_: Any) -> None:
        for key, value in self.saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def test_a_zeroed_negative_is_not_used() -> None:
    """`ConditioningZeroOut` in front of the sampler means the text never arrives."""
    workflow = {
        "3": {"class_type": "KSampler", "inputs": {"cfg": 3.0, "negative": ["15", 0]}},
        "15": {"class_type": "ConditioningZeroOut", "inputs": {"conditioning": ["7", 0]}},
        "7": {"class_type": "CLIPTextEncode", "inputs": {"text": "blurry"}},
    }
    got = imagegen.read_negative(workflow)
    assert got == {"used": False, "default": "", "why": "ConditioningZeroOut"}, got


def test_cfg_one_means_no_negative() -> None:
    """At `cfg` 1.0 the unconditional branch is never evaluated, whatever feeds it."""
    workflow = {
        "3": {"class_type": "KSampler", "inputs": {"cfg": 1.0, "negative": ["7", 0]}},
        "7": {"class_type": "CLIPTextEncode", "inputs": {"text": "blurry"}},
    }
    got = imagegen.read_negative(workflow)
    assert got == {"used": False, "default": "", "why": "cfg 1"}, got


def test_a_negative_through_a_controlnet_is_followed() -> None:
    """A node in front of the sampler is walked through to the text behind it."""
    workflow = {
        "3": {"class_type": "KSampler", "inputs": {"cfg": 7.0, "negative": ["13", 1]}},
        "13": {
            "class_type": "ControlNetApplyAdvanced",
            "inputs": {"positive": ["6", 0], "negative": ["7", 0]},
        },
        "7": {"class_type": "CLIPTextEncode", "inputs": {"text": "lowres"}},
    }
    got = imagegen.read_negative(workflow)
    assert got == {"used": True, "default": "lowres", "why": ""}, got


def test_a_workflow_it_cannot_follow_counts_as_used() -> None:
    """**Unknown is not "ignored".** Greying a field that works would be the worse lie."""
    for workflow in (
        {},
        {"3": {"class_type": "KSampler", "inputs": {"cfg": 7.0, "negative": ["99", 0]}}},
        {"3": {"class_type": "KSampler", "inputs": {"cfg": "x", "negative": ["7", 0]}}},
    ):
        got = imagegen.read_negative(workflow)
        assert got == {"used": True, "default": "", "why": ""}, (workflow, got)


def test_the_shipped_workflows() -> None:
    """**The measurement this change rests on**, against the files hearth ships."""
    for name, used, default in (
        ("flux_txt2img.json", False, ""),
        ("flux_img2img.json", False, ""),
        ("sdxl_txt2img.json", True, "lowres, blurry, watermark, text"),
        ("sdxl_img2img.json", True, None),
        ("sdxl_controlnet.json", True, None),
    ):
        got = imagegen.read_negative(imagegen.load_workflow(name))
        assert got["used"] is used, (name, got)
        if default is not None:
            assert got["default"] == default, (name, got)


def test_capabilities_carry_the_format() -> None:
    """The table says it, and the negative default is the workflow's text."""
    with _Env(_ENV):
        tags = imagegen.capabilities("tagger")
        sentences = imagegen.capabilities("sentencer")
    assert tags["prompt_format"] == {"negative": True, "style": "tags", "max_words": 60}, tags
    assert tags["params"]["negative"]["default"] == "lowres, blurry, watermark, text"
    # The shipped FLUX workflow is both zeroed and at cfg 1.0; the cfg is found
    # first, and either is reason enough.
    assert sentences["prompt_format"] == {
        "negative": False,
        "negative_why": "cfg 1",
        "style": "natural",
    }, sentences
    assert sentences["params"]["negative"]["default"] == ""
    # **The shared table is not edited in place**: one model's default must not
    # become every model's.
    assert imagegen.COMMON_PARAMS["negative"]["default"] == ""


def test_an_undeclared_style_is_absent() -> None:
    """Nothing declared is nothing said, rather than a guess a caller cannot tell apart."""
    env = dict(_ENV, HEARTH_IMAGE_MODEL_TAGGER_PROMPT_STYLE="")
    with _Env(env):
        got = imagegen.capabilities("tagger")["prompt_format"]
    assert "style" not in got, got


def test_a_style_it_does_not_know_is_refused() -> None:
    """A misspelt style fails with a reason, and `all_capabilities` carries it."""
    env = dict(_ENV, HEARTH_IMAGE_MODEL_TAGGER_PROMPT_STYLE="prose")
    with _Env(env):
        try:
            imagegen.capabilities("tagger")
        except ValueError as exc:
            assert "prose" in str(exc), exc
        else:
            raise AssertionError("an unknown style was accepted")
        table = imagegen.all_capabilities()
    assert "error" in table["tagger"] and "error" not in table["sentencer"], table


def test_effective_params_use_the_models_default() -> None:
    """Sending no negative sends the workflow's, and sending one still wins."""
    with _Env(_ENV):
        silent = imagegen.effective_params("text_to_image", {"prompt": "a cat"}, "tagger")
        spoken = imagegen.effective_params(
            "text_to_image", {"prompt": "a cat", "negative": "dogs"}, "tagger"
        )
        flux = imagegen.effective_params("text_to_image", {"prompt": "a cat"}, "sentencer")
    assert silent["negative"] == "lowres, blurry, watermark, text", silent
    assert spoken["negative"] == "dogs", spoken
    assert flux["negative"] == "", flux
    assert imagegen.effective_params("text_to_image", {"prompt": "a cat"})["negative"] == ""


def test_the_spec_names_the_prompt_keys() -> None:
    """`config` reads both keys, so `.env.example` and the code agree on their names."""
    with _Env(_ENV):
        spec = config.image_model_spec("tagger")
    assert spec["prompt_style"] == "tags" and spec["prompt_max_words"] == "60", spec
    example = (REPO_ROOT / ".env.example").read_text(encoding="utf-8")
    for key in ("PROMPT_STYLE", "PROMPT_MAX_WORDS"):
        assert f"HEARTH_IMAGE_MODEL_SDXL_{key}=" in example, key


def main() -> int:
    """Run every test."""
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"  OK   {t.__name__}")
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f"  FAIL {t.__name__}: {type(exc).__name__}: {exc}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
