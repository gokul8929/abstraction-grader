"""Model profiles: everything that differs between served models, in one place.

A *setting* is the per-run knob that the grader treats as a separate configuration
(Qwen: reasoning effort; gpt-oss: reasoning effort; GLM: thinking on/off).

  python -m grader.models                 # list profiles
  python -m grader.models gpt-oss-120b    # print the vllm serve command for one profile
"""
from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class ModelProfile:
    key: str                                  # short name used on the command line and in run tags
    hf_id: str                                # Hugging Face repo id; also the served model name
    settings: dict[str, dict]                 # setting name -> request extras (see LLM.chat)
    default_setting: str
    gpus: int                                 # tensor-parallel GPUs held by vLLM (for GPU-seconds and Rs)
    serve_args: list[str]                     # vllm serve flags after the model id
    temperature: float | None = None          # None = read from the model's generation_config.json
    top_p: float | None = None
    top_k: int | None = None
    resend_reasoning: bool = False            # pass the model's reasoning back on later turns (gpt-oss needs this)
    force_tool_call: tuple[str, ...] = ()     # agent roles whose turns are sent with tool_choice="required"
    notes: str = ""

    def serve_command(self) -> str:
        return "vllm serve " + self.hf_id + " " + " ".join(self.serve_args)


COMMON = ["--max-model-len 65536", "--gpu-memory-utilization 0.92", "--enable-auto-tool-choice",
          "--enable-prompt-tokens-details", "--host 127.0.0.1", "--port 8000"]

PROFILES: dict[str, ModelProfile] = {p.key: p for p in [
    ModelProfile(
        key="qwen3.8-27b", hf_id="Qwen/Qwen3.8-27B",
        # effort goes to vLLM's top-level reasoning_effort field (as in pilot1); GRADER_EFFORT_VIA=template sends it
        # as a chat-template kwarg instead
        settings={e: {"reasoning_effort": e} for e in ("low", "medium", "xhigh")}, default_setting="xhigh",
        gpus=2, temperature=0.6, top_p=0.95, top_k=20,
        serve_args=["--tensor-parallel-size 2", "--language-model-only", "--reasoning-parser qwen3",
                    "--tool-call-parser qwen3_xml"] + COMMON,
        notes="bf16; settings = reasoning effort inserted by the chat template"),
    ModelProfile(
        key="gpt-oss-120b", hf_id="openai/gpt-oss-120b",
        settings={e: {"reasoning_effort": e} for e in ("low", "medium", "high")}, default_setting="medium",
        gpus=4, temperature=1.0, top_p=1.0, top_k=None, resend_reasoning=True, force_tool_call=("orchestrator",),
        serve_args=["--tensor-parallel-size 4", "--tool-call-parser openai"] + COMMON,
        notes="MXFP4 weights (Marlin MoE kernels on A100); temperature/top-p 1.0 per OpenAI's gpt-oss guidance; "
              "reasoning must be returned to the model between tool-call turns; orchestrator turns use "
              "tool_choice=required (otherwise it can write final_answer as a harmony channel, which vLLM drops)"),
    ModelProfile(
        key="glm-4.5-air", hf_id="zai-org/GLM-4.5-Air-FP8",
        settings={"think": {"chat_template_kwargs": {"enable_thinking": True}},
                  "nothink": {"chat_template_kwargs": {"enable_thinking": False}}}, default_setting="think",
        gpus=4, temperature=None, top_p=None, top_k=None,
        serve_args=["--tensor-parallel-size 4", "--tool-call-parser glm45", "--reasoning-parser glm45"] + COMMON,
        notes="FP8 weights (weight-only FP8 on A100); sampling from the model's generation_config.json; "
              "nothink = no reasoning text, so nothing exists at L3"),
    ModelProfile(
        key="muse-glimmer-30b", hf_id="meta-models/Muse-Glimmer-30B",
        settings={e: {"chat_template_kwargs": {"reasoning_strength": e}} for e in ("low", "medium", "xhigh")},
        default_setting="xhigh",
        gpus=2, temperature=1.0, top_p=0.95, top_k=64,
        serve_args=["--tensor-parallel-size 2", "--language-model-only", "--generation-config auto",
                    "--tool-call-parser muse_glimmer", "--reasoning-parser muse_glimmer"] + COMMON,
        notes="BF16 dense; reasoning strength via chat-template kwarg reasoning_strength (reasoning cannot be "
              "turned off); sampling per the model card; Meta notes outputs are not reproducible even with a seed"),
]}


def get(key: str | None) -> ModelProfile:
    key = key or os.environ.get("GRADER_MODEL", "qwen3.8-27b")
    if key not in PROFILES:
        raise SystemExit(f"unknown model {key!r}; known: {', '.join(PROFILES)}")
    return PROFILES[key]


def snapshot_dir(hf_id: str) -> Path | None:
    """The local Hugging Face snapshot of the served weights, if present."""
    hf = Path(os.environ.get("HF_HOME", f"/tmp/{os.environ.get('USER', '')}/hf"))
    ref = hf / "hub" / f"models--{hf_id.replace('/', '--')}" / "refs" / "main"
    if not ref.exists():
        return None
    d = ref.parent.parent / "snapshots" / ref.read_text().strip()
    return d if d.exists() else None


def generation_defaults(hf_id: str) -> dict:
    """temperature / top_p / top_k from the model's generation_config.json (what vLLM uses when none are sent)."""
    d = snapshot_dir(hf_id)
    f = d / "generation_config.json" if d else None
    if not f or not f.exists():
        return {}
    cfg = json.loads(f.read_text())
    return {k: cfg[k] for k in ("temperature", "top_p", "top_k") if k in cfg}


if __name__ == "__main__":
    if len(sys.argv) > 1:
        print(get(sys.argv[1]).serve_command())
    else:
        for p in PROFILES.values():
            print(f"{p.key:14s} {p.hf_id:28s} gpus={p.gpus} settings={','.join(p.settings)}  # {p.notes}")
