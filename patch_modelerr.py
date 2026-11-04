"""Patch: a model output the server cannot parse no longer crashes a run.

Run from ~/abstraction-grader after patch_robust.py:  python patch_modelerr.py
gpt-oss sometimes emits a malformed tool-call header; vLLM then answers HTTP 500
("unexpected tokens remaining in message header"). With a fixed seed every retry reproduces the same
output, so retries are made with a shifted seed (seed + 1009*k, recorded in the llm_call event's seed).
If all attempts fail, ModelOutputError is raised: a worker step ends incomplete, and an orchestrator
failure ends the run with end reason 'model_output_error' instead of a crash.
"""
from pathlib import Path


def sub(p, old, new):
    s = Path(p).read_text()
    assert s.count(old) == 1, f"{p}: expected text not found exactly once:\n{old[:120]}"
    Path(p).write_text(s.replace(old, new))


L, A = "grader/llm.py", "grader/agent.py"
if "ModelOutputError" in Path(L).read_text():
    raise SystemExit("already patched; nothing to do")

sub(L, '''class ContextExhausted(Exception):
    """The conversation no longer fits in the model's context window."""
''', '''class ContextExhausted(Exception):
    """The conversation no longer fits in the model's context window."""


class ModelOutputError(Exception):
    """The model kept producing output the server could not parse (e.g. a malformed tool-call header)."""


def _unparseable(e: Exception) -> bool:
    s = str(e)
    return "message header" in s or "unexpected tokens" in s or "HarmonyError" in s
''')
sub(L, '''            except RETRYABLE as e:
                self.rec.emit(LLMCall(start=start, end=time.time(), attempt=attempt,
                                      error=f"{type(e).__name__}: {e}", **meta))
                if attempt == self.max_attempts:
                    raise
                await asyncio.sleep(2 ** attempt)''', '''            except RETRYABLE as e:
                self.rec.emit(LLMCall(start=start, end=time.time(), attempt=attempt,
                                      error=f"{type(e).__name__}: {e}", **meta))
                bad_output = isinstance(e, openai.InternalServerError) and _unparseable(e)
                if attempt == self.max_attempts:
                    if bad_output:
                        raise ModelOutputError(str(e)[:500]) from e
                    raise
                if bad_output and self.seed is not None:   # same seed would reproduce the same bad output
                    kwargs["seed"] = meta["seed"] = self.seed + 1009 * attempt
                await asyncio.sleep(2 ** attempt)''')

sub(A, "from grader.llm import LLM, ContextExhausted", "from grader.llm import LLM, ContextExhausted, ModelOutputError")
sub(A, '''        except ContextExhausted:                       # the orchestrator's own context is full
            return AgentResult("", "context_exhausted", 0)''',
       '''        except ContextExhausted:                       # the orchestrator's own context is full
            return AgentResult("", "context_exhausted", 0)
        except ModelOutputError:                       # the orchestrator's output could not be parsed
            return AgentResult("", "model_output_error", 0)''')
sub(A, '''            except ContextExhausted:
                return StepResult(step.step_id, f"Stopped: context limit reached. Last message: {last[:1000]}",
                                  finished=False)''', '''            except ContextExhausted:
                return StepResult(step.step_id, f"Stopped: context limit reached. Last message: {last[:1000]}",
                                  finished=False)
            except ModelOutputError:
                return StepResult(step.step_id, f"Stopped: model output could not be parsed by the server. "
                                                f"Last message: {last[:1000]}", finished=False)''')
print("patched: llm.py (seed-shifted retries on unparseable output), agent.py (model_output_error)")
