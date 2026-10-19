"""Plan-then-execute multi-agent system.

Orchestrator: sees the question and the dataset description, never touches data.
  1. commits a plan (PlanCommitted)          -> O bucket: commitment happens before any data access
  2. workers run the steps in dependency order
  3. sees step summaries -> final_answer, or revise_plan (new PlanCommitted linked to the old one)
Workers: one per step; only that step's databases (+ python); log their judgements as
  Interpretation events (D2); end with finish_step.
Every prompt part is tagged with its origin (L2) and every call names its prompt author (L1).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml
from pydantic import ValidationError

from grader.client import ENGINE_OF, ToolHub
from grader.llm import LLM
from grader.record import Interpretation, MessagePart, PlanCommitted, PlanStep, RunRecorder, new_id

ORCH_TEMPLATE = "orchestrator_v2"
WORKER_TEMPLATE = "worker_v2"

ORCH_SYSTEM = """You are the orchestrator of a data-analysis team. You answer a question over the databases described below, but you never query data yourself.

First, call commit_plan. Each step is carried out by a separate worker that can query only the databases you list for that step, and can run Python. A worker sees only its own step description plus the summaries of the steps in its depends_on, so write every description so it stands on its own: say what to find or compute, in which database, and exactly what to report back. Keep plans short (1-6 steps). The last step must produce the final answer.

When the workers finish you will see their summaries. Then call final_answer with the answer to the question, or call revise_plan if a step failed or the results do not answer the question. In final_answer, state the answer itself first (the exact value, name or list), then at most a short explanation."""

WORKER_SYSTEM = """You are a worker on a data-analysis team. Do exactly the step you are given, using the tools.

- Look at the tables (list_tables) before you query them.
- Tool results may be cut short. The full result of any earlier call can be loaded in Python with load('<call_id>').
- Whenever you make a judgement about the data - which table or column to use, what a column means, how to match or normalise values, where a filter boundary lies - call record_interpretation before acting on it. Say where it comes from: "data" (cite the call_ids whose results show it), "model_knowledge" (your own background knowledge, not in the data), or "question" (read from the question's wording).
- When you are done, call finish_step with a short summary: what you found, the key values, and the call_ids that hold the results."""

NUDGE_TOOL = "Call finish_step to report your result, or call a tool to continue."
NUDGE_PLAN = "Call commit_plan now."
NUDGE_FINAL = "Call final_answer or revise_plan now."
CUT_OFF = "Your last reply was cut off by the length limit. Continue, and keep your reasoning shorter."

KINDS = ["source_choice", "column_meaning", "metric_definition", "join_key", "normalisation",
         "filter_boundary", "entity_match", "other"]

_STEPS_SCHEMA = {"type": "array", "items": {"type": "object", "properties": {
    "step_id": {"type": "string", "description": "short id, e.g. s1"},
    "description": {"type": "string"},
    "databases": {"type": "array", "items": {"type": "string"}},
    "depends_on": {"type": "array", "items": {"type": "string"}}},
    "required": ["step_id", "description", "databases"]}}


def _fn(name: str, description: str, properties: dict, required: list[str]) -> dict:
    return {"type": "function", "function": {"name": name, "description": description,
            "parameters": {"type": "object", "properties": properties, "required": required}}}


COMMIT_PLAN = _fn("commit_plan", "Commit the plan. It is recorded and cannot be edited afterwards.",
                  {"steps": _STEPS_SCHEMA}, ["steps"])
REVISE_PLAN = _fn("revise_plan", "Replace the current plan with a new one; all its steps are run again.",
                  {"reason": {"type": "string"}, "steps": _STEPS_SCHEMA}, ["reason", "steps"])
FINAL_ANSWER = _fn("final_answer", "Give the final answer to the question.",
                   {"answer": {"type": "string"}}, ["answer"])
RECORD_INTERP = _fn("record_interpretation", "Record a judgement you are making about the data.",
                    {"kind": {"type": "string", "enum": KINDS},
                     "subject": {"type": "string", "description": "e.g. a table.column"},
                     "statement": {"type": "string"},
                     "source": {"type": "string", "enum": ["data", "model_knowledge", "question"]},
                     "evidence": {"type": "array", "items": {"type": "string"},
                                  "description": "call_ids whose results support this (required when source is data)"}},
                    ["kind", "subject", "statement", "source"])
FINISH_STEP = _fn("finish_step", "Finish this step and report back to the orchestrator.",
                  {"summary": {"type": "string"},
                   "result_call_ids": {"type": "array", "items": {"type": "string"}}}, ["summary"])
ORCH_TOOLS = [COMMIT_PLAN, REVISE_PLAN, FINAL_ANSWER]   # always all three: keeps the prompt prefix cacheable


def dataset_context(dataset_dir: str | Path, hints: bool = False) -> tuple[str, dict[str, str]]:
    """Description text the orchestrator sees, and {database name: engine}."""
    d = Path(dataset_dir).expanduser()
    cfg = yaml.safe_load(open(d / "db_config.yaml"))["db_clients"]
    dbs = {name: ENGINE_OF[c["db_type"]] for name, c in cfg.items()}
    desc_file = d / ("db_description_withhint.txt" if hints else "db_description.txt")
    desc = desc_file.read_text() if desc_file.exists() else ""
    listing = "\n".join(f"- {n} ({e})" for n, e in dbs.items())
    return f"Databases:\n{listing}\n\nDescription:\n{desc}".strip(), dbs


@dataclass
class StepResult:
    step_id: str
    summary: str
    result_call_ids: list[str] = field(default_factory=list)
    finished: bool = True


@dataclass
class AgentResult:
    answer: str
    end_reason: str
    plans: int


class DataAgent:
    def __init__(self, rec: RunRecorder, llm: LLM, hub: ToolHub, dataset_dir: str | Path,
                 effort: str | None = None, hints: bool = False, max_worker_turns: int = 25,
                 max_revisions: int = 2, max_nudges: int = 2):
        self.rec, self.llm, self.hub, self.effort = rec, llm, hub, effort
        self.context, self.dbs = dataset_context(dataset_dir, hints)
        self.max_worker_turns, self.max_revisions, self.max_nudges = max_worker_turns, max_revisions, max_nudges

    # ---------------- orchestrator ----------------
    def _validate(self, steps: list[dict]) -> str | None:
        ids = [s.get("step_id") for s in steps]
        if not steps:
            return "The plan has no steps."
        if len(set(ids)) != len(ids) or not all(ids):
            return "Every step needs a unique, non-empty step_id."
        for s in steps:
            bad = [d for d in s.get("databases", []) if d not in self.dbs]
            if bad:
                return f"Step {s['step_id']}: unknown databases {bad}. Known: {list(self.dbs)}"
            missing = [d for d in s.get("depends_on", []) if d not in ids]
            if missing:
                return f"Step {s['step_id']}: depends_on unknown steps {missing}."
        try:
            self._order(steps)
        except ValueError as e:
            return str(e)
        return None

    @staticmethod
    def _order(steps: list[dict]) -> list[dict]:
        done, order, pending = set(), [], list(steps)
        while pending:
            ready = [s for s in pending if set(s.get("depends_on", [])) <= done]
            if not ready:
                raise ValueError("The plan has a dependency cycle.")
            s = ready[0]
            order.append(s); done.add(s["step_id"]); pending.remove(s)
        return order

    async def _orchestrate(self, msgs: list[MessagePart], tools: list[dict], nudge: str):
        """One orchestrator decision: returns (tool call, llm result); retries on invalid output."""
        for _ in range(self.max_nudges + 1):
            r = await self.llm.chat(msgs, agent_id="orchestrator", role="orchestrator", prompt_author="harness",
                                    tools=ORCH_TOOLS, effort=self.effort)
            msgs.append(r.as_message())
            names = {t["function"]["name"] for t in tools}
            call = next((c for c in r.tool_calls if c["name"] in names), None)
            for c in r.tool_calls:                         # every tool call needs a reply
                if c is not call:
                    msgs.append(self._tool_reply(c, "Ignored: call exactly one of " + ", ".join(sorted(names))))
            if call and call["parse_error"]:
                msgs.append(self._tool_reply(call, f"ERROR: {call['parse_error']}"))
                continue
            if call:
                return call, r
            msgs.append(MessagePart(role="user", origin="system_template", template_id=ORCH_TEMPLATE,
                                    content=CUT_OFF if r.finish_reason == "length" else nudge))
        return None, r

    @staticmethod
    def _tool_reply(call: dict, content: str, origin: str = "tool_result") -> MessagePart:
        return MessagePart(role="tool", origin=origin, tool_call_id=call["id"], content=content)

    async def run(self, question: str) -> AgentResult:
        msgs = [MessagePart(role="system", origin="system_template", template_id=ORCH_TEMPLATE, content=ORCH_SYSTEM),
                MessagePart(role="user", origin="task_context", content=self.context),
                MessagePart(role="user", origin="user_question", content=f"Question: {question}")]
        tools, nudge, plan_id, version, reason = [COMMIT_PLAN], NUDGE_PLAN, None, 0, None

        while True:
            # --- get a valid plan ---
            for _ in range(self.max_nudges + 1):
                call, r = await self._orchestrate(msgs, tools, nudge)
                if call is None:
                    return AgentResult("", "no_plan" if version == 0 else "no_final_answer", version)
                if call["name"] == "final_answer":
                    msgs.append(self._tool_reply(call, "Final answer recorded."))
                    return AgentResult(str(call["arguments"].get("answer", "")), "answered", version)
                steps = call["arguments"].get("steps", [])
                err = self._validate(steps)
                if err is None:
                    break
                msgs.append(self._tool_reply(call, f"ERROR: plan rejected: {err} Fix it and call {call['name']} again."))
            else:
                return AgentResult("", "invalid_plan", version)

            version += 1
            reason = call["arguments"].get("reason") if call["name"] == "revise_plan" else None
            new_plan_id = new_id("plan")
            plan = [PlanStep(step_id=s["step_id"], description=s["description"], databases=s.get("databases", []),
                             depends_on=s.get("depends_on", []), assigned_agent=f"worker-{s['step_id']}-v{version}")
                    for s in steps]
            self.rec.emit(PlanCommitted(plan_id=new_plan_id, version=version, author="orchestrator", steps=plan,
                                        revises=plan_id, revision_reason=reason, llm_call_id=r.llm_call_id))
            plan_id = new_plan_id

            # --- execute ---
            results: dict[str, StepResult] = {}
            for s in self._order(steps):
                step = next(p for p in plan if p.step_id == s["step_id"])
                results[step.step_id] = await self._work(step, [results[d] for d in step.depends_on])
            report = "\n\n".join(f"[{sid}] {'' if res.finished else '(INCOMPLETE) '}{res.summary}"
                                 for sid, res in results.items())
            msgs.append(self._tool_reply(call, f"Plan v{version} executed. Step summaries:\n\n{report}",
                                         origin="subagent"))
            tools = [FINAL_ANSWER] + ([REVISE_PLAN] if version <= self.max_revisions else [])
            nudge = NUDGE_FINAL if version <= self.max_revisions else "Call final_answer now."

    # ---------------- workers ----------------
    async def _work(self, step: PlanStep, deps: list[StepResult]) -> StepResult:
        agent = step.assigned_agent
        engines = sorted({self.dbs[d] for d in step.databases}) + ["python"]
        tools = self.hub.openai_tools(engines) + [RECORD_INTERP, FINISH_STEP]
        msgs = [MessagePart(role="system", origin="system_template", template_id=WORKER_TEMPLATE, content=WORKER_SYSTEM),
                MessagePart(role="user", origin="orchestrator",
                            content=f"Your step ({step.step_id}): {step.description}\n"
                                    f"Databases you may use: {', '.join(step.databases) or 'none'}")]
        if deps:
            msgs.append(MessagePart(role="user", origin="subagent", content="Results of earlier steps:\n\n" + "\n\n".join(
                f"[{d.step_id}] {d.summary}\nresult call_ids: {d.result_call_ids}" for d in deps)))

        nudges, last = 0, ""
        for _ in range(self.max_worker_turns):
            r = await self.llm.chat(msgs, agent_id=agent, role="subagent", prompt_author="orchestrator",
                                    step_id=step.step_id, tools=tools, effort=self.effort)
            msgs.append(r.as_message())
            last = r.content or last
            if not r.tool_calls:
                if nudges >= self.max_nudges and r.finish_reason != "length":
                    return StepResult(step.step_id, last or "(no summary)", finished=False)
                nudges += 1
                msgs.append(MessagePart(role="user", origin="system_template", template_id=WORKER_TEMPLATE,
                                        content=CUT_OFF if r.finish_reason == "length" else NUDGE_TOOL))
                continue
            finish = None
            for c in r.tool_calls:
                if c["parse_error"]:
                    msgs.append(self._tool_reply(c, f"ERROR: {c['parse_error']}"))
                elif c["name"] == "finish_step":
                    finish = c
                    msgs.append(self._tool_reply(c, "Step finished."))
                elif c["name"] == "record_interpretation":
                    a = c["arguments"]
                    evidence = [str(x) for x in a.get("evidence") or []]
                    unknown = [x for x in evidence if not (self.rec.results_dir / f"{x}.json").exists()]
                    if a.get("source") == "data" and not evidence:
                        msgs.append(self._tool_reply(c, "ERROR: source is data, so cite the call_ids whose results show it in evidence."))
                    elif unknown:
                        msgs.append(self._tool_reply(c, f"ERROR: evidence {unknown} are not call_ids of results in this run."))
                    else:
                        try:
                            self.rec.emit(Interpretation(interp_id=new_id("interp"), agent_id=agent, step_id=step.step_id,
                                                         llm_call_id=r.llm_call_id, evidence=evidence,
                                                         **{k: a.get(k) for k in ("kind", "subject", "statement", "source")}))
                            msgs.append(self._tool_reply(c, "Recorded."))
                        except ValidationError:
                            msgs.append(self._tool_reply(c, f"ERROR: kind must be one of {KINDS}; source one of data, "
                                                            "model_knowledge, question; subject and statement are required."))
                else:
                    out = await self.hub.call(c["name"], c["arguments"], agent_id=agent, step_id=step.step_id,
                                              llm_call_id=r.llm_call_id)
                    msgs.append(self._tool_reply(c, out.text))
            if finish:
                a = finish["arguments"]
                return StepResult(step.step_id, str(a.get("summary", "")), [str(x) for x in a.get("result_call_ids", [])])
        return StepResult(step.step_id, f"Ran out of turns. Last message: {last[:1000]}", finished=False)
