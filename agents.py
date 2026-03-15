"""
Multi-Agent System — 4 specialized agents running in parallel via ThreadPoolExecutor.

Agent pipeline:
  Phase 1 (parallel): KnowledgeAgent  +  CalculationAgent
  Phase 2 (parallel): ReasoningAgent  (uses Phase 1 output)
  Phase 3:            ModelAgent      (uses Phase 2 output)
  Final:              Orchestrator synthesizes all outputs
"""

import re
import math
import time
from concurrent.futures import ThreadPoolExecutor

from knowledge_base import KnowledgeBase
from reasoning_engine import ReasoningEngine


# ── Agent 1: Knowledge Retrieval ───────────────────────────────────────

class KnowledgeAgent:
    NAME = "KnowledgeAgent"

    def __init__(self, kb: KnowledgeBase):
        self.kb = kb

    def run(self, query: str) -> dict:
        t0 = time.perf_counter()
        try:
            results = self.kb.search(query, top_k=6)
            return {
                "agent": self.NAME,
                "results": results,
                "success": True,
                "elapsed": round(time.perf_counter() - t0, 3),
            }
        except Exception as exc:
            return {
                "agent": self.NAME,
                "results": [],
                "success": False,
                "error": str(exc),
                "elapsed": round(time.perf_counter() - t0, 3),
            }


# ── Agent 2: Reasoning & Decomposition ────────────────────────────────

class ReasoningAgent:
    NAME = "ReasoningAgent"

    def __init__(self, engine: ReasoningEngine):
        self.engine = engine

    def run(
        self,
        query: str,
        retrieved: list[dict],
        history: list[dict] | None = None,
    ) -> dict:
        t0 = time.perf_counter()
        try:
            sub_questions = self.engine.decompose(query)
            context, qtype = self.engine.build_context(query, retrieved, history)
            return {
                "agent": self.NAME,
                "sub_questions": sub_questions,
                "question_type": qtype,
                "context": context,
                "success": True,
                "elapsed": round(time.perf_counter() - t0, 3),
            }
        except Exception as exc:
            return {
                "agent": self.NAME,
                "sub_questions": [query],
                "question_type": "general",
                "context": "",
                "success": False,
                "error": str(exc),
                "elapsed": round(time.perf_counter() - t0, 3),
            }


# ── Agent 3: Financial Calculation ────────────────────────────────────

class CalculationAgent:
    NAME = "CalculationAgent"

    # (name, trigger_pattern, compute_fn)
    # compute_fn receives the list of matched float strings
    _FORMULAS = [
        (
            "Sharpe Ratio",
            r"sharpe.{0,50}?(\d+\.?\d*)\s*%?.{0,30}?(\d+\.?\d*)\s*%?.{0,30}?(\d+\.?\d*)\s*%?",
            lambda vs: f"Sharpe = ({vs[0]}% - {vs[1]}%) / {vs[2]}% = {(float(vs[0]) - float(vs[1])) / float(vs[2]):.4f}"
            if float(vs[2]) != 0 else None,
        ),
        (
            "P/E Ratio",
            r"p/?e.{0,20}?\$?(\d+\.?\d*).{0,20}?(?:eps|earnings).{0,10}?\$?(\d+\.?\d*)",
            lambda vs: f"P/E = ${vs[0]} / ${vs[1]} = {float(vs[0]) / float(vs[1]):.2f}x"
            if float(vs[1]) != 0 else None,
        ),
        (
            "ROI",
            r"roi.{0,20}?(?:gain|return|made).{0,10}?\$?(\d[\d,]*\.?\d*).{0,20}?(?:cost|invest|spent).{0,10}?\$?(\d[\d,]*\.?\d*)",
            lambda vs: f"ROI = (${vs[0]} - ${vs[1]}) / ${vs[1]} * 100 = {(float(vs[0].replace(',','')) - float(vs[1].replace(',',''))) / float(vs[1].replace(',','')) * 100:.2f}%"
            if float(vs[1].replace(",", "")) != 0 else None,
        ),
        (
            "Compound Interest",
            r"compound.{0,30}?\$?(\d[\d,]*\.?\d*).{0,20}?(\d+\.?\d*)\s*%?.{0,20}?(\d+)\s*year",
            lambda vs: (
                lambda p, r, n: f"A = ${p:,.2f} * (1 + {r}%)^{n} = ${p * (1 + r/100)**n:,.2f}"
            )(float(vs[0].replace(",", "")), float(vs[1]), int(vs[2])),
        ),
        (
            "Present Value",
            r"present value.{0,20}?\$?(\d[\d,]*\.?\d*).{0,20}?(\d+\.?\d*)\s*%?.{0,20}?(\d+)\s*year",
            lambda vs: f"PV = ${float(vs[0].replace(',','')):,.2f} / (1 + {vs[1]}%)^{vs[2]} = ${float(vs[0].replace(',','')) / (1 + float(vs[1])/100)**int(vs[2]):,.2f}",
        ),
        (
            "EV/EBITDA",
            r"ev.{0,20}?ebitda.{0,20}?(?:market cap|ev).{0,10}?\$?(\d[\d,]*\.?\d*)m?.{0,20}?debt.{0,10}?\$?(\d[\d,]*\.?\d*)m?.{0,20}?cash.{0,10}?\$?(\d[\d,]*\.?\d*)m?.{0,20}?ebitda.{0,10}?\$?(\d[\d,]*\.?\d*)m?",
            lambda vs: (
                lambda mc, d, c, e: f"EV = ${mc}M + ${d}M - ${c}M = ${mc+d-c:.0f}M; EV/EBITDA = {(mc+d-c)/e:.2f}x"
            )(*[float(v.replace(",", "")) for v in vs]),
        ),
    ]

    def run(self, query: str) -> dict:
        t0 = time.perf_counter()
        results = []
        q_lower = query.lower()

        # Only attempt calculation if the query contains numbers
        numbers = re.findall(r"\d+\.?\d*", query)
        if len(numbers) >= 2:
            for name, pattern, compute in self._FORMULAS:
                m = re.search(pattern, q_lower)
                if m:
                    try:
                        result = compute(list(m.groups()))
                        if result:
                            results.append(f"{name}: {result}")
                    except Exception:
                        pass

        return {
            "agent": self.NAME,
            "calculations": results if results else None,
            "success": True,
            "elapsed": round(time.perf_counter() - t0, 3),
        }


# ── Agent 4: Model Generation ─────────────────────────────────────────

class ModelAgent:
    NAME = "ModelAgent"

    def __init__(self, model, tokenizer, gen_config: dict, device):
        self.model = model
        self.tok = tokenizer
        self.cfg = gen_config
        self.device = device

    def run(self, prompt: str) -> dict:
        t0 = time.perf_counter()
        try:
            from tokenizer import SPECIAL
            eos_id = SPECIAL["<EOS>"]
            sep_id = SPECIAL["<SEP>"]
            bos_id = SPECIAL["<BOS>"]

            ids = [bos_id] + self.tok.encode(prompt, add_special=False)
            # Strip trailing EOS if tokenizer added one
            if ids and ids[-1] == eos_id:
                ids = ids[:-1]

            # Truncate to fit context minus generation budget
            max_ctx = self.model.config["max_seq_len"] - self.cfg["max_new_tokens"]
            if len(ids) > max_ctx:
                ids = ids[:1] + ids[-(max_ctx - 1):]   # keep BOS, trim middle

            idx = torch.tensor([ids], dtype=torch.long, device=self.device)

            with torch.no_grad():
                out = self.model.generate(
                    idx,
                    max_new_tokens=self.cfg["max_new_tokens"],
                    temperature=self.cfg["temperature"],
                    top_k=self.cfg["top_k"],
                    top_p=self.cfg["top_p"],
                    repetition_penalty=self.cfg["repetition_penalty"],
                    stop_ids=[eos_id],
                )

            new_ids = out[0, len(ids):].tolist()

            # Trim at EOS or SEP
            clean = []
            for tid in new_ids:
                if tid in (eos_id, sep_id):
                    break
                clean.append(tid)

            raw = self.tok.decode(clean, skip_special=True).strip()
            response = self._clean(raw)

            return {
                "agent": self.NAME,
                "response": response,
                "success": bool(response),
                "elapsed": round(time.perf_counter() - t0, 3),
            }
        except Exception as exc:
            return {
                "agent": self.NAME,
                "response": "",
                "success": False,
                "error": str(exc),
                "elapsed": round(time.perf_counter() - t0, 3),
            }

    def _clean(self, text: str) -> str:
        """Remove bleed-over (next question starting) and ensure proper ending."""
        # Cut at next question marker
        for stopper in ["Q:", "Question:", "\n\n\n", "==="]:
            idx = text.find(stopper)
            if idx > 30:
                text = text[:idx]

        text = text.strip()

        # Ensure sentence ends properly
        if text and text[-1] not in ".!?":
            last = max(text.rfind("."), text.rfind("!"), text.rfind("?"))
            if last > len(text) * 0.4:
                text = text[: last + 1]

        return text.strip()


# ── Orchestrator ───────────────────────────────────────────────────────

class AgentOrchestrator:
    """
    Runs all 4 agents and synthesizes a final response.

    Timing:
      t=0   → KnowledgeAgent and CalculationAgent start (parallel)
      t=1   → ReasoningAgent starts with knowledge results
      t=2   → ModelAgent starts with full reasoning context
      t=3   → Synthesis combines everything
    """

    def __init__(
        self,
        kb: KnowledgeBase,
        engine: ReasoningEngine,
        model,
        tokenizer,
        gen_config: dict,
        device,
    ):
        self.kb_agent   = KnowledgeAgent(kb)
        self.r_engine   = engine
        self.calc_agent = CalculationAgent()
        self.model_agent = ModelAgent(model, tokenizer, gen_config, device)

    def process(
        self,
        query: str,
        history: list[dict] | None = None,
    ) -> tuple[str, dict]:
        """
        Returns (final_response_text, agent_info_dict).
        """
        # ── Phase 1: Knowledge + Calculation in parallel ────────────
        with ThreadPoolExecutor(max_workers=2) as ex:
            kb_future   = ex.submit(self.kb_agent.run, query)
            calc_future = ex.submit(self.calc_agent.run, query)
            kb_result   = kb_future.result()
            calc_result = calc_future.result()

        # ── Phase 2: Reasoning (depends on knowledge) ───────────────
        r_agent = ReasoningAgent(self.r_engine)
        reasoning_result = r_agent.run(
            query,
            kb_result.get("results", []),
            history,
        )

        # ── Phase 3: Model generation ────────────────────────────────
        prompt = self.r_engine.build_prompt(
            query,
            reasoning_result.get("context", ""),
            calc_result.get("calculations"),
        )
        model_result = self.model_agent.run(prompt)

        # ── Synthesis ────────────────────────────────────────────────
        response = self._synthesize(
            query, model_result, kb_result, calc_result, reasoning_result
        )

        agent_info = {
            "knowledge":   kb_result,
            "reasoning":   reasoning_result,
            "calculation": calc_result,
            "model":       model_result,
        }
        return response, agent_info

    def _synthesize(
        self,
        query: str,
        model_result: dict,
        kb_result: dict,
        calc_result: dict,
        reasoning_result: dict,
    ) -> str:
        parts: list[str] = []

        # Prepend any computed calculations
        calcs = calc_result.get("calculations")
        if calcs:
            parts.append("**Calculated:**")
            for c in calcs:
                parts.append(f"  {c}")
            parts.append("")

        # Primary: model response
        model_text = model_result.get("response", "").strip()
        if model_text and len(model_text) > 20:
            parts.append(model_text)
        else:
            # Fallback: best knowledge-base match
            results = kb_result.get("results", [])
            if results:
                best = results[0]
                parts.append(best["answer"])
                src = best["source"].replace("_", " ").title()
                parts.append(f"\n[Source: {src}]")
            else:
                parts.append(
                    "I don't have specific data on that yet. "
                    "Try training on more CSV files or rephrase your question."
                )

        return "\n".join(parts)
