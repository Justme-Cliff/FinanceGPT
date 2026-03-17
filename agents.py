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

from knowledge_base import KnowledgeBase, _REPHRASE_RULES
from reasoning_engine import ReasoningEngine


# ── Agent 1: Knowledge Retrieval ───────────────────────────────────────

class KnowledgeAgent:
    NAME = "KnowledgeAgent"

    def __init__(self, kb: KnowledgeBase):
        self.kb = kb

    def run(self, query: str) -> dict:
        t0 = time.perf_counter()
        try:
            # Search with both the original and normalized query; merge results
            normalized = query
            for pattern, replacement in _REPHRASE_RULES:
                normalized = re.sub(pattern, replacement, normalized, flags=re.IGNORECASE)
            normalized = re.sub(r" {2,}", " ", normalized).strip()

            results = self.kb.search(query, top_k=6)
            if normalized != query:
                extra = self.kb.search(normalized, top_k=6)
                # Merge: add any extra results not already present (by question text)
                seen = {r["question"] for r in results}
                for r in extra:
                    if r["question"] not in seen:
                        results.append(r)
                        seen.add(r["question"])
                # Re-sort by score descending and keep top 6
                results.sort(key=lambda x: -x["score"])
                results = results[:6]

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
        (
            "Monthly Mortgage Payment",
            r"mortgage.{0,40}?\$?(\d[\d,]*\.?\d*).{0,20}?(\d+\.?\d*)\s*%?.{0,20}?(\d+)\s*year",
            lambda vs: (
                lambda p, annual_r, years: (
                    f"Monthly payment = ${p * (r := annual_r/100/12) * (1+r)**(n := years*12) / ((1+r)**n - 1):,.2f}"
                    if annual_r > 0 else f"Monthly payment = ${p / (years * 12):,.2f}"
                )
            )(float(vs[0].replace(",", "")), float(vs[1]), int(vs[2])),
        ),
        (
            "Simple Interest",
            r"simple interest.{0,30}?\$?(\d[\d,]*\.?\d*).{0,20}?(\d+\.?\d*)\s*%?.{0,20}?(\d+)\s*year",
            lambda vs: (
                lambda p, r, n: f"Simple Interest = ${p:,.2f} × {r}% × {n} years = ${p * r/100 * n:,.2f}; Total = ${p + p * r/100 * n:,.2f}"
            )(float(vs[0].replace(",", "")), float(vs[1]), int(vs[2])),
        ),
        (
            "Savings Goal (Future Value)",
            r"(?:save|savings?|invest).{0,30}?\$?(\d[\d,]*\.?\d*)\s*(?:per month|monthly|a month).{0,30}?(\d+\.?\d*)\s*%?.{0,20}?(\d+)\s*year",
            lambda vs: (
                lambda pmt, annual_r, years: (
                    f"Future value of ${pmt:,.2f}/month at {annual_r}% for {years} years = "
                    f"${pmt * ((1 + (r := annual_r/100/12))**(n := years*12) - 1) / r:,.2f}"
                    if annual_r > 0 else f"Future value = ${pmt * years * 12:,.2f}"
                )
            )(float(vs[0].replace(",", "")), float(vs[1]), int(vs[2])),
        ),
        (
            "Debt Payoff Time",
            r"(?:pay off|payoff).{0,30}?\$?(\d[\d,]*\.?\d*).{0,20}?(\d+\.?\d*)\s*%?.{0,20}?\$?(\d[\d,]*\.?\d*)\s*(?:per month|monthly|a month)",
            lambda vs: (
                lambda balance, annual_r, monthly_pmt: (
                    f"Months to pay off ${balance:,.2f} at {annual_r}% with ${monthly_pmt:,.2f}/month = "
                    f"{-math.log(1 - balance * (r := annual_r/100/12) / monthly_pmt) / math.log(1 + r):.0f} months"
                    if annual_r > 0 and monthly_pmt > balance * annual_r/100/12 else
                    f"Months to pay off ${balance:,.2f} with ${monthly_pmt:,.2f}/month = {balance/monthly_pmt:.0f} months (0% interest)"
                )
            )(float(vs[0].replace(",", "")), float(vs[1]), float(vs[2].replace(",", ""))),
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

        # ── Phase 1b: Web search fallback (when KB confidence is low) ─
        web_result = {"agent": "WebSearchAgent", "results": [], "used": False, "success": True, "off_topic": False}
        # Exclude conversation.csv — it inflates scores for unrelated queries
        finance_kb_results = [
            r for r in kb_result.get("results", [])
            if r.get("source", "") != "conversation"
        ]
        try:
            from web_search import needs_web_search, web_search, format_web_results, _is_off_topic
            if _is_off_topic(query):
                web_result["off_topic"] = True
            elif needs_web_search(finance_kb_results, query):
                raw = web_search(query, max_results=4)
                web_result["results"] = raw
                web_result["used"]    = bool(raw)
        except Exception:
            pass

        # ── Phase 2: Reasoning (depends on knowledge + web) ─────────
        r_agent = ReasoningAgent(self.r_engine)
        # Merge KB results with web snippets for context building
        kb_results_for_reasoning = kb_result.get("results", [])
        if web_result["used"]:
            from web_search import format_web_results
            web_context = format_web_results(web_result["results"])
        else:
            web_context = ""

        reasoning_result = r_agent.run(
            query,
            kb_results_for_reasoning,
            history,
        )

        # Append web context to reasoning context if available
        if web_context:
            existing = reasoning_result.get("context", "")
            reasoning_result["context"] = (
                web_context + "\n\n" + existing if existing else web_context
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
            query, model_result, kb_result, calc_result, reasoning_result, web_result
        )

        agent_info = {
            "knowledge":   kb_result,
            "reasoning":   reasoning_result,
            "calculation": calc_result,
            "model":       model_result,
            "web":         web_result,
        }
        return response, agent_info

    def _synthesize(
        self,
        query: str,
        model_result: dict,
        kb_result: dict,
        calc_result: dict,
        reasoning_result: dict,
        web_result: dict | None = None,
    ) -> str:
        parts: list[str] = []

        # Prepend any computed calculations
        calcs = calc_result.get("calculations")
        if calcs:
            parts.append("**Calculated:**")
            for c in calcs:
                parts.append(f"  {c}")
            parts.append("")

        model_text = model_result.get("response", "").strip()
        kb_results = kb_result.get("results", [])
        # Exclude conversation.csv from best-match scoring — it pollutes relevance for real queries
        finance_kb_results = [r for r in kb_results if r.get("source", "") != "conversation"]
        best_kb = finance_kb_results[0] if finance_kb_results else (kb_results[0] if kb_results else None)
        best_kb_score = best_kb["score"] if best_kb else 0.0
        web_used      = web_result and web_result.get("used") and web_result.get("results")
        off_topic     = web_result and web_result.get("off_topic", False)

        # Off-topic query — politely redirect
        if off_topic:
            parts.append(
                "I'm a finance AI — that topic is outside my scope. "
                "Ask me about stocks, investing, crypto, budgeting, markets, or anything money-related!"
            )
            return "\n".join(parts)

        # If web search fired, prioritize web results over static KB answers —
        # the user asked for live/current info and the KB answer is likely stale/generic.
        if web_used:
            web_results = web_result["results"]
            parts.append("Here's what I found online:\n")
            for r in web_results[:3]:
                title   = r.get("title", "").strip()
                snippet = r.get("snippet", "").strip()
                url     = r.get("url", "").strip()
                if snippet:
                    parts.append(f"• {title}")
                    parts.append(f"  {snippet}")
                    if url:
                        parts.append(f"  {url}")
                    parts.append("")
            parts.append("[Source: Web Search]")
            return "\n".join(parts)

        # Use KB answer directly when:
        #   1. Model output is absent or too short
        #   2. KB has a high-confidence match (score > 0.20) — trained answer is authoritative
        use_kb_direct = (not model_text or len(model_text) < 30) or best_kb_score >= 0.20

        if model_text and len(model_text) >= 30 and not use_kb_direct:
            parts.append(model_text)
        elif best_kb:
            parts.append(best_kb["answer"])
            src = best_kb["source"].replace("_", " ").title()
            parts.append(f"\n[Source: {src}]")
        elif model_text and len(model_text) >= 30:
            parts.append(model_text)
        else:
            parts.append(
                "I don't have specific data on that in my training data. "
                "Try running /fetch to pull live market data, or add a CSV for this topic and retrain."
            )

        return "\n".join(parts)
