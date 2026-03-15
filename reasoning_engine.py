"""
Reasoning Engine — Chain-of-thought scaffolding and question decomposition.
No external APIs. Pure logic that augments the model's context window with
structured reasoning before generation.
"""

import re


# ── Question classification ────────────────────────────────────────────

_PATTERNS: list[tuple[str, str]] = [
    # Calculation — only when explicit compute/calculate verb is present
    ("calculation",  r"\b(calculat|comput|how much\b|how many\b|formula\b|percentage\b|return on\b|roi\b|irr\b|npv\b|wacc\b|eps\b|ebitda\b)"),
    # Risk checked before process so "how do I hedge" → risk
    ("risk",         r"\b(risk\b|hedge\b|hedging\b|downside\b|drawdown\b|protect\b|insur\b|dangerous\b)"),
    ("comparison",   r"\b(compar|versus\b|vs\.?\b|difference between|better than|worse than|prefer\b|choose between|pros and cons)"),
    ("definition",   r"\b(what is\b|what are\b|define\b|explain\b|describe\b|mean by|tell me about)"),
    ("causal",       r"\b(why\b|reason\b|cause\b|because\b|result of|impact of|effect of|consequence\b|lead to|due to)"),
    ("strategy",     r"\b(should i\b|strategy\b|approach\b|best way|recommend\b|advice\b|plan\b)"),
    ("process",      r"\b(how does\b|how do\b|process\b|mechanism\b|how to\b|procedure\b|steps to)"),
    ("historical",   r"\b(histor|when did|what happened|crisis\b|crash\b|bubble\b|past\b)"),
]

_REASONING_SCAFFOLDS: dict[str, str] = {
    "calculation": (
        "This is a calculation question. "
        "I will identify the inputs, state the formula, compute step-by-step, and interpret the result."
    ),
    "comparison": (
        "This is a comparison question. "
        "I will outline each option's key characteristics, then highlight the critical differences, "
        "and conclude with when each is most appropriate."
    ),
    "definition": (
        "This is a definition question. "
        "I will give the core meaning first, then break it into components, and finally show a real-world example."
    ),
    "causal": (
        "This is a cause-and-effect question. "
        "I will trace the mechanism step by step, identify the key drivers, and state the implication."
    ),
    "strategy": (
        "This is a strategy question. "
        "I will consider goals, constraints, and risk tolerance before outlining actionable steps."
    ),
    "process": (
        "This is a process question. "
        "I will walk through each phase in order: setup, execution, and outcome."
    ),
    "historical": (
        "This is a historical question. "
        "I will outline the key events in sequence, identify root causes, and draw lessons learned."
    ),
    "risk": (
        "This is a risk question. "
        "I will identify the specific risks, quantify where possible, and suggest mitigation strategies."
    ),
    "general": (
        "Let me think through this carefully, considering the relevant financial concepts and context."
    ),
}


class ReasoningEngine:
    """
    Decomposes questions, classifies intent, and builds a structured
    chain-of-thought context that primes the model for step-by-step reasoning.
    """

    def classify(self, question: str) -> str:
        q_lower = question.lower()
        for qtype, pattern in _PATTERNS:
            if re.search(pattern, q_lower):
                return qtype
        return "general"

    def decompose(self, question: str) -> list[str]:
        """
        Split compound questions into sub-questions so each can be
        answered independently, then synthesized.
        """
        # Connectors that often join two separate questions
        split_re = re.compile(
            r"\s+(?:and also|additionally|furthermore|moreover|also,?|plus,?)\s+",
            re.IGNORECASE,
        )
        parts = split_re.split(question)
        # Keep only parts that look like real questions/phrases (>8 chars)
        cleaned = [p.strip() for p in parts if len(p.strip()) > 8]
        return cleaned if len(cleaned) > 1 else [question]

    def build_context(
        self,
        question: str,
        retrieved: list[dict],
        history: list[dict] | None = None,
    ) -> tuple[str, str]:
        """
        Assemble the full reasoning context string and return
        (context_text, question_type).
        """
        qtype = self.classify(question)
        scaffold = _REASONING_SCAFFOLDS[qtype]

        lines: list[str] = []

        # 1. Conversation history (most recent 3 turns)
        if history:
            recent = history[-3:]
            if recent:
                lines.append("=== Recent conversation ===")
                for turn in recent:
                    lines.append(f"Q: {turn['question']}")
                    # Truncate long answers so context stays compact
                    ans_preview = turn["answer"][:200].rstrip()
                    if len(turn["answer"]) > 200:
                        ans_preview += "..."
                    lines.append(f"A: {ans_preview}")
                lines.append("")

        # 2. Retrieved knowledge
        if retrieved:
            lines.append("=== Relevant knowledge ===")
            for i, doc in enumerate(retrieved[:3], 1):
                src = doc["source"].replace("_", " ").title()
                # Include the full answer (the model uses it for generation)
                lines.append(f"[{i}] [{src}] {doc['answer'][:400]}")
            lines.append("")

        # 3. Reasoning scaffold
        lines.append(f"=== Reasoning approach ===")
        lines.append(scaffold)
        lines.append("")

        return "\n".join(lines), qtype

    def build_prompt(
        self,
        question: str,
        context: str,
        calc_results: list[str] | None = None,
    ) -> str:
        """
        Final prompt string passed to the model.
        Format mirrors training data so the model knows what to do.
        """
        parts: list[str] = []

        if context.strip():
            parts.append(context.strip())
            parts.append("")

        if calc_results:
            parts.append("=== Pre-computed results ===")
            for r in calc_results:
                parts.append(r)
            parts.append("")

        parts.append(f"Q: {question} <SEP> A:")
        return "\n".join(parts)
