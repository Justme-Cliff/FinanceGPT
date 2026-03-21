#include "reasoning.h"
#include "knowledge_base.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <ctype.h>

/* ─────────────────────────────────────────────────────────────────────────
   Keyword tables for question classification
   Each entry is a NULL-terminated list of lowercase substrings.
   The first table whose pattern fires wins (priority order).
   ───────────────────────────────────────────────────────────────────────── */
static const char* CALC_WORDS[] = {
    "calculat","comput","how much","how many","formula",
    "percentage","return on","roi","irr","npv","wacc","eps","ebitda",
    "mortgage payment","monthly payment","interest owe","annual return",
    "compound interest","simple interest","future value","present value",
    "break even","break-even","cost basis","profit margin",
    NULL
};
static const char* RISK_WORDS[] = {
    "risk","hedge","hedging","downside","drawdown",
    "protect","insur","dangerous","safe","lose money",
    "worried about","volatility","worst case","stop loss","tail risk",
    NULL
};
static const char* COMP_WORDS[] = {
    "compar","versus"," vs ","difference between","better than",
    "worse than","prefer","choose between","pros and cons",
    "or rent","or invest","or pay","roth vs","ira vs",
    "stocks vs","bonds vs","etf vs",
    NULL
};
static const char* DEF_WORDS[] = {
    "what is","what are","what's","define","explain","describe",
    "mean by","tell me about","help me understand","i don't understand",
    "can you explain","what does","what do","how does work",
    "how do work","overview of","basics of","introduction to",
    NULL
};
static const char* CAUSAL_WORDS[] = {
    "why","reason","cause","because","result of",
    "impact of","effect of","consequence","lead to",
    "due to","affect","influence","driven by","responsible for",
    NULL
};
static const char* STRAT_WORDS[] = {
    "should i","strategy","approach","best way","recommend",
    "advice","plan","where do i start","how do i start",
    "i want to","what should i","where should i",
    "i'm trying to","how can i","tips for","get started",
    "what steps","action plan",
    NULL
};
static const char* PROC_WORDS[] = {
    "how does","how do","process","mechanism","how to",
    "procedure","steps to","how would","walk me through",
    "step by step","works","operate","function",
    NULL
};
static const char* HIST_WORDS[] = {
    "histor","when did","what happened","crisis","crash",
    "bubble","past","used to","back in","origin","founded",
    "invented","created","first time","during the",
    NULL
};

/* Case-insensitive substring search for any keyword in the list */
static int contains_keyword(const char* text, const char** words) {
    char lower[4096];
    strncpy(lower, text, sizeof(lower) - 1);
    lower[sizeof(lower) - 1] = '\0';
    for (int i = 0; lower[i]; i++) lower[i] = (char)tolower((unsigned char)lower[i]);

    for (int i = 0; words[i]; i++)
        if (strstr(lower, words[i])) return 1;
    return 0;
}

/* ─────────────────────────────────────────────────────────────────────────
   reasoning_classify
   Priority: calculation > risk > comparison > definition > causal >
             strategy > process > historical > general
   ───────────────────────────────────────────────────────────────────────── */
QuestionType reasoning_classify(const char* q) {
    if (contains_keyword(q, CALC_WORDS))  return QTYPE_CALCULATION;
    if (contains_keyword(q, RISK_WORDS))  return QTYPE_RISK;
    if (contains_keyword(q, COMP_WORDS))  return QTYPE_COMPARISON;
    if (contains_keyword(q, DEF_WORDS))   return QTYPE_DEFINITION;
    if (contains_keyword(q, CAUSAL_WORDS))return QTYPE_CAUSAL;
    if (contains_keyword(q, STRAT_WORDS)) return QTYPE_STRATEGY;
    if (contains_keyword(q, PROC_WORDS))  return QTYPE_PROCESS;
    if (contains_keyword(q, HIST_WORDS))  return QTYPE_HISTORICAL;
    return QTYPE_GENERAL;
}

const char* reasoning_qtype_str(QuestionType qt) {
    switch (qt) {
        case QTYPE_CALCULATION: return "calculation";
        case QTYPE_COMPARISON:  return "comparison";
        case QTYPE_DEFINITION:  return "definition";
        case QTYPE_CAUSAL:      return "causal";
        case QTYPE_STRATEGY:    return "strategy";
        case QTYPE_PROCESS:     return "process";
        case QTYPE_HISTORICAL:  return "historical";
        case QTYPE_RISK:        return "risk";
        default:                return "general";
    }
}

/* ─────────────────────────────────────────────────────────────────────────
   Reasoning scaffolds — one per question type
   ───────────────────────────────────────────────────────────────────────── */
const char* reasoning_scaffold(QuestionType qt) {
    switch (qt) {
        case QTYPE_CALCULATION:
            return "This is a calculation question. "
                   "I will identify the inputs, state the relevant formula, "
                   "compute each step clearly, and interpret the numerical result "
                   "in plain language.";

        case QTYPE_COMPARISON:
            return "This is a comparison question. "
                   "I will outline each option's key characteristics, "
                   "highlight the critical differences, "
                   "and conclude with guidance on when each is most appropriate.";

        case QTYPE_DEFINITION:
            return "This is a definition question. "
                   "I will give the core meaning first, "
                   "break it into its components, "
                   "and finish with a concrete real-world example.";

        case QTYPE_CAUSAL:
            return "This is a cause-and-effect question. "
                   "I will trace the mechanism step by step, "
                   "identify the key drivers, "
                   "and state the practical implication.";

        case QTYPE_STRATEGY:
            return "This is a strategy question. "
                   "I will consider goals, constraints, and risk tolerance "
                   "before outlining clear, actionable steps.";

        case QTYPE_PROCESS:
            return "This is a process question. "
                   "I will walk through each phase in order: "
                   "setup, execution, and outcome.";

        case QTYPE_HISTORICAL:
            return "This is a historical question. "
                   "I will outline the key events in sequence, "
                   "identify the root causes, "
                   "and draw the lessons learned.";

        case QTYPE_RISK:
            return "This is a risk question. "
                   "I will identify the specific risks, "
                   "quantify them where possible, "
                   "and suggest concrete mitigation strategies.";

        default:
            return "Let me think through this carefully, "
                   "considering the relevant financial concepts and context.";
    }
}

/* ─────────────────────────────────────────────────────────────────────────
   Query normalization — mirrors _REPHRASE_RULES in reasoning_engine.py
   ───────────────────────────────────────────────────────────────────────── */
void reasoning_normalize(const char* in, char* out, int out_cap) {
    strncpy(out, in, (size_t)out_cap - 1);
    out[out_cap - 1] = '\0';

    static const struct { const char* from; const char* to; } rules[] = {
        {"what's",                "what is"},
        {"whats",                 "what is"},
        {"how's",                 "how does"},
        {"hows",                  "how does"},
        {"i don't understand",    "explain"},
        {"help me understand",    "explain"},
        {"can you explain",       "explain"},
        {"tell me about",         "what is"},
        {"please ",               ""},
        {"kind of ",              ""},
        {"sort of ",              ""},
        {NULL, NULL}
    };

    for (int r = 0; rules[r].from; r++) {
        /* Build lowercase copy to find position case-insensitively */
        char lower[4096];
        strncpy(lower, out, sizeof(lower) - 1);
        lower[sizeof(lower) - 1] = '\0';
        for (int i = 0; lower[i]; i++) lower[i] = (char)tolower((unsigned char)lower[i]);

        char* pos = strstr(lower, rules[r].from);
        if (!pos) continue;

        size_t offset = (size_t)(pos - lower);
        size_t flen   = strlen(rules[r].from);
        size_t tlen   = strlen(rules[r].to);
        size_t cur    = strlen(out);

        char tmp[4096];
        memcpy(tmp, out, offset);
        memcpy(tmp + offset, rules[r].to, tlen);
        memcpy(tmp + offset + tlen, out + offset + flen, cur - offset - flen + 1);
        strncpy(out, tmp, (size_t)out_cap - 1);
        out[out_cap - 1] = '\0';
    }

    /* Trim trailing whitespace */
    int len = (int)strlen(out);
    while (len > 0 && isspace((unsigned char)out[len - 1])) out[--len] = '\0';
}

/* ─────────────────────────────────────────────────────────────────────────
   Decompose compound questions
   Splits on connectors like "and also", "additionally", etc.
   Returns at most 2 sub-questions; falls back to the original.
   ───────────────────────────────────────────────────────────────────────── */
Decomposition* reasoning_decompose(const char* question) {
    Decomposition* d = (Decomposition*)xcalloc(1, sizeof(Decomposition));
    d->questions = (char**)xmalloc(4 * sizeof(char*));

    static const char* connectors[] = {
        " and also ",   " additionally ", " furthermore ",
        " moreover ",   " also, ",        " plus, ",
        NULL
    };

    /* Lowercase copy for pattern matching */
    char lower[4096];
    strncpy(lower, question, sizeof(lower) - 1);
    lower[sizeof(lower) - 1] = '\0';
    for (int i = 0; lower[i]; i++) lower[i] = (char)tolower((unsigned char)lower[i]);

    int    split_offset = -1;
    size_t split_len    = 0;

    for (int ci = 0; connectors[ci]; ci++) {
        char* pos = strstr(lower, connectors[ci]);
        if (pos) {
            split_offset = (int)(pos - lower);
            split_len    = strlen(connectors[ci]);
            break;
        }
    }

    if (split_offset > 8) {
        /* First sub-question */
        char* q1 = (char*)xmalloc((size_t)split_offset + 1);
        memcpy(q1, question, (size_t)split_offset);
        q1[split_offset] = '\0';

        /* Second sub-question */
        const char* rest = question + split_offset + split_len;
        char*        q2  = xstrdup(rest);

        /* Trim trailing/leading whitespace */
        int l1 = (int)strlen(q1);
        while (l1 > 0 && isspace((unsigned char)q1[l1-1])) q1[--l1] = '\0';
        char* q2p = q2;
        while (*q2p && isspace((unsigned char)*q2p)) q2p++;

        if (l1 > 8)          d->questions[d->n_questions++] = q1;   else free(q1);
        if (strlen(q2p) > 8) d->questions[d->n_questions++] = xstrdup(q2p);
        free(q2);
    }

    /* Fallback: keep original */
    if (d->n_questions == 0)
        d->questions[d->n_questions++] = xstrdup(question);

    return d;
}

void reasoning_decompose_free(Decomposition* d) {
    if (!d) return;
    for (int i = 0; i < d->n_questions; i++) free(d->questions[i]);
    free(d->questions);
    free(d);
}

/* ─────────────────────────────────────────────────────────────────────────
   Context builder
   Assembles: recent conversation history + top-3 KB docs + scaffold
   ───────────────────────────────────────────────────────────────────────── */

/* Macro: append string s to growable buffer (ctx, len, cap) */
#define CTX_APPEND(s) do {                                              \
    size_t _sl = strlen(s);                                             \
    while (len + _sl + 4 > cap) { cap *= 2; ctx = (char*)xrealloc(ctx, cap); } \
    memcpy(ctx + len, (s), _sl);                                        \
    len += _sl;                                                         \
    ctx[len] = '\0';                                                    \
} while (0)

char* reasoning_build_context(const char* question,
                               const void* results_void, int n_results,
                               const HistoryTurn* history, int n_history) {
    const KbResult* results = (const KbResult*)results_void;
    QuestionType    qt      = reasoning_classify(question);
    const char*     scaffold = reasoning_scaffold(qt);

    size_t cap = 8192;
    char*  ctx = (char*)xmalloc(cap);
    size_t len = 0;
    ctx[0] = '\0';

    /* ── Recent conversation (last 3 turns) ─────────────────────────── */
    if (n_history > 0) {
        int hist_start = n_history > 3 ? n_history - 3 : 0;
        CTX_APPEND("=== Recent conversation ===\n");
        for (int i = hist_start; i < n_history; i++) {
            CTX_APPEND("Q: ");
            CTX_APPEND(history[i].question);
            CTX_APPEND("\n");

            /* Show at most 200 chars of each previous answer */
            char preview[204];
            strncpy(preview, history[i].answer, 200);
            preview[200] = '\0';
            if (strlen(history[i].answer) > 200) {
                preview[200] = '.'; preview[201] = '.'; preview[202] = '.'; preview[203] = '\0';
            }
            CTX_APPEND("A: ");
            CTX_APPEND(preview);
            CTX_APPEND("\n");
        }
        CTX_APPEND("\n");
    }

    /* ── Retrieved knowledge (top 3) ───────────────────────────────── */
    int n_show = n_results > 3 ? 3 : n_results;
    if (n_show > 0 && results) {
        CTX_APPEND("=== Relevant knowledge ===\n");
        for (int i = 0; i < n_show; i++) {
            /* Skip very low-quality results */
            if (results[i].score < KB_MIN_SCORE) continue;

            /* Format: "[1] [Source_name] <answer text>" */
            char source_title[64];
            strncpy(source_title, results[i].source, 63);
            source_title[63] = '\0';
            /* Replace underscores with spaces and title-case first char */
            for (int j = 0; source_title[j]; j++)
                if (source_title[j] == '_') source_title[j] = ' ';
            if (source_title[0])
                source_title[0] = (char)toupper((unsigned char)source_title[0]);

            char prefix[80];
            snprintf(prefix, sizeof(prefix), "[%d] [%s] ", i + 1, source_title);
            CTX_APPEND(prefix);

            /* Truncate very long answers to keep context window sane */
            char ans[512];
            strncpy(ans, results[i].answer, 508);
            ans[508] = '\0';
            if (strlen(results[i].answer) > 508) {
                strcat(ans, "...");
            }
            CTX_APPEND(ans);
            CTX_APPEND("\n");
        }
        CTX_APPEND("\n");
    }

    /* ── Reasoning scaffold ─────────────────────────────────────────── */
    CTX_APPEND("=== Reasoning approach ===\n");
    CTX_APPEND(scaffold);
    CTX_APPEND("\n\n");

    return ctx;
}

#undef CTX_APPEND

/* ─────────────────────────────────────────────────────────────────────────
   Prompt builder
   Final prompt = context + optional calc results + "Q: ... <SEP> A:"
   ───────────────────────────────────────────────────────────────────────── */
#define PROMPT_APPEND(s) do {                                               \
    size_t _sl = strlen(s);                                                 \
    while (plen + _sl + 4 > pcap) { pcap *= 2; prompt = (char*)xrealloc(prompt, pcap); } \
    memcpy(prompt + plen, (s), _sl);                                        \
    plen += _sl;                                                            \
    prompt[plen] = '\0';                                                    \
} while (0)

char* reasoning_build_prompt(const char* question, const char* context,
                              const char** calc_results, int n_calcs) {
    size_t pcap   = 8192;
    char*  prompt = (char*)xmalloc(pcap);
    size_t plen   = 0;
    prompt[0]     = '\0';

    if (context && context[0]) {
        PROMPT_APPEND(context);
    }

    if (n_calcs > 0 && calc_results) {
        PROMPT_APPEND("=== Pre-computed results ===\n");
        for (int i = 0; i < n_calcs; i++) {
            if (calc_results[i]) {
                PROMPT_APPEND(calc_results[i]);
                PROMPT_APPEND("\n");
            }
        }
        PROMPT_APPEND("\n");
    }

    PROMPT_APPEND("Q: ");
    PROMPT_APPEND(question);
    PROMPT_APPEND(" <SEP> A:");

    return prompt;
}

#undef PROMPT_APPEND
