#include "agents.h"
#include "math_ops.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>
#include <ctype.h>

/* ── Calculation agent ────────────────────────────────────────────── */
static int find_numbers(const char* text, double* nums, int max_nums) {
    int n = 0;
    const char* p = text;
    while (*p && n < max_nums) {
        while (*p && !isdigit((unsigned char)*p) && *p != '-') p++;
        if (!*p) break;
        char* end;
        double v = strtod(p, &end);
        if (end > p && (isdigit((unsigned char)*p) || (*p == '-' && isdigit((unsigned char)p[1])))) {
            nums[n++] = v;
            p = end;
        } else {
            p++;
        }
    }
    return n;
}

static int ci_contains(const char* text, const char* needle) {
    char lower[4096];
    strncpy(lower, text, sizeof(lower)-1);
    lower[sizeof(lower)-1] = '\0';
    for (int i = 0; lower[i]; i++) lower[i] = (char)tolower((unsigned char)lower[i]);
    return strstr(lower, needle) != NULL;
}

CalcResult calc_agent_run(const char* query) {
    CalcResult res = {NULL, 0};
    double nums[16];
    int n = find_numbers(query, nums, 16);
    if (n < 2) return res;

    int cap = 10;
    res.results = (char**)xmalloc(cap * sizeof(char*));
    char buf[512];

    /* Sharpe Ratio */
    if (ci_contains(query, "sharpe") && n >= 3) {
        double ret = nums[0], rf = nums[1], sd = nums[2];
        if (sd != 0.0) {
            snprintf(buf, sizeof(buf),
                     "Sharpe Ratio: (%.2f%% - %.2f%%) / %.2f%% = %.4f",
                     ret, rf, sd, (ret - rf) / sd);
            res.results[res.n_results++] = xstrdup(buf);
        }
    }

    /* Compound Interest */
    if ((ci_contains(query, "compound") || ci_contains(query, "compounded")) && n >= 3) {
        double P = nums[0], r = nums[1], t = nums[2];
        double A = P * pow(1.0 + r / 100.0, t);
        snprintf(buf, sizeof(buf),
                 "Compound Interest: $%.2f * (1 + %.2f%%)^%.0f = $%.2f",
                 P, r, t, A);
        res.results[res.n_results++] = xstrdup(buf);
    }

    /* ROI */
    if ((ci_contains(query, "roi") || ci_contains(query, "return on investment")) && n >= 2) {
        double gain = nums[0], cost = nums[1];
        if (cost != 0.0) {
            snprintf(buf, sizeof(buf),
                     "ROI: ($%.2f - $%.2f) / $%.2f * 100 = %.2f%%",
                     gain, cost, cost, (gain - cost) / cost * 100.0);
            res.results[res.n_results++] = xstrdup(buf);
        }
    }

    /* Present Value */
    if ((ci_contains(query, "present value") || ci_contains(query, " pv ")) && n >= 3) {
        double FV = nums[0], r = nums[1], t = nums[2];
        double PV = FV / pow(1.0 + r / 100.0, t);
        snprintf(buf, sizeof(buf),
                 "Present Value: $%.2f / (1 + %.2f%%)^%.0f = $%.2f",
                 FV, r, t, PV);
        res.results[res.n_results++] = xstrdup(buf);
    }

    /* EV/EBITDA */
    if (ci_contains(query, "ev/ebitda") || ci_contains(query, "enterprise value")) {
        if (n >= 2) {
            double ev = nums[0], ebitda = nums[1];
            if (ebitda != 0.0) {
                snprintf(buf, sizeof(buf),
                         "EV/EBITDA: $%.2f / $%.2f = %.2fx",
                         ev, ebitda, ev / ebitda);
                res.results[res.n_results++] = xstrdup(buf);
            }
        }
    }

    /* Mortgage */
    if (ci_contains(query, "mortgage") && n >= 3) {
        double P = nums[0], annual_r = nums[1];
        int years = (int)nums[2];
        int N = years * 12;
        double r_m = annual_r / 100.0 / 12.0;
        double payment;
        if (r_m > 0.0)
            payment = P * r_m * pow(1.0 + r_m, N) / (pow(1.0 + r_m, N) - 1.0);
        else
            payment = N > 0 ? P / N : 0.0;
        snprintf(buf, sizeof(buf),
                 "Monthly Mortgage Payment: $%.2f at %.2f%% for %d years = $%.2f/month",
                 P, annual_r, years, payment);
        res.results[res.n_results++] = xstrdup(buf);
    }

    /* Simple Interest */
    if (ci_contains(query, "simple interest") && n >= 3) {
        double P = nums[0], r = nums[1], t = nums[2];
        double I = P * r / 100.0 * t;
        snprintf(buf, sizeof(buf),
                 "Simple Interest: $%.2f * %.2f%% * %.0f = $%.2f; Total = $%.2f",
                 P, r, t, I, P + I);
        res.results[res.n_results++] = xstrdup(buf);
    }

    /* P/E Ratio */
    if ((ci_contains(query, "p/e") || ci_contains(query, "price to earnings")) && n >= 2) {
        double price = nums[0], eps_val = nums[1];
        if (eps_val != 0.0) {
            snprintf(buf, sizeof(buf),
                     "P/E Ratio: $%.2f / $%.2f = %.2fx",
                     price, eps_val, price / eps_val);
            res.results[res.n_results++] = xstrdup(buf);
        }
    }

    /* Savings Goal (future value of annuity) */
    if ((ci_contains(query, "save") || ci_contains(query, "saving")) &&
        (ci_contains(query, "per month") || ci_contains(query, "monthly")) && n >= 3) {
        double pmt = nums[0], annual_r = nums[1];
        int years = (int)nums[2];
        int N = years * 12;
        double r_m = annual_r / 100.0 / 12.0;
        double FV;
        if (r_m > 0.0) FV = pmt * (pow(1.0 + r_m, N) - 1.0) / r_m;
        else            FV = pmt * N;
        snprintf(buf, sizeof(buf),
                 "Savings Goal: $%.2f/month at %.2f%% for %d years = $%.2f",
                 pmt, annual_r, years, FV);
        res.results[res.n_results++] = xstrdup(buf);
    }

    /* Debt Payoff (time to pay off with fixed payment) */
    if ((ci_contains(query, "debt") || ci_contains(query, "payoff") ||
         ci_contains(query, "pay off")) && n >= 3) {
        double balance = nums[0], annual_r = nums[1], pmt = nums[2];
        double r_m = annual_r / 100.0 / 12.0;
        if (r_m > 0.0 && pmt > balance * r_m) {
            double months = -log(1.0 - balance * r_m / pmt) / log(1.0 + r_m);
            snprintf(buf, sizeof(buf),
                     "Debt Payoff: $%.2f balance at %.2f%% paying $%.2f/month = %.1f months (%.1f years)",
                     balance, annual_r, pmt, months, months / 12.0);
            res.results[res.n_results++] = xstrdup(buf);
        }
    }

    if (res.n_results == 0) {
        free(res.results);
        res.results = NULL;
    }
    return res;
}

void calc_result_free(CalcResult* c) {
    if (!c || !c->results) return;
    for (int i = 0; i < c->n_results; i++) free(c->results[i]);
    free(c->results);
    c->results   = NULL;
    c->n_results = 0;
}

/* ── Orchestrator ─────────────────────────────────────────────────── */
Orchestrator* orchestrator_create(KnowledgeBase* kb, Model* model, Tokenizer* tok) {
    Orchestrator* o = (Orchestrator*)xcalloc(1, sizeof(Orchestrator));
    o->kb    = kb;
    o->model = model;
    o->tok   = tok;
    return o;
}

void orchestrator_free(Orchestrator* o) {
    free(o);
}

void agent_result_free(AgentResult* r) {
    if (!r) return;
    free(r->agent_name);
    free(r->error);
    if (r->kb_results) kb_results_free(r->kb_results, r->n_kb_results);
    free(r->context);
    for (int i = 0; i < r->n_sub_questions; i++) free(r->sub_questions[i]);
    free(r->sub_questions);
    calc_result_free(&r->calcs);
    free(r->response);
}

void agent_results_free(AgentResult* results, int n) {
    if (!results) return;
    for (int i = 0; i < n; i++) agent_result_free(&results[i]);
    free(results);
}

/* ── Run model agent ──────────────────────────────────────────────── */
static char* run_model_agent(Orchestrator* o, const char* prompt) {
    if (!o->model || !o->tok) return xstrdup("");

    int prompt_len;
    int* ids = tok_encode(o->tok, prompt, &prompt_len, 0);

    /* Prepend BOS */
    int* full = (int*)xmalloc((size_t)(prompt_len + 1) * sizeof(int));
    full[0] = TOK_BOS_ID;
    memcpy(full + 1, ids, (size_t)prompt_len * sizeof(int));
    free(ids);
    prompt_len++;

    /* Truncate to fit context */
    int max_ctx = o->model->cfg.max_seq_len - GEN_MAX_NEW_TOKENS;
    if (max_ctx < 16) max_ctx = 16;
    if (prompt_len > max_ctx) {
        /* Keep BOS + tail of the prompt */
        memmove(full + 1, full + prompt_len - max_ctx + 1,
                (size_t)(max_ctx - 1) * sizeof(int));
        prompt_len = max_ctx;
    }

    int out_len;
    int* out_ids = model_generate(o->model, full, prompt_len,
                                   GEN_MAX_NEW_TOKENS, GEN_TEMPERATURE,
                                   GEN_TOP_K, GEN_TOP_P, GEN_REP_PENALTY,
                                   TOK_EOS_ID, &out_len);
    free(full);

    /* Decode only the new tokens */
    const int* new_ids = out_ids + prompt_len;
    int new_len = out_len - prompt_len;

    /* Trim at EOS or SEP */
    for (int i = 0; i < new_len; i++) {
        if (new_ids[i] == TOK_EOS_ID || new_ids[i] == TOK_SEP_ID) {
            new_len = i;
            break;
        }
    }

    char* text = tok_decode(o->tok, new_ids, new_len, 1);
    free(out_ids);

    /* Clean up: cut at common re-prompt patterns */
    const char* stoppers[] = {"Q:", "Question:", "\n\n\n", "===", NULL};
    for (int si = 0; stoppers[si]; si++) {
        char* pos = strstr(text, stoppers[si]);
        if (pos && (pos - text) > 30) *pos = '\0';
    }

    /* Trim trailing whitespace */
    size_t tlen = strlen(text);
    while (tlen > 0 && isspace((unsigned char)text[tlen-1])) text[--tlen] = '\0';

    /* Ensure output ends on a sentence boundary */
    if (tlen > 0 && text[tlen-1] != '.' && text[tlen-1] != '!' && text[tlen-1] != '?') {
        int last = -1;
        for (int i = (int)tlen - 1; i >= 0; i--) {
            if (text[i] == '.' || text[i] == '!' || text[i] == '?') {
                last = i;
                break;
            }
        }
        if (last > (int)tlen * 2 / 5) text[last + 1] = '\0';
    }

    return text;
}

/* ── Main orchestration ───────────────────────────────────────────── */
char* orchestrator_process(Orchestrator* o,
                            const char* query,
                            const HistoryTurn* history, int n_history,
                            AgentResult** agent_info_out, int* n_agents_out) {
    AgentResult* agents = (AgentResult*)xcalloc(4, sizeof(AgentResult));

    /* ── Phase 1: Knowledge + Calculation ────────────────────── */
    double t0 = now_sec();

    /* Knowledge Agent */
    agents[0].agent_name   = xstrdup("KnowledgeAgent");
    agents[0].kb_results   = kb_search(o->kb, query, KB_TOP_K);
    agents[0].n_kb_results = KB_TOP_K;
    /* Count actual results with non-zero score */
    int actual_kb = 0;
    for (int i = 0; i < KB_TOP_K; i++) {
        if (agents[0].kb_results[i].score > 0.0f) actual_kb++;
    }
    agents[0].n_kb_results = actual_kb;
    agents[0].success      = 1;
    agents[0].elapsed      = now_sec() - t0;

    /* Calculation Agent */
    double t1 = now_sec();
    agents[1].agent_name = xstrdup("CalculationAgent");
    agents[1].calcs      = calc_agent_run(query);
    agents[1].success    = 1;
    agents[1].elapsed    = now_sec() - t1;

    /* ── Phase 2: Reasoning Agent ─────────────────────────────── */
    double t2 = now_sec();
    agents[2].agent_name    = xstrdup("ReasoningAgent");
    agents[2].question_type = reasoning_classify(query);

    Decomposition* decomp = reasoning_decompose(query);
    agents[2].n_sub_questions = decomp->n_questions;
    agents[2].sub_questions   = (char**)xmalloc((size_t)decomp->n_questions * sizeof(char*));
    for (int i = 0; i < decomp->n_questions; i++)
        agents[2].sub_questions[i] = xstrdup(decomp->questions[i]);
    reasoning_decompose_free(decomp);

    agents[2].context = reasoning_build_context(query,
                            agents[0].kb_results, agents[0].n_kb_results,
                            history, n_history);
    agents[2].success = 1;
    agents[2].elapsed = now_sec() - t2;

    /* ── Phase 3: Model Agent ─────────────────────────────────── */
    double t3 = now_sec();
    agents[3].agent_name = xstrdup("ModelAgent");

    /* Build prompt using reasoning context and any calculation results */
    const char** calc_strs = NULL;
    int n_calcs = agents[1].calcs.n_results;
    if (n_calcs > 0) {
        calc_strs = (const char**)xmalloc((size_t)n_calcs * sizeof(const char*));
        for (int i = 0; i < n_calcs; i++)
            calc_strs[i] = agents[1].calcs.results[i];
    }

    char* prompt = reasoning_build_prompt(query, agents[2].context,
                                           calc_strs, n_calcs);
    free(calc_strs);

    agents[3].response = run_model_agent(o, prompt);
    free(prompt);
    agents[3].success = agents[3].response && strlen(agents[3].response) >= 20;
    agents[3].elapsed = now_sec() - t3;

    /* ── Synthesis ────────────────────────────────────────────── */
    float       best_kb_score = 0.0f;
    const KbResult* best_kb  = NULL;
    for (int i = 0; i < agents[0].n_kb_results; i++) {
        if (agents[0].kb_results[i].score > best_kb_score) {
            best_kb_score = agents[0].kb_results[i].score;
            best_kb       = &agents[0].kb_results[i];
        }
    }

    /* Build output */
    size_t out_cap = 8192;
    char*  out     = (char*)xmalloc(out_cap);
    size_t out_len = 0;
    out[0] = '\0';

#define OAPP(s) do {                                                         \
    size_t _sl = strlen(s);                                                  \
    while (out_len + _sl + 4 > out_cap) {                                    \
        out_cap *= 2;                                                         \
        out = (char*)xrealloc(out, out_cap);                                 \
    }                                                                         \
    memcpy(out + out_len, (s), _sl);                                         \
    out_len += _sl;                                                           \
    out[out_len] = '\0';                                                      \
} while (0)

    /* Prepend any calculations */
    if (n_calcs > 0) {
        OAPP("**Calculated:**\n");
        for (int i = 0; i < n_calcs; i++) {
            OAPP("  ");
            OAPP(agents[1].calcs.results[i]);
            OAPP("\n");
        }
        OAPP("\n");
    }

    const char* model_text = agents[3].response ? agents[3].response : "";
    int model_ok = (int)(strlen(model_text) >= 30);

    /* Use KB answer when score >= threshold or model output is too short */
    int use_kb = (!model_ok) || (best_kb_score >= KB_DIRECT_SCORE);

    if (model_ok && !use_kb) {
        OAPP(model_text);
    } else if (best_kb) {
        OAPP(best_kb->answer);
        OAPP("\n\n[Source: ");
        OAPP(best_kb->source);
        OAPP("]");
    } else if (model_ok) {
        OAPP(model_text);
    } else {
        OAPP("I don't have information on that yet. Try rephrasing your question.");
    }

#undef OAPP

    if (agent_info_out) {
        *agent_info_out = agents;
        *n_agents_out   = 4;
    } else {
        agent_results_free(agents, 4);
    }

    return out;
}
