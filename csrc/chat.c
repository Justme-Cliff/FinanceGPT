#include "chat.h"
#include "config.h"
#include "model.h"
#include "tokenizer.h"
#include "knowledge_base.h"
#include "reasoning.h"
#include "agents.h"
#include "conversation.h"
#include "math_ops.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <ctype.h>

/* ── ANSI color codes ──────────────────────────────────────────── */
#define CYAN    "\033[36m"
#define GREEN   "\033[32m"
#define YELLOW  "\033[33m"
#define WHITE   "\033[37m"
#define DIM     "\033[2m"
#define RESET   "\033[0m"
#define BOLD    "\033[1m"

/* Enable VT100 escape processing on Windows 10+.
   On POSIX this is a no-op. */
#ifdef PLATFORM_WINDOWS
static void enable_vt(void) {
    HANDLE h = GetStdHandle(STD_OUTPUT_HANDLE);
    if (h == INVALID_HANDLE_VALUE) return;
    DWORD mode = 0;
    if (!GetConsoleMode(h, &mode)) return;
    SetConsoleMode(h, mode | ENABLE_VIRTUAL_TERMINAL_PROCESSING);
}
#else
/* Suppress unused-function warning on non-Windows builds */
static void enable_vt(void) __attribute__((unused));
static void enable_vt(void) {}
#endif

/* ── Banner ────────────────────────────────────────────────────── */
static void print_banner(int epochs, float best_ppl, int kb_pairs, float model_params) {
    printf(CYAN);
    printf("╭─── FinanceGPT v1.0 (C Edition) ────────────────────────────────────────────╮\n");
    printf("│                                                                               │\n");
    printf("│          Welcome to  FinanceGPT!          │  Agents                          │\n");
    printf("│                                            │  ────────────────────────────   │\n");
    printf("│            _______                         │   * Knowledge    TF-IDF          │\n");
    printf("│           / $   $ \\                        │   * Calculation  Formulas        │\n");
    printf("│          | $ $ $ $ |                       │   * Reasoning    Chain-of-thought│\n");
    printf("│          |  $   $  |                       │   * Model        Transformer     │\n");
    printf("│           \\_______/                        │                                  │\n");
    printf("│                                            │  Commands                        │\n");
    printf("│      Multi-Agent Reasoning System          │  ────────────────────────────   │\n");
    printf("│       %.1fM params  /  31 datasets            │  /help  /agents  /history        │\n",
           model_params);
    printf("│       Epochs: %d  /  Best ppl: %.2f            │  /reset /clear  /info  exit      │\n",
           epochs, best_ppl > 0.0f ? best_ppl : 0.0f);
    printf("│       KB pairs: %d                          │                                  │\n",
           kb_pairs);
    printf("│                                            │                                  │\n");
    printf("╰───────────────────────────────────────────────────────────────────────────────╯\n");
    printf(RESET);
}

/* ── /help ─────────────────────────────────────────────────────── */
static void print_help(void) {
    printf(YELLOW "\n  Commands:\n" RESET);
    printf("  " CYAN "/help" RESET "     show this message\n");
    printf("  " CYAN "/agents" RESET "   per-agent breakdown for last query\n");
    printf("  " CYAN "/history" RESET "  show recent conversation history\n");
    printf("  " CYAN "/reset" RESET "    clear session memory\n");
    printf("  " CYAN "/clear" RESET "    clear screen\n");
    printf("  " CYAN "/info" RESET "     model & knowledge base stats\n");
    printf("  " CYAN "exit" RESET "      quit\n\n");
}

/* ── Word-wrapped print ─────────────────────────────────────────── */
static void wrap_print(const char* text, const char* indent) {
    const int  WIDTH      = 82;
    int        indent_len = (int)strlen(indent);
    int        col        = 0;
    const char* p         = text;

    while (*p) {
        if (*p == '\n') {
            printf("\n");
            col = 0;
            p++;
            continue;
        }
        /* Scan to end of word */
        const char* word_end = p;
        while (*word_end && *word_end != ' ' && *word_end != '\n') word_end++;
        int wlen = (int)(word_end - p);

        if (col == 0) {
            printf("%s", indent);
            col = indent_len;
        } else if (col + 1 + wlen > WIDTH) {
            printf("\n%s", indent);
            col = indent_len;
        } else {
            printf(" ");
            col++;
        }
        fwrite(p, 1, (size_t)wlen, stdout);
        col += wlen;
        p = word_end;
        if (*p == ' ') p++;
    }
    printf("\n");
}

/* ── Time/date query detection ─────────────────────────────────── */
static int is_time_query(const char* q) {
    static const char* patterns[] = {
        "what time", "current time", "what date", "today's date",
        "todays date", "what day", "what year", "what month",
        "whats the time", "whats the date", NULL
    };
    char lower[1024];
    strncpy(lower, q, sizeof(lower) - 1);
    lower[sizeof(lower) - 1] = '\0';
    for (int i = 0; lower[i]; i++)
        lower[i] = (char)tolower((unsigned char)lower[i]);
    for (int i = 0; patterns[i]; i++)
        if (strstr(lower, patterns[i])) return 1;
    return 0;
}

/* ── Trim leading/trailing whitespace in-place ─────────────────── */
static void trim_inplace(char* s) {
    /* Leading */
    int start = 0;
    while (s[start] && isspace((unsigned char)s[start])) start++;
    if (start > 0) memmove(s, s + start, strlen(s) - (size_t)start + 1);
    /* Trailing */
    int end = (int)strlen(s) - 1;
    while (end >= 0 && isspace((unsigned char)s[end])) s[end--] = '\0';
}

/* ── Main chat entry point ─────────────────────────────────────── */
void chat_main(void) {
    enable_vt();

    /* Verify trained model exists before loading anything else */
    {
        FILE* cf = fopen(CHECKPOINT_PATH, "rb");
        if (!cf) {
            fprintf(stderr,
                    "\n  " YELLOW "[ERROR]" RESET
                    " No trained model found at %s\n"
                    "  Run: financegpt /train\n\n",
                    CHECKPOINT_PATH);
            return;
        }
        fclose(cf);
    }

    printf("\n" CYAN "  Loading FinanceGPT...\n" RESET "\n");

    /* 1. Tokenizer */
    printf(CYAN "  [1/5] Tokenizer..." RESET);
    fflush(stdout);
    Tokenizer* tok = tok_load(TOKENIZER_PATH);
    if (!tok) { fprintf(stderr, " FAILED\n"); return; }
    printf(GREEN " ok" RESET "  (%d tokens)\n", tok->vocab_size);

    /* 2. Model */
    printf(CYAN "  [2/5] Model..." RESET);
    fflush(stdout);
    Model* model = model_load(CHECKPOINT_PATH);
    if (!model) { fprintf(stderr, " FAILED\n"); tok_free(tok); return; }
    printf(GREEN " ok" RESET "  (%.1fM params)\n",
           (float)model_n_params(model) / 1.0e6f);

    /* 3. Knowledge base */
    printf(CYAN "  [3/5] Knowledge base..." RESET);
    fflush(stdout);
    KnowledgeBase* kb = kb_create(DATA_DIR);
    printf(GREEN " ok" RESET "  (%d pairs)\n", kb_size(kb));

    /* 4. Reasoning engine — stateless, no explicit init */
    printf(CYAN "  [4/5] Reasoning engine..." RESET);
    printf(GREEN " ok" RESET "\n");

    /* 5. Orchestrator */
    printf(CYAN "  [5/5] Spawning agents..." RESET);
    Orchestrator* orch = orchestrator_create(kb, model, tok);
    printf(GREEN " ok" RESET
           "  [Knowledge | Calculation | Reasoning | Model]\n");

    /* Conversation memory */
    ConversationMemory* mem = conv_create(CONV_HISTORY_PATH, LOAD_HISTORY_TURNS);

    /* Clear screen and show banner */
    system(CLEAR_SCREEN);
    printf("\n");
    print_banner(0, 0.0f, kb_size(kb), (float)model_n_params(model) / 1.0e6f);
    printf("\n" DIM "  Memory turns loaded: %d" RESET "\n", mem->n_turns);
    printf(GREEN
           "  Ready -- ask anything about finance.  Type /help for commands.\n"
           RESET);

    /* Storage for last agent info (used by /agents) */
    AgentResult* last_agents  = NULL;
    int          last_n_agents = 0;

    const char* divider =
        DIM "──────────────────────────────────────────────────────────────"
            "────────────────────" RESET "\n";

    char line[4096];

    while (1) {
        printf("%s", divider);
        printf(YELLOW "> " RESET);
        fflush(stdout);

        if (!fgets(line, (int)sizeof(line), stdin)) break;

        /* Strip trailing newline */
        size_t ll = strlen(line);
        while (ll > 0 && (line[ll-1] == '\n' || line[ll-1] == '\r'))
            line[--ll] = '\0';
        if (ll == 0) continue;

        /* Lowercase copy for command matching */
        char cmd[4096];
        strncpy(cmd, line, sizeof(cmd) - 1);
        cmd[sizeof(cmd) - 1] = '\0';
        for (int i = 0; cmd[i]; i++)
            cmd[i] = (char)tolower((unsigned char)cmd[i]);
        trim_inplace(cmd);

        /* ── Time/date shortcut ─────────────────────────────────── */
        if (is_time_query(cmd)) {
            time_t     t       = time(NULL);
            struct tm* tm_info = localtime(&t);
            char       msg[128];
            strftime(msg, sizeof(msg), "It's %A, %B %d, %Y -- %I:%M %p.", tm_info);
            printf("\n" GREEN "  FinanceGPT >" RESET "\n");
            wrap_print(msg, "  ");
            printf("\n");
            conv_add_turn(mem, line, msg);
            continue;
        }

        /* ── exit / quit ───────────────────────────────────────── */
        if (strcmp(cmd, "exit")  == 0 || strcmp(cmd, "quit")  == 0 ||
            strcmp(cmd, "bye")   == 0 || strcmp(cmd, "/exit") == 0) {
            printf("%s" YELLOW "  Goodbye!\n\n" RESET, divider);
            break;
        }

        /* ── /help ─────────────────────────────────────────────── */
        if (strcmp(cmd, "/help") == 0) {
            print_help();
            continue;
        }

        /* ── /clear ────────────────────────────────────────────── */
        if (strcmp(cmd, "/clear") == 0) {
            system(CLEAR_SCREEN);
            printf("\n");
            print_banner(0, 0.0f, kb_size(kb),
                         (float)model_n_params(model) / 1.0e6f);
            continue;
        }

        /* ── /reset ────────────────────────────────────────────── */
        if (strcmp(cmd, "/reset") == 0) {
            conv_clear_session(mem);
            printf("\n" YELLOW "  Session memory cleared.\n" RESET "\n");
            continue;
        }

        /* ── /history ──────────────────────────────────────────── */
        if (strcmp(cmd, "/history") == 0) {
            printf("\n" CYAN "  Recent conversation:\n" RESET);
            char* hist = conv_format_recent(mem, 5);
            printf("%s\n", hist);
            free(hist);
            continue;
        }

        /* ── /agents ───────────────────────────────────────────── */
        if (strcmp(cmd, "/agents") == 0) {
            if (last_agents && last_n_agents > 0) {
                printf("\n" CYAN "  Agent activity (last query):\n" RESET);
                static const char* names[] = {
                    "KnowledgeAgent", "CalculationAgent",
                    "ReasoningAgent", "ModelAgent"
                };
                for (int i = 0; i < last_n_agents && i < 4; i++) {
                    AgentResult* a  = &last_agents[i];
                    const char*  ok = a->success
                                      ? GREEN "ok" RESET
                                      : "\033[31mFAIL" RESET;
                    char detail[128] = {0};
                    if (i == 0) {
                        snprintf(detail, sizeof(detail),
                                 "retrieved %d docs", a->n_kb_results);
                    } else if (i == 1) {
                        snprintf(detail, sizeof(detail),
                                 "%d calculations", a->calcs.n_results);
                    } else if (i == 2) {
                        snprintf(detail, sizeof(detail),
                                 "type=%s",
                                 reasoning_qtype_str(a->question_type));
                    } else {
                        snprintf(detail, sizeof(detail),
                                 "generated %zu chars",
                                 a->response ? strlen(a->response) : (size_t)0);
                    }
                    printf("  [%s] %-22s %-30s (%.2fs)\n",
                           ok, names[i], detail, a->elapsed);
                }
                printf("\n");
            } else {
                printf("\n" YELLOW "  No queries yet.\n" RESET "\n");
            }
            continue;
        }

        /* ── /info ─────────────────────────────────────────────── */
        if (strcmp(cmd, "/info") == 0) {
            printf("\n" CYAN "  System info:\n" RESET);
            printf("  Model params   : %zu\n",   model_n_params(model));
            printf("  Vocab size     : %d\n",    tok->vocab_size);
            printf("  Context window : %d tokens\n", model->cfg.max_seq_len);
            printf("  KB pairs       : %d\n",    kb_size(kb));
            printf("  Conv. turns    : %d\n",    mem->n_turns);
            printf("\n");
            continue;
        }

        /* ── Multi-agent query ─────────────────────────────────── */
        printf("\n" CYAN "  Agents thinking..." RESET);
        fflush(stdout);

        double t_start = now_sec();

        /* Gather conversation history */
        int             n_hist;
        const ConvTurn* hist_turns =
            conv_get_context(mem, 5, &n_hist);
        HistoryTurn* history =
            (HistoryTurn*)xcalloc((size_t)(n_hist > 0 ? n_hist : 1),
                                   sizeof(HistoryTurn));
        for (int i = 0; i < n_hist; i++) {
            history[i].question = (char*)hist_turns[i].question;
            history[i].answer   = (char*)hist_turns[i].answer;
        }

        /* Free previous agent info */
        if (last_agents) {
            agent_results_free(last_agents, last_n_agents);
            last_agents   = NULL;
            last_n_agents = 0;
        }

        char* response = orchestrator_process(orch, line,
                                               history, n_hist,
                                               &last_agents,
                                               &last_n_agents);
        double elapsed = now_sec() - t_start;
        free(history);

        /* Clear the "thinking..." line */
        printf("\r                              \r");

        /* Print response */
        printf(GREEN "  FinanceGPT >" RESET "\n");
        wrap_print(response ? response : "(no response)", "  ");

        /* Footer with source and question type */
        char src_str[256] = {0};
        if (last_agents && last_n_agents > 0 &&
            last_agents[0].n_kb_results > 0 &&
            last_agents[0].kb_results[0].source) {
            snprintf(src_str, sizeof(src_str),
                     "  Sources: %s  |  ",
                     last_agents[0].kb_results[0].source);
        }
        const char* qtype_str = (last_agents && last_n_agents >= 3)
            ? reasoning_qtype_str(last_agents[2].question_type)
            : "general";
        printf("\n" CYAN DIM "%stype: %s  |  %.1fs" RESET "\n\n",
               src_str, qtype_str, elapsed);

        conv_add_turn(mem, line, response ? response : "");
        free(response);
    }

    /* ── Cleanup ──────────────────────────────────────────────── */
    if (last_agents) agent_results_free(last_agents, last_n_agents);
    conv_free(mem);
    orchestrator_free(orch);
    kb_free(kb);
    model_free(model);
    tok_free(tok);
}
