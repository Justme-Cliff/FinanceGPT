#ifndef AGENTS_H
#define AGENTS_H
#include "compat.h"
#include "model.h"
#include "tokenizer.h"
#include "knowledge_base.h"
#include "reasoning.h"
#include "config.h"

/* Financial calculation result */
typedef struct {
    char** results;   /* array of calculation result strings */
    int    n_results;
} CalcResult;

/* Agent result */
typedef struct {
    char*  agent_name;
    int    success;
    double elapsed;
    char*  error;

    /* Knowledge agent fields */
    KbResult* kb_results;
    int       n_kb_results;

    /* Reasoning agent fields */
    char*        context;
    QuestionType question_type;
    char**       sub_questions;
    int          n_sub_questions;

    /* Calculation agent fields */
    CalcResult calcs;

    /* Model agent fields */
    char*  response;
} AgentResult;

typedef struct {
    KnowledgeBase* kb;
    Model*         model;
    Tokenizer*     tok;
} Orchestrator;

/* Create orchestrator with all agents */
Orchestrator* orchestrator_create(KnowledgeBase* kb, Model* model, Tokenizer* tok);
void          orchestrator_free  (Orchestrator* o);

/* Process a query through all 4 agents.
   Returns final response string (caller frees).
   agent_info is filled with per-agent details (caller frees with agent_info_free). */
char* orchestrator_process(Orchestrator* o,
                            const char* query,
                            const HistoryTurn* history, int n_history,
                            AgentResult** agent_info_out, int* n_agents_out);

void agent_result_free(AgentResult* r);
void agent_results_free(AgentResult* results, int n);

/* Financial formula detection and calculation */
CalcResult calc_agent_run(const char* query);
void       calc_result_free(CalcResult* c);

#endif /* AGENTS_H */
