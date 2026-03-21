#ifndef REASONING_H
#define REASONING_H
#include "compat.h"

typedef enum {
    QTYPE_CALCULATION,
    QTYPE_COMPARISON,
    QTYPE_DEFINITION,
    QTYPE_CAUSAL,
    QTYPE_STRATEGY,
    QTYPE_PROCESS,
    QTYPE_HISTORICAL,
    QTYPE_RISK,
    QTYPE_GENERAL
} QuestionType;

typedef struct {
    char** questions;  /* decomposed sub-questions */
    int    n_questions;
} Decomposition;

/* Classify question into one of 8 types */
QuestionType reasoning_classify(const char* question);
const char*  reasoning_qtype_str(QuestionType qt);

/* Get reasoning scaffold for this type */
const char*  reasoning_scaffold(QuestionType qt);

/* Decompose compound question into sub-questions */
Decomposition* reasoning_decompose(const char* question);
void           reasoning_decompose_free(Decomposition* d);

/* Normalize query (apply rephrase rules) */
void reasoning_normalize(const char* in, char* out, int out_cap);

/* Build context string from retrieved docs + history + scaffold.
   Returns malloc'd string, caller frees.
   results_void is a KbResult* (void* to avoid circular header dependency). */
typedef struct {
    char* question;
    char* answer;
} HistoryTurn;

char* reasoning_build_context(const char* question,
                               const void* results_void, int n_results,
                               const HistoryTurn* history, int n_history);

/* Build final prompt for model */
char* reasoning_build_prompt(const char* question, const char* context,
                              const char** calc_results, int n_calcs);

#endif /* REASONING_H */
