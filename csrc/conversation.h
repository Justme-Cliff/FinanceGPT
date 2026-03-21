#ifndef CONVERSATION_H
#define CONVERSATION_H
#include "compat.h"
#include "json.h"

/* JSON-backed persistent conversation history — port of conversation_memory.py */

typedef struct {
    char* session;
    char* timestamp;
    char* question;
    char* answer;
} ConvTurn;

typedef struct {
    ConvTurn* turns;      /* all turns in RAM */
    int       n_turns;
    int       cap_turns;

    ConvTurn* session_turns;  /* this session only */
    int       n_session;
    int       cap_session;

    char*     path;
    char*     session_id;

    int       max_sessions;
    int       runtime_limit;
} ConversationMemory;

ConversationMemory* conv_create(const char* path, int load_turns);
void                conv_free  (ConversationMemory* m);

void        conv_add_turn    (ConversationMemory* m, const char* q, const char* a);
void        conv_clear_session(ConversationMemory* m);
void        conv_clear_all   (ConversationMemory* m);

/* Get last n_turns for context (pointers into internal array — don't free) */
const ConvTurn* conv_get_context(ConversationMemory* m, int n_turns, int* out_n);

/* Format recent turns as human-readable string (caller frees) */
char* conv_format_recent(ConversationMemory* m, int n_turns);

#endif /* CONVERSATION_H */
