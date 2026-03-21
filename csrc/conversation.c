#include "conversation.h"
#include "config.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <time.h>

static char* make_session_id(void) {
    time_t t = time(NULL);
    struct tm* tm_info = localtime(&t);
    char buf[32];
    strftime(buf, sizeof(buf), "%Y%m%d_%H%M%S", tm_info);
    return xstrdup(buf);
}

static char* make_timestamp(void) {
    time_t t = time(NULL);
    struct tm* tm_info = localtime(&t);
    char buf[32];
    strftime(buf, sizeof(buf), "%Y-%m-%dT%H:%M:%S", tm_info);
    return xstrdup(buf);
}

ConversationMemory* conv_create(const char* path, int load_turns) {
    ConversationMemory* m = (ConversationMemory*)xcalloc(1, sizeof(ConversationMemory));
    m->path          = xstrdup(path);
    m->session_id    = make_session_id();
    m->max_sessions  = MAX_SESSIONS;
    m->runtime_limit = MAX_HISTORY_TURNS;

    m->cap_turns     = 64;
    m->turns         = (ConvTurn*)xmalloc(m->cap_turns * sizeof(ConvTurn));
    m->cap_session   = 32;
    m->session_turns = (ConvTurn*)xmalloc(m->cap_session * sizeof(ConvTurn));

    /* Load last load_turns from disk */
    JsonNode* root = json_parse_file(path);
    if (root) {
        JsonNode* sessions = json_get(root, "sessions");
        if (sessions) {
            /* First pass: count all turns */
            int total = 0;
            for (int si = 0; si < json_len(sessions); si++) {
                JsonNode* sess  = json_get_index(sessions, si);
                JsonNode* turns = json_get(sess, "turns");
                if (turns) total += json_len(turns);
            }
            /* Load only the last load_turns */
            int skip = total > load_turns ? total - load_turns : 0;
            int seen = 0;
            for (int si = 0; si < json_len(sessions); si++) {
                JsonNode* sess  = json_get_index(sessions, si);
                JsonNode* turns = json_get(sess, "turns");
                if (!turns) continue;
                for (int ti = 0; ti < json_len(turns); ti++) {
                    if (seen++ < skip) continue;
                    JsonNode* turn = json_get_index(turns, ti);
                    if (!turn) continue;
                    if (m->n_turns >= m->cap_turns) {
                        m->cap_turns *= 2;
                        m->turns = (ConvTurn*)xrealloc(m->turns,
                                        m->cap_turns * sizeof(ConvTurn));
                    }
                    ConvTurn* ct  = &m->turns[m->n_turns++];
                    ct->session   = xstrdup(json_str(json_get(turn, "session"),   ""));
                    ct->timestamp = xstrdup(json_str(json_get(turn, "timestamp"), ""));
                    ct->question  = xstrdup(json_str(json_get(turn, "question"),  ""));
                    ct->answer    = xstrdup(json_str(json_get(turn, "answer"),    ""));
                }
            }
        }
        json_free(root);
    }
    return m;
}

static void conv_turn_free_data(ConvTurn* t) {
    free(t->session);
    free(t->timestamp);
    free(t->question);
    free(t->answer);
    t->session = t->timestamp = t->question = t->answer = NULL;
}

void conv_free(ConversationMemory* m) {
    if (!m) return;
    for (int i = 0; i < m->n_turns; i++)   conv_turn_free_data(&m->turns[i]);
    free(m->turns);
    for (int i = 0; i < m->n_session; i++) conv_turn_free_data(&m->session_turns[i]);
    free(m->session_turns);
    free(m->path);
    free(m->session_id);
    free(m);
}

static void conv_save(ConversationMemory* m) {
    /* Load existing JSON or create a fresh root */
    JsonNode* root = json_parse_file(m->path);
    if (!root) {
        root = json_new_obj(NULL);
        json_append(root, json_new_arr("sessions"));
    }

    JsonNode* sessions = json_get(root, "sessions");
    if (!sessions) {
        sessions = json_new_arr("sessions");
        json_append(root, sessions);
    }

    /* Find or create an entry for the current session */
    JsonNode* current = NULL;
    for (int i = 0; i < json_len(sessions); i++) {
        JsonNode* s = json_get_index(sessions, i);
        if (s && strcmp(json_str(json_get(s, "id"), ""), m->session_id) == 0) {
            current = s;
            break;
        }
    }
    if (!current) {
        current = json_new_obj(NULL);
        json_append(current, json_new_str("id", m->session_id));
        json_append(current, json_new_arr("turns"));
        json_append(sessions, current);
    }

    /* Get (or create) the turns array inside the session object */
    JsonNode* turns_node = json_get(current, "turns");
    if (!turns_node) {
        turns_node = json_new_arr("turns");
        json_append(current, turns_node);
    }

    /* Clear existing turn nodes */
    for (int i = 0; i < json_len(turns_node); i++) {
        json_free(turns_node->arr.items[i]);
        turns_node->arr.items[i] = NULL;
    }
    turns_node->arr.count = 0;

    /* Re-populate from session_turns */
    for (int i = 0; i < m->n_session; i++) {
        ConvTurn*  t    = &m->session_turns[i];
        JsonNode*  turn = json_new_obj(NULL);
        json_append(turn, json_new_str("session",   t->session));
        json_append(turn, json_new_str("timestamp", t->timestamp));
        json_append(turn, json_new_str("question",  t->question));
        json_append(turn, json_new_str("answer",    t->answer));
        json_append(turns_node, turn);
    }

    /* Trim to max_sessions (oldest first) */
    while (json_len(sessions) > m->max_sessions) {
        json_free(sessions->arr.items[0]);
        memmove(sessions->arr.items,
                sessions->arr.items + 1,
                (size_t)(json_len(sessions) - 1) * sizeof(JsonNode*));
        sessions->arr.count--;
    }

    json_write_file(root, m->path);
    json_free(root);
}

void conv_add_turn(ConversationMemory* m, const char* q, const char* a) {
    /* Build the new turn */
    ConvTurn t;
    t.session   = xstrdup(m->session_id);
    t.timestamp = make_timestamp();
    t.question  = xstrdup(q);
    t.answer    = xstrdup(a);

    /* Append to all-turns array */
    if (m->n_turns >= m->cap_turns) {
        m->cap_turns *= 2;
        m->turns = (ConvTurn*)xrealloc(m->turns, m->cap_turns * sizeof(ConvTurn));
    }
    m->turns[m->n_turns++] = t;

    /* Append a copy to session_turns */
    ConvTurn t2;
    t2.session   = xstrdup(t.session);
    t2.timestamp = xstrdup(t.timestamp);
    t2.question  = xstrdup(t.question);
    t2.answer    = xstrdup(t.answer);
    if (m->n_session >= m->cap_session) {
        m->cap_session *= 2;
        m->session_turns = (ConvTurn*)xrealloc(m->session_turns,
                                m->cap_session * sizeof(ConvTurn));
    }
    m->session_turns[m->n_session++] = t2;

    /* Trim RAM to runtime_limit (oldest entries) */
    while (m->n_turns > m->runtime_limit) {
        conv_turn_free_data(&m->turns[0]);
        memmove(m->turns, m->turns + 1, (size_t)(m->n_turns - 1) * sizeof(ConvTurn));
        m->n_turns--;
    }

    conv_save(m);
}

void conv_clear_session(ConversationMemory* m) {
    /* Free and remove session turns */
    for (int i = 0; i < m->n_session; i++)
        conv_turn_free_data(&m->session_turns[i]);
    m->n_session = 0;

    /* Remove matching turns from the global array */
    int new_n = 0;
    for (int i = 0; i < m->n_turns; i++) {
        if (strcmp(m->turns[i].session, m->session_id) == 0) {
            conv_turn_free_data(&m->turns[i]);
        } else {
            m->turns[new_n++] = m->turns[i];
        }
    }
    m->n_turns = new_n;

    conv_save(m);
}

void conv_clear_all(ConversationMemory* m) {
    for (int i = 0; i < m->n_turns; i++)   conv_turn_free_data(&m->turns[i]);
    m->n_turns = 0;
    for (int i = 0; i < m->n_session; i++) conv_turn_free_data(&m->session_turns[i]);
    m->n_session = 0;
    remove(m->path);
}

const ConvTurn* conv_get_context(ConversationMemory* m, int n_turns, int* out_n) {
    int start = m->n_turns > n_turns ? m->n_turns - n_turns : 0;
    *out_n    = m->n_turns - start;
    return m->turns + start;
}

char* conv_format_recent(ConversationMemory* m, int n_turns) {
    int n;
    const ConvTurn* turns = conv_get_context(m, n_turns, &n);
    if (n == 0) return xstrdup("  (no history yet)");

    size_t cap = 4096;
    char*  out = (char*)xmalloc(cap);
    size_t len = 0;

    for (int i = 0; i < n; i++) {
        char line[640];
        int  ll;

        /* Format "User:" line */
        ll = snprintf(line, sizeof(line), "User: %.512s\n", turns[i].question);
        if (ll < 0) ll = 0;
        while (len + (size_t)ll + 4 > cap) { cap *= 2; out = (char*)xrealloc(out, cap); }
        memcpy(out + len, line, (size_t)ll);
        len += (size_t)ll;

        /* Format "Assistant:" line with full answer */
        ll = snprintf(line, sizeof(line), "Assistant: %.512s\n", turns[i].answer);
        if (ll < 0) ll = 0;
        while (len + (size_t)ll + 4 > cap) { cap *= 2; out = (char*)xrealloc(out, cap); }
        memcpy(out + len, line, (size_t)ll);
        len += (size_t)ll;

        /* Blank line between turns */
        while (len + 4 > cap) { cap *= 2; out = (char*)xrealloc(out, cap); }
        out[len++] = '\n';
    }
    out[len] = '\0';
    return out;
}
