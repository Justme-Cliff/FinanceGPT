#include "json.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <ctype.h>
#include <math.h>

#ifdef PLATFORM_WINDOWS
#  include <direct.h>   /* _mkdir */
#endif

/* ── Parser state ──────────────────────────────────────────────────── */

typedef struct {
    const char* p;
    const char* end;
    int         error;
} Parser;

static void skip_ws(Parser* ps) {
    while (ps->p < ps->end && isspace((unsigned char)*ps->p))
        ps->p++;
}

static char peek_c(Parser* ps) {
    skip_ws(ps);
    return (ps->p < ps->end) ? *ps->p : '\0';
}

/* peek without advancing, no ws-skip */
static char raw_peek(Parser* ps) {
    return (ps->p < ps->end) ? *ps->p : '\0';
}

static JsonNode* node_new(JsonType t, const char* key) {
    JsonNode* n = (JsonNode*)xcalloc(1, sizeof(JsonNode));
    n->type = t;
    n->key  = key ? xstrdup(key) : NULL;
    return n;
}

/* ── String parsing ────────────────────────────────────────────────── */

/* Decode one \uXXXX codepoint; p points AFTER the 'u'.
   Writes UTF-8 bytes into out, returns bytes written (1-4).
   Advances *pp by 4 hex digits (or fewer if truncated). */
static int decode_unicode_escape(const char** pp, const char* end, char* out) {
    unsigned int cp = 0;
    for (int i = 0; i < 4 && *pp < end; i++) {
        char c = **pp; (*pp)++;
        unsigned int nibble;
        if      (c >= '0' && c <= '9') nibble = (unsigned int)(c - '0');
        else if (c >= 'a' && c <= 'f') nibble = (unsigned int)(c - 'a' + 10);
        else if (c >= 'A' && c <= 'F') nibble = (unsigned int)(c - 'A' + 10);
        else { nibble = 0; }
        cp = (cp << 4) | nibble;
    }

    /* Encode as UTF-8 */
    if (cp < 0x80) {
        out[0] = (char)cp;
        return 1;
    } else if (cp < 0x800) {
        out[0] = (char)(0xC0 | (cp >> 6));
        out[1] = (char)(0x80 | (cp & 0x3F));
        return 2;
    } else if (cp < 0x10000) {
        out[0] = (char)(0xE0 | (cp >> 12));
        out[1] = (char)(0x80 | ((cp >> 6) & 0x3F));
        out[2] = (char)(0x80 | (cp & 0x3F));
        return 3;
    } else {
        out[0] = (char)(0xF0 | (cp >> 18));
        out[1] = (char)(0x80 | ((cp >> 12) & 0x3F));
        out[2] = (char)(0x80 | ((cp >> 6)  & 0x3F));
        out[3] = (char)(0x80 | (cp & 0x3F));
        return 4;
    }
}

static char* parse_string_raw(Parser* ps) {
    skip_ws(ps);
    if (ps->p >= ps->end || *ps->p != '"') {
        ps->error = 1;
        return xstrdup("");
    }
    ps->p++; /* skip opening '"' */

    size_t cap = 128;
    char*  buf = (char*)xmalloc(cap);
    size_t len = 0;

    while (ps->p < ps->end && *ps->p != '"') {
        /* Ensure at least 8 bytes of headroom for UTF-8 + null */
        if (len + 8 >= cap) {
            cap *= 2;
            buf = (char*)xrealloc(buf, cap);
        }

        if (*ps->p == '\\') {
            ps->p++; /* skip backslash */
            if (ps->p >= ps->end) break;
            char esc = *ps->p++;
            switch (esc) {
                case '"':  buf[len++] = '"';  break;
                case '\\': buf[len++] = '\\'; break;
                case '/':  buf[len++] = '/';  break;
                case 'n':  buf[len++] = '\n'; break;
                case 'r':  buf[len++] = '\r'; break;
                case 't':  buf[len++] = '\t'; break;
                case 'b':  buf[len++] = '\b'; break;
                case 'f':  buf[len++] = '\f'; break;
                case 'u': {
                    char utf8[4];
                    int nb = decode_unicode_escape(&ps->p, ps->end, utf8);
                    /* Check for surrogate pair \uD800-\uDBFF followed by \uDC00-\uDFFF */
                    /* We only handle BMP here; surrogate pairs fall back to replacement char */
                    if (len + (size_t)nb + 8 >= cap) { cap *= 2; buf = (char*)xrealloc(buf, cap); }
                    memcpy(buf + len, utf8, (size_t)nb);
                    len += (size_t)nb;
                    break;
                }
                default:
                    buf[len++] = esc;
                    break;
            }
        } else {
            /* Raw UTF-8 byte — copy multi-byte sequences intact */
            unsigned char c = (unsigned char)*ps->p;
            int bytes = 1;
            if      (c >= 0xF0) bytes = 4;
            else if (c >= 0xE0) bytes = 3;
            else if (c >= 0xC0) bytes = 2;

            if (len + (size_t)bytes + 8 >= cap) { cap *= 2; buf = (char*)xrealloc(buf, cap); }
            for (int i = 0; i < bytes && ps->p < ps->end; i++)
                buf[len++] = *ps->p++;
        }
    }

    if (ps->p < ps->end && *ps->p == '"') ps->p++; /* skip closing '"' */
    buf[len] = '\0';
    return buf;
}

/* ── Forward declaration ────────────────────────────────────────────── */
static JsonNode* parse_value(Parser* ps, const char* key);

/* ── Object parser ──────────────────────────────────────────────────── */
static JsonNode* parse_object(Parser* ps, const char* key) {
    ps->p++; /* skip '{' */
    JsonNode* obj   = node_new(JSON_OBJECT, key);
    obj->arr.cap    = 8;
    obj->arr.items  = (JsonNode**)xmalloc(obj->arr.cap * sizeof(JsonNode*));
    obj->arr.count  = 0;

    while (!ps->error && ps->p < ps->end) {
        char c = peek_c(ps);
        if (c == '}') { ps->p++; break; }
        if (c == ',') { ps->p++; continue; }
        if (c == '\0') break;

        if (c != '"') { ps->error = 1; break; }

        char* k = parse_string_raw(ps);
        skip_ws(ps);
        if (peek_c(ps) == ':') ps->p++;

        JsonNode* child = parse_value(ps, k);
        free(k);

        if (child) {
            if (obj->arr.count >= obj->arr.cap) {
                obj->arr.cap *= 2;
                obj->arr.items = (JsonNode**)xrealloc(obj->arr.items,
                                  obj->arr.cap * sizeof(JsonNode*));
            }
            obj->arr.items[obj->arr.count++] = child;
        }
    }
    return obj;
}

/* ── Array parser ───────────────────────────────────────────────────── */
static JsonNode* parse_array(Parser* ps, const char* key) {
    ps->p++; /* skip '[' */
    JsonNode* arr  = node_new(JSON_ARRAY, key);
    arr->arr.cap   = 8;
    arr->arr.items = (JsonNode**)xmalloc(arr->arr.cap * sizeof(JsonNode*));
    arr->arr.count = 0;

    while (!ps->error && ps->p < ps->end) {
        char c = peek_c(ps);
        if (c == ']') { ps->p++; break; }
        if (c == ',') { ps->p++; continue; }
        if (c == '\0') break;

        JsonNode* child = parse_value(ps, NULL);
        if (child) {
            if (arr->arr.count >= arr->arr.cap) {
                arr->arr.cap *= 2;
                arr->arr.items = (JsonNode**)xrealloc(arr->arr.items,
                                  arr->arr.cap * sizeof(JsonNode*));
            }
            arr->arr.items[arr->arr.count++] = child;
        }
    }
    return arr;
}

/* ── Value dispatcher ───────────────────────────────────────────────── */
static JsonNode* parse_value(Parser* ps, const char* key) {
    skip_ws(ps);
    if (ps->p >= ps->end) return NULL;

    char c = *ps->p;

    if (c == '{') return parse_object(ps, key);
    if (c == '[') return parse_array(ps, key);

    if (c == '"') {
        char* s = parse_string_raw(ps);
        JsonNode* n = node_new(JSON_STRING, key);
        n->s = s;
        return n;
    }

    /* true */
    if (c == 't' && ps->p + 3 < ps->end &&
        ps->p[1]=='r' && ps->p[2]=='u' && ps->p[3]=='e') {
        ps->p += 4;
        JsonNode* n = node_new(JSON_BOOL, key);
        n->b = 1;
        return n;
    }

    /* false */
    if (c == 'f' && ps->p + 4 < ps->end &&
        ps->p[1]=='a' && ps->p[2]=='l' && ps->p[3]=='s' && ps->p[4]=='e') {
        ps->p += 5;
        JsonNode* n = node_new(JSON_BOOL, key);
        n->b = 0;
        return n;
    }

    /* null */
    if (c == 'n' && ps->p + 3 < ps->end &&
        ps->p[1]=='u' && ps->p[2]=='l' && ps->p[3]=='l') {
        ps->p += 4;
        return node_new(JSON_NULL, key);
    }

    /* number */
    if (c == '-' || c == '+' || isdigit((unsigned char)c)) {
        char* endp;
        double val = strtod(ps->p, &endp);
        if (endp == ps->p) { ps->error = 1; return NULL; }
        ps->p = endp;
        JsonNode* n = node_new(JSON_NUMBER, key);
        n->n = val;
        return n;
    }

    /* Unknown character — skip it to attempt recovery */
    ps->p++;
    return NULL;
}

/* ── Public parse API ──────────────────────────────────────────────── */

JsonNode* json_parse(const char* text) {
    if (!text) return NULL;
    Parser ps;
    ps.p     = text;
    ps.end   = text + strlen(text);
    ps.error = 0;
    JsonNode* root = parse_value(&ps, NULL);
    if (ps.error) {
        json_free(root);
        return NULL;
    }
    return root;
}

JsonNode* json_parse_file(const char* path) {
    FILE* f = fopen(path, "rb");
    if (!f) return NULL;

    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    rewind(f);

    if (sz <= 0) { fclose(f); return NULL; }

    char* buf = (char*)xmalloc((size_t)sz + 1);
    size_t rd = fread(buf, 1, (size_t)sz, f);
    fclose(f);
    buf[rd] = '\0';

    JsonNode* root = json_parse(buf);
    free(buf);
    return root;
}

/* ── Memory management ─────────────────────────────────────────────── */

void json_free(JsonNode* n) {
    if (!n) return;
    free(n->key);
    switch (n->type) {
        case JSON_STRING:
            free(n->s);
            break;
        case JSON_ARRAY:
        case JSON_OBJECT:
            for (int i = 0; i < n->arr.count; i++)
                json_free(n->arr.items[i]);
            free(n->arr.items);
            break;
        default:
            break;
    }
    free(n);
}

/* ── Access helpers ────────────────────────────────────────────────── */

JsonNode* json_get(JsonNode* obj, const char* key) {
    if (!obj || !key) return NULL;
    if (obj->type != JSON_OBJECT && obj->type != JSON_ARRAY) return NULL;
    for (int i = 0; i < obj->arr.count; i++) {
        JsonNode* c = obj->arr.items[i];
        if (c && c->key && strcmp(c->key, key) == 0) return c;
    }
    return NULL;
}

JsonNode* json_get_index(JsonNode* arr, int idx) {
    if (!arr) return NULL;
    if (arr->type != JSON_ARRAY && arr->type != JSON_OBJECT) return NULL;
    if (idx < 0 || idx >= arr->arr.count) return NULL;
    return arr->arr.items[idx];
}

const char* json_str(JsonNode* n, const char* def) {
    if (!n || n->type != JSON_STRING) return def;
    return n->s;
}

double json_num(JsonNode* n, double def) {
    if (!n || n->type != JSON_NUMBER) return def;
    return n->n;
}

int json_bool(JsonNode* n, int def) {
    if (!n || n->type != JSON_BOOL) return def;
    return n->b;
}

int json_len(JsonNode* n) {
    if (!n) return 0;
    if (n->type != JSON_ARRAY && n->type != JSON_OBJECT) return 0;
    return n->arr.count;
}

/* ── Builder helpers ───────────────────────────────────────────────── */

static JsonNode* make_container(JsonType t, const char* key) {
    JsonNode* n   = node_new(t, key);
    n->arr.cap    = 8;
    n->arr.items  = (JsonNode**)xmalloc(n->arr.cap * sizeof(JsonNode*));
    n->arr.count  = 0;
    return n;
}

JsonNode* json_new_str(const char* key, const char* val) {
    JsonNode* n = node_new(JSON_STRING, key);
    n->s = xstrdup(val ? val : "");
    return n;
}

JsonNode* json_new_num(const char* key, double val) {
    JsonNode* n = node_new(JSON_NUMBER, key);
    n->n = val;
    return n;
}

JsonNode* json_new_bool(const char* key, int val) {
    JsonNode* n = node_new(JSON_BOOL, key);
    n->b = (val != 0);
    return n;
}

JsonNode* json_new_obj(const char* key) {
    return make_container(JSON_OBJECT, key);
}

JsonNode* json_new_arr(const char* key) {
    return make_container(JSON_ARRAY, key);
}

void json_append(JsonNode* parent, JsonNode* child) {
    if (!parent || !child) return;
    if (parent->type != JSON_OBJECT && parent->type != JSON_ARRAY) return;
    if (parent->arr.count >= parent->arr.cap) {
        parent->arr.cap *= 2;
        parent->arr.items = (JsonNode**)xrealloc(parent->arr.items,
                            parent->arr.cap * sizeof(JsonNode*));
    }
    parent->arr.items[parent->arr.count++] = child;
}

void json_set_str(JsonNode* obj, const char* key, const char* val) {
    if (!obj || !key) return;
    /* Update in-place if key exists */
    JsonNode* existing = json_get(obj, key);
    if (existing && existing->type == JSON_STRING) {
        free(existing->s);
        existing->s = xstrdup(val ? val : "");
        return;
    }
    /* Otherwise append a new node */
    json_append(obj, json_new_str(key, val));
}

/* ── Serializer ────────────────────────────────────────────────────── */

typedef struct {
    char*  buf;
    size_t len;
    size_t cap;
} SBuf;

static void sb_init(SBuf* s) {
    s->cap = 256;
    s->buf = (char*)xmalloc(s->cap);
    s->len = 0;
    s->buf[0] = '\0';
}

static void sb_ensure(SBuf* s, size_t extra) {
    while (s->len + extra + 1 > s->cap) {
        s->cap *= 2;
        s->buf  = (char*)xrealloc(s->buf, s->cap);
    }
}

static void sb_push(SBuf* s, const char* data, size_t n) {
    sb_ensure(s, n);
    memcpy(s->buf + s->len, data, n);
    s->len += n;
    s->buf[s->len] = '\0';
}

static void sb_cstr(SBuf* s, const char* str) {
    sb_push(s, str, strlen(str));
}

static void sb_char(SBuf* s, char c) {
    sb_push(s, &c, 1);
}

static void sb_indent(SBuf* s, int depth) {
    for (int i = 0; i < depth; i++) sb_cstr(s, "  ");
}

static void write_string_escaped(SBuf* s, const char* str) {
    sb_char(s, '"');
    for (const char* p = str; *p; ) {
        unsigned char c = (unsigned char)*p;
        if      (c == '"')  { sb_cstr(s, "\\\""); p++; }
        else if (c == '\\') { sb_cstr(s, "\\\\"); p++; }
        else if (c == '\n') { sb_cstr(s, "\\n");  p++; }
        else if (c == '\r') { sb_cstr(s, "\\r");  p++; }
        else if (c == '\t') { sb_cstr(s, "\\t");  p++; }
        else if (c == '\b') { sb_cstr(s, "\\b");  p++; }
        else if (c == '\f') { sb_cstr(s, "\\f");  p++; }
        else if (c < 0x20) {
            /* Control character — encode as \uXXXX */
            char tmp[8];
            snprintf(tmp, sizeof(tmp), "\\u%04x", c);
            sb_cstr(s, tmp);
            p++;
        } else {
            /* Raw UTF-8 — copy multi-byte sequence intact */
            int bytes = 1;
            if      (c >= 0xF0) bytes = 4;
            else if (c >= 0xE0) bytes = 3;
            else if (c >= 0xC0) bytes = 2;
            sb_push(s, p, (size_t)bytes);
            p += bytes;
        }
    }
    sb_char(s, '"');
}

static void serialize(SBuf* s, JsonNode* n, int indent) {
    if (!n) { sb_cstr(s, "null"); return; }

    switch (n->type) {
        case JSON_NULL:
            sb_cstr(s, "null");
            break;

        case JSON_BOOL:
            sb_cstr(s, n->b ? "true" : "false");
            break;

        case JSON_NUMBER: {
            char tmp[64];
            /* Use integer representation if the value is a whole number
               and fits in a 64-bit int (avoids "1.0" for ids, counts, etc.) */
            double iv;
            if (modf(n->n, &iv) == 0.0 &&
                n->n >= -9.007199254740992e15 &&
                n->n <=  9.007199254740992e15) {
                snprintf(tmp, sizeof(tmp), "%.0f", n->n);
            } else {
                snprintf(tmp, sizeof(tmp), "%.10g", n->n);
            }
            sb_cstr(s, tmp);
            break;
        }

        case JSON_STRING:
            write_string_escaped(s, n->s);
            break;

        case JSON_ARRAY:
        case JSON_OBJECT: {
            int is_obj  = (n->type == JSON_OBJECT);
            char open   = is_obj ? '{' : '[';
            char close  = is_obj ? '}' : ']';

            sb_char(s, open);
            for (int i = 0; i < n->arr.count; i++) {
                if (i) sb_char(s, ',');
                sb_char(s, '\n');
                sb_indent(s, indent + 1);
                JsonNode* c = n->arr.items[i];
                if (is_obj && c && c->key) {
                    write_string_escaped(s, c->key);
                    sb_cstr(s, ": ");
                }
                serialize(s, c, indent + 1);
            }
            if (n->arr.count) {
                sb_char(s, '\n');
                sb_indent(s, indent);
            }
            sb_char(s, close);
            break;
        }
    }
}

char* json_to_string(JsonNode* root) {
    SBuf s;
    sb_init(&s);
    serialize(&s, root, 0);
    return s.buf; /* caller must free */
}

/* ── mkdir -p (portable) ────────────────────────────────────────────── */
static void mkdir_for_file(const char* path) {
    char dir[4096];
    strncpy(dir, path, sizeof(dir) - 1);
    dir[sizeof(dir) - 1] = '\0';

    /* Find last separator */
    char* last_sep = NULL;
    for (char* p = dir; *p; p++)
        if (*p == '/' || *p == '\\') last_sep = p;

    if (!last_sep) return; /* no directory component */
    *last_sep = '\0';

#ifdef PLATFORM_WINDOWS
    /* Recursively create directories on Windows */
    /* Simple approach: walk forward creating each component */
    for (char* p = dir + 1; *p; p++) {
        if (*p == '/' || *p == '\\') {
            char saved = *p;
            *p = '\0';
            _mkdir(dir);
            *p = saved;
        }
    }
    _mkdir(dir);
#else
    {
        /* Use shell for POSIX; safer than reimplementing mkdir -p */
        char cmd[4096 + 16];
        snprintf(cmd, sizeof(cmd), "mkdir -p \"%s\" 2>/dev/null", dir);
        (void)system(cmd);
    }
#endif
}

void json_write_file(JsonNode* root, const char* path) {
    mkdir_for_file(path);
    char* str = json_to_string(root);
    FILE* f = fopen(path, "w");
    if (f) {
        fputs(str, f);
        fclose(f);
    } else {
        fprintf(stderr, "json_write_file: cannot open '%s' for writing\n", path);
    }
    free(str);
}
