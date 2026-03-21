#include "csv.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <ctype.h>

/* ── String utilities ──────────────────────────────────────────────── */

/* Trim leading and trailing ASCII whitespace in-place.
   Returns pointer into s (not a new allocation). */
static char* trim(char* s) {
    /* Leading whitespace */
    while (*s && isspace((unsigned char)*s)) s++;
    if (*s == '\0') return s;
    /* Trailing whitespace */
    char* e = s + strlen(s) - 1;
    while (e > s && isspace((unsigned char)*e)) { *e = '\0'; e--; }
    return s;
}

/* ── Line parser ───────────────────────────────────────────────────── */

/*
 * Parse one CSV line into an array of field strings.
 * Handles:
 *   - Quoted fields (double-quote delimited).
 *   - Embedded commas inside quoted fields.
 *   - Escaped double-quotes ("").
 *   - Un-quoted fields.
 * The returned array and each string in it are heap-allocated.
 * *n_cols is set to the number of fields found.
 * Caller is responsible for freeing each element and then the array.
 */
static char** parse_csv_line(const char* line, int* n_cols) {
    int cap = 16;
    char** fields = (char**)xmalloc(cap * sizeof(char*));
    *n_cols = 0;

    /* Use a dynamic buffer for building each field. */
    size_t buf_cap = 65536;
    char*  buf     = (char*)xmalloc(buf_cap);

    const char* p = line;
    /* Process one field per loop iteration; stop when we reach end-of-string. */
    for (;;) {
        /* Grow the fields array if needed */
        if (*n_cols >= cap) {
            cap *= 2;
            fields = (char**)xrealloc(fields, cap * sizeof(char*));
        }

        size_t len = 0;

        if (*p == '"') {
            /* Quoted field */
            p++; /* skip opening '"' */
            while (*p) {
                if (*p == '"') {
                    if (*(p + 1) == '"') {
                        /* Escaped double-quote "" -> single " */
                        if (len + 1 >= buf_cap) { buf_cap *= 2; buf = (char*)xrealloc(buf, buf_cap); }
                        buf[len++] = '"';
                        p += 2;
                    } else {
                        /* Closing quote */
                        p++;
                        break;
                    }
                } else {
                    if (len + 1 >= buf_cap) { buf_cap *= 2; buf = (char*)xrealloc(buf, buf_cap); }
                    buf[len++] = *p++;
                }
            }
            /* Skip the delimiter ',' after closing quote, if present */
            if (*p == ',') p++;
        } else {
            /* Un-quoted field — read until comma or end */
            while (*p && *p != ',') {
                if (len + 1 >= buf_cap) { buf_cap *= 2; buf = (char*)xrealloc(buf, buf_cap); }
                buf[len++] = *p++;
            }
            if (*p == ',') p++;
        }

        buf[len] = '\0';
        /* Trim whitespace from un-quoted fields; quoted fields are stored verbatim. */
        fields[(*n_cols)++] = xstrdup(trim(buf));

        /* If we have consumed the entire line, stop. */
        if (*p == '\0') break;
    }

    free(buf);
    return fields;
}

/* ── Public API ────────────────────────────────────────────────────── */

CsvTable* csv_load(const char* path) {
    FILE* f = fopen(path, "rb");
    if (!f) return NULL;

    CsvTable* t    = (CsvTable*)xcalloc(1, sizeof(CsvTable));
    t->cap_rows    = 256;
    t->rows        = (char***)xmalloc(t->cap_rows * sizeof(char**));
    t->n_rows      = 0;
    t->headers     = NULL;
    t->n_cols      = 0;

    /* Large line buffer to accommodate long CSV answer fields */
    size_t line_cap = 1 << 18; /* 256 KB */
    char* line = (char*)xmalloc(line_cap);

    int first_line = 1;
    while (fgets(line, (int)line_cap, f)) {
        /* Strip trailing CR/LF */
        size_t ln = strlen(line);
        while (ln > 0 && (line[ln - 1] == '\n' || line[ln - 1] == '\r'))
            line[--ln] = '\0';

        /* Skip blank lines */
        if (ln == 0) continue;

        int nc;
        char** fields = parse_csv_line(line, &nc);

        if (first_line) {
            /* Header row */
            t->headers = fields;
            t->n_cols  = nc;
            first_line = 0;
        } else {
            /* Data row */
            if (t->n_rows >= t->cap_rows) {
                t->cap_rows *= 2;
                t->rows = (char***)xrealloc(t->rows, t->cap_rows * sizeof(char**));
            }

            /* Normalise to exactly n_cols fields */
            char** row = (char**)xmalloc(t->n_cols * sizeof(char*));
            for (int c = 0; c < t->n_cols; c++) {
                if (c < nc) {
                    row[c]    = fields[c]; /* transfer ownership */
                    fields[c] = NULL;
                } else {
                    row[c] = xstrdup(""); /* pad short rows */
                }
            }
            /* Free any extra fields beyond n_cols */
            for (int c = 0; c < nc; c++) {
                if (fields[c]) free(fields[c]);
            }
            free(fields);

            t->rows[t->n_rows++] = row;
        }
    }

    free(line);
    fclose(f);

    /* Handle degenerate case: file was empty or had only whitespace */
    if (first_line) {
        /* No header was parsed — return an empty table */
        t->headers = NULL;
        t->n_cols  = 0;
    }

    return t;
}

void csv_free(CsvTable* t) {
    if (!t) return;

    /* Free header strings */
    if (t->headers) {
        for (int c = 0; c < t->n_cols; c++) free(t->headers[c]);
        free(t->headers);
    }

    /* Free all row strings and row arrays */
    for (int r = 0; r < t->n_rows; r++) {
        if (t->rows[r]) {
            for (int c = 0; c < t->n_cols; c++) free(t->rows[r][c]);
            free(t->rows[r]);
        }
    }
    free(t->rows);
    free(t);
}

int csv_col(CsvTable* t, const char* name) {
    if (!t || !name || !t->headers) return -1;
    size_t nn = strlen(name);
    for (int c = 0; c < t->n_cols; c++) {
        const char* h  = t->headers[c];
        size_t      hn = strlen(h);
        if (hn != nn) continue;
        int match = 1;
        for (size_t i = 0; i < hn; i++) {
            if (tolower((unsigned char)h[i]) != tolower((unsigned char)name[i])) {
                match = 0;
                break;
            }
        }
        if (match) return c;
    }
    return -1;
}

const char* csv_get(CsvTable* t, int row, int col) {
    if (!t)                       return "";
    if (row < 0 || row >= t->n_rows) return "";
    if (col < 0 || col >= t->n_cols) return "";
    const char* v = t->rows[row][col];
    return v ? v : "";
}
