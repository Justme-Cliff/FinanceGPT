#ifndef CSV_H
#define CSV_H
#include "compat.h"

typedef struct {
    char**  headers;   /* headers[c]    = column name string     */
    int     n_cols;    /* number of columns                      */
    char*** rows;      /* rows[r][c]    = field string           */
    int     n_rows;    /* number of data rows (excluding header) */
    int     cap_rows;  /* allocated row capacity                 */
} CsvTable;

/* Load a CSV file.  Returns NULL if the file cannot be opened. */
CsvTable*   csv_load (const char* path);

/* Free all memory owned by the table. */
void        csv_free (CsvTable* t);

/* Find a column index by name (case-insensitive). Returns -1 if not found. */
int         csv_col  (CsvTable* t, const char* name);

/* Safe field access.  Returns "" for out-of-range indices. */
const char* csv_get  (CsvTable* t, int row, int col);

#endif /* CSV_H */
