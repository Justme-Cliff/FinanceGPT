#ifndef JSON_H
#define JSON_H
#include "compat.h"

typedef enum {
    JSON_NULL,
    JSON_BOOL,
    JSON_NUMBER,
    JSON_STRING,
    JSON_ARRAY,
    JSON_OBJECT
} JsonType;

typedef struct JsonNode JsonNode;
struct JsonNode {
    JsonType     type;
    char*        key;      /* for object members */
    union {
        int      b;        /* JSON_BOOL */
        double   n;        /* JSON_NUMBER */
        char*    s;        /* JSON_STRING */
        struct {
            JsonNode** items;
            int        count;
            int        cap;
        } arr;             /* JSON_ARRAY + JSON_OBJECT */
    };
    JsonNode* next;        /* linked list of siblings */
};

/* Parse JSON string, returns root node or NULL on error */
JsonNode*   json_parse      (const char* text);
/* Parse JSON file */
JsonNode*   json_parse_file (const char* path);
/* Free all memory */
void        json_free       (JsonNode* node);

/* Access helpers */
JsonNode*   json_get        (JsonNode* obj,  const char* key);
JsonNode*   json_get_index  (JsonNode* arr,  int idx);
const char* json_str        (JsonNode* node, const char* def);
double      json_num        (JsonNode* node, double def);
int         json_bool       (JsonNode* node, int def);
int         json_len        (JsonNode* node);

/* Build helpers */
JsonNode*   json_new_str    (const char* key, const char* val);
JsonNode*   json_new_num    (const char* key, double val);
JsonNode*   json_new_bool   (const char* key, int val);
JsonNode*   json_new_obj    (const char* key);
JsonNode*   json_new_arr    (const char* key);
void        json_append     (JsonNode* parent, JsonNode* child);
void        json_set_str    (JsonNode* obj, const char* key, const char* val);

/* Serialize to file */
void        json_write_file (JsonNode* root, const char* path);
/* Serialize to string (caller frees) */
char*       json_to_string  (JsonNode* root);

#endif /* JSON_H */
