#ifndef ARENA_H
#define ARENA_H
#include "compat.h"

/* Fast linear arena allocator.
   Ideal for inference: alloc freely, reset in O(1).
   Not suitable for training (use regular malloc there). */

typedef struct Arena {
    char*  base;
    size_t used;
    size_t cap;
    size_t align; /* default alignment */
} Arena;

/* Create/destroy */
Arena* arena_create(size_t cap);
void   arena_destroy(Arena* a);

/* Allocate (zero-initialized) */
void*  arena_alloc(Arena* a, size_t n);
/* Allocate aligned */
void*  arena_alloc_aligned(Arena* a, size_t n, size_t align);
/* Reset (keeps memory, resets pointer) */
void   arena_reset(Arena* a);
/* Save/restore a position */
size_t arena_save(const Arena* a);
void   arena_restore(Arena* a, size_t pos);

#endif /* ARENA_H */
