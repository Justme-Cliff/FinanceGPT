#include "arena.h"

Arena* arena_create(size_t cap) {
    Arena* a = (Arena*)xmalloc(sizeof(Arena));
    a->base  = (char*)xmalloc_aligned(cap, 64);
    a->used  = 0;
    a->cap   = cap;
    a->align = 64;
    return a;
}

void arena_destroy(Arena* a) {
    if (!a) return;
    xfree_aligned(a->base);
    free(a);
}

void* arena_alloc(Arena* a, size_t n) {
    return arena_alloc_aligned(a, n, a->align);
}

void* arena_alloc_aligned(Arena* a, size_t n, size_t align) {
    /* round up to alignment */
    size_t start = (a->used + align - 1) & ~(align - 1);
    size_t end   = start + n;
    if (UNLIKELY(end > a->cap)) {
        fprintf(stderr, "Arena OOM: need %zu, have %zu\n", end, a->cap);
        exit(1);
    }
    a->used = end;
    memset(a->base + start, 0, n);
    return a->base + start;
}

void arena_reset(Arena* a) {
    a->used = 0;
}

size_t arena_save(const Arena* a) {
    return a->used;
}

void arena_restore(Arena* a, size_t pos) {
    a->used = pos;
}
