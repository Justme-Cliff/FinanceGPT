/*
 * FinanceGPT — C edition entry point
 * Usage:
 *   financegpt /train              train on all CSVs
 *   financegpt /train path/to.csv  fine-tune on one CSV
 *   financegpt /chat               interactive chat
 *   financegpt /info               show model stats
 */
#include "compat.h"
#include "chat.h"
#include "trainer.h"
#include "model.h"
#include "tokenizer.h"
#include "knowledge_base.h"
#include "config.h"
#include <stdio.h>
#include <string.h>

static void print_info(void) {
    FILE* f = fopen(CHECKPOINT_PATH, "rb");
    if (!f) {
        printf("  No model found at %s\n", CHECKPOINT_PATH);
        return;
    }
    fclose(f);

    Model* m = model_load(CHECKPOINT_PATH);
    if (!m) return;

    printf("\n  FinanceGPT -- Model Info\n");
    printf("  Parameters   : %zu\n",  model_n_params(m));
    printf("  d_model      : %d\n",   m->cfg.d_model);
    printf("  n_heads      : %d\n",   m->cfg.n_heads);
    printf("  n_layers     : %d\n",   m->cfg.n_layers);
    printf("  d_ff         : %d\n",   m->cfg.d_ff);
    printf("  max_seq_len  : %d\n",   m->cfg.max_seq_len);
    printf("  vocab_size   : %d\n",   m->cfg.vocab_size);

    Tokenizer* tok = tok_load(TOKENIZER_PATH);
    if (tok) {
        printf("  Tokenizer    : %d tokens\n", tok->vocab_size);
        tok_free(tok);
    }

    KnowledgeBase* kb = kb_create(DATA_DIR);
    if (kb) {
        printf("  KB pairs     : %d\n", kb_size(kb));
        kb_free(kb);
    }

    model_free(m);
    printf("\n");
}

int main(int argc, char** argv) {
#ifdef _OPENMP
    int ncpu = omp_get_max_threads();
    omp_set_num_threads(ncpu);
    printf("  OpenMP: %d threads\n", ncpu);
#endif

    if (argc < 2) {
        printf("\nFinanceGPT -- C Edition\n");
        printf("  Usage:\n");
        printf("    financegpt /train              train on all CSVs\n");
        printf("    financegpt /train data/foo.csv fine-tune on one CSV\n");
        printf("    financegpt /chat               interactive chat\n");
        printf("    financegpt /info               model & KB stats\n\n");
        return 0;
    }

    const char* cmd = argv[1];

    if (strcmp(cmd, "/train") == 0) {
        const char* csv_file = (argc >= 3) ? argv[2] : NULL;
        train(csv_file);
    } else if (strcmp(cmd, "/chat") == 0) {
        chat_main();
    } else if (strcmp(cmd, "/info") == 0) {
        print_info();
    } else {
        fprintf(stderr, "Unknown command: %s\n", cmd);
        fprintf(stderr, "Use /train, /chat, or /info\n");
        return 1;
    }

    return 0;
}
