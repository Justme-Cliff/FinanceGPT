#ifndef TRAINER_H
#define TRAINER_H
#include "compat.h"
#include "config.h"
#include "model.h"
#include "tokenizer.h"

/* ── Dataset ────────────────────────────────────────────────────── */
typedef struct {
    int*   tokens;     /* flat token array */
    size_t n_tokens;
    int    block_size;
    int    stride;
    size_t n_samples;
} Dataset;

Dataset* dataset_create(int* tokens, size_t n_tokens, int block_size, int stride);
void     dataset_free  (Dataset* d);
void     dataset_get   (const Dataset* d, size_t idx, int* x_out, int* y_out);

/* ── Training history ───────────────────────────────────────────── */
typedef struct {
    float train_loss;
    float val_loss;
    int   epoch;
    int   step;
    float train_ppl;
    float val_ppl;
} EpochRecord;

typedef struct {
    float*       losses;      /* per step */
    float*       lrs;         /* per step */
    float*       grad_norms;  /* per step */
    int*         steps;
    int          n_steps;
    int          cap_steps;
    EpochRecord* val_epochs;
    int          n_epochs;
    int          cap_epochs;
} TrainHistory;

TrainHistory* history_create(void);
void          history_free  (TrainHistory* h);
void          history_add_step(TrainHistory* h, int step, float loss, float lr, float gnorm);
void          history_add_epoch(TrainHistory* h, EpochRecord rec);

/* ── AdamW optimizer ────────────────────────────────────────────── */
void optimizer_init     (Model* m);  /* allocate m,v moment buffers */
void optimizer_step     (Model* m, float lr, float beta1, float beta2,
                         float eps, float weight_decay, int step);
void optimizer_zero_grad(Model* m);

/* ── Learning rate schedule (SGDR: cosine annealing with warm restarts) */
float lr_schedule(int step, int total_steps, float lr, float min_lr, int warmup);

/* ── SVG plot generation ────────────────────────────────────────── */
void generate_plots(const TrainHistory* h, const char* plots_dir);

/* ── Training entry point ───────────────────────────────────────── */
/* csv_file: NULL = train on all CSVs, else path to one CSV */
void train(const char* csv_file);

/* Evaluate on validation dataset, return avg loss */
float evaluate(Model* m, Tokenizer* tok, Dataset* val_ds,
               int block_size, float label_smoothing);

#endif /* TRAINER_H */
