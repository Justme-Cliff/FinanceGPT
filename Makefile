# FinanceGPT -- C Edition Makefile
# Targets: all (default), clean, info

CC      ?= gcc
TARGET  := financegpt
SRCDIR  := csrc
SRCS    := $(wildcard $(SRCDIR)/*.c)
OBJS    := $(SRCS:.c=.o)

# ── Base flags ──────────────────────────────────────────────────────
CFLAGS  := -O3 -march=native -std=c11 -Wall -Wextra -Wno-unused-parameter -Wno-unused-result
CFLAGS  += -I$(SRCDIR)
LDFLAGS :=
LDLIBS  := -lm

# ── Platform detection ───────────────────────────────────────────────
UNAME_S := $(shell uname -s 2>/dev/null || echo Windows)
ifeq ($(UNAME_S),Windows)
    TARGET  := financegpt.exe
    LDLIBS  += -lws2_32
    CFLAGS  += -DPLATFORM_WINDOWS
    RM      := del /Q
    FIXPATH  = $(subst /,\,$1)
else ifeq ($(UNAME_S),Darwin)
    CFLAGS  += -DPLATFORM_POSIX
    LDLIBS  += -lpthread
    RM      := rm -f
    FIXPATH  = $1
else
    CFLAGS  += -DPLATFORM_POSIX
    LDLIBS  += -lpthread
    RM      := rm -f
    FIXPATH  = $1
endif

# ── AVX2 detection ───────────────────────────────────────────────────
AVX2_CHECK := $(shell $(CC) -mavx2 -dM -E - < /dev/null 2>/dev/null | grep -c __AVX2__)
ifeq ($(AVX2_CHECK),1)
    CFLAGS  += -mavx2 -mfma
    $(info AVX2: enabled)
else
    $(info AVX2: not available -- using scalar fallback)
endif

# ── OpenMP detection ─────────────────────────────────────────────────
OMP_CHECK := $(shell echo 'int main(){return 0;}' | $(CC) -fopenmp -x c - -o _omp_test.exe 2>/dev/null && echo yes; rm -f _omp_test.exe 2>/dev/null)
ifeq ($(OMP_CHECK),yes)
    CFLAGS  += -fopenmp
    LDFLAGS += -fopenmp
    $(info OpenMP: enabled)
else
    $(info OpenMP: not available)
endif

# ── OpenBLAS detection (best GEMM available) ─────────────────────────
BLAS_CHECK := $(shell echo 'int main(){return 0;}' | $(CC) -lopenblas -x c - -o /dev/null 2>/dev/null && echo yes)
ifeq ($(BLAS_CHECK),yes)
    CFLAGS  += -DHAVE_OPENBLAS
    LDLIBS  += -lopenblas
    $(info OpenBLAS: enabled)
else
    $(info OpenBLAS: not found -- using built-in GEMM)
endif

# ── Targets ──────────────────────────────────────────────────────────
.PHONY: all clean info

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CC) $(LDFLAGS) -o $@ $^ $(LDLIBS)
	@echo ""
	@echo "  Build complete: ./$(TARGET)"
	@echo "  Run:  ./$(TARGET) /train   (train model)"
	@echo "        ./$(TARGET) /chat    (interactive chat)"
	@echo "        ./$(TARGET) /info    (show stats)"
	@echo ""

HDRS    := $(wildcard $(SRCDIR)/*.h)

$(SRCDIR)/%.o: $(SRCDIR)/%.c $(HDRS)
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	$(RM) $(call FIXPATH,$(SRCDIR)/*.o) 2>/dev/null; true
	$(RM) $(TARGET) $(TARGET).exe 2>/dev/null; true

info:
	@echo "Compiler : $(CC)"
	@echo "CFLAGS   : $(CFLAGS)"
	@echo "Sources  : $(SRCS)"
	@echo "Target   : $(TARGET)"
