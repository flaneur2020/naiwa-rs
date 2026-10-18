// Does interleaving one expert's three stacks (gate|up|down) beat the current layout
// (three far-apart regions per expert)? Same bytes, same threads, only the address
// pattern differs - which is exactly the hypothesis behind M7 "repack".
//
// Build/run: ./run.sh
#define _GNU_SOURCE
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define EXPERTS 512
#define PER_PART (2u * 1024u * 1024u / 3u)   // ~0.7 MB per projection, 2.1 MB per expert
#define THREADS 6
#define TRIALS 200

static unsigned char *split[3];   // current: three stacks of EXPERTS*PER_PART
static unsigned char *packed;     // proposed: one stack of EXPERTS*(3*PER_PART)

static double now_s(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec + t.tv_nsec * 1e-9;
}

struct args { int id; unsigned long long sink; };
static int *order;                // shared list of experts to touch

// Streaming read like the gemv: one byte per cache line, over one range.
static inline unsigned long long touch(const unsigned char *p, size_t n) {
    unsigned long long s = 0;
    for (size_t i = 0; i < n; i += 64) s ^= p[i];
    return s;
}

static void *worker(void *v) {
    struct args *a = v;
    unsigned long long s = 0;
    for (int t = a->id; t < TRIALS; t += THREADS) {
        int e = order[t];
        for (int k = 0; k < 3; k++) s ^= touch(split[k] + (size_t)e * PER_PART, PER_PART);
    }
    a->sink = s;
    return NULL;
}

static void *worker_packed(void *v) {
    struct args *a = v;
    unsigned long long s = 0;
    for (int t = a->id; t < TRIALS; t += THREADS) {
        int e = order[t];
        s ^= touch(packed + (size_t)e * 3 * PER_PART, 3u * PER_PART);
    }
    a->sink = s;
    return NULL;
}

int main(void) {
    size_t one = (size_t)EXPERTS * PER_PART;
    for (int k = 0; k < 3; k++) {
        if (posix_memalign((void **)&split[k], 2u << 20, one)) return 1;
        memset(split[k], k + 1, one);
    }
    if (posix_memalign((void **)&packed, 2u << 20, one * 3)) return 1;
    memset(packed, 7, one * 3);
    order = malloc(sizeof(int) * TRIALS);
    srand(1);
    for (int i = 0; i < TRIALS; i++) order[i] = rand() % EXPERTS;   // random experts, like routing

    pthread_t th[THREADS];
    struct args a[THREADS];
    double t0, t1;
    unsigned long long sink = 0;

    t0 = now_s();
    for (int i = 0; i < THREADS; i++) { a[i].id = i; pthread_create(&th[i], 0, worker, &a[i]); }
    for (int i = 0; i < THREADS; i++) { pthread_join(th[i], 0); sink ^= a[i].sink; }
    t1 = now_s();
    double split_s = t1 - t0;

    t0 = now_s();
    for (int i = 0; i < THREADS; i++) { a[i].id = i; pthread_create(&th[i], 0, worker_packed, &a[i]); }
    for (int i = 0; i < THREADS; i++) { pthread_join(th[i], 0); sink ^= a[i].sink; }
    t1 = now_s();
    double packed_s = t1 - t0;

    double bytes = (double)TRIALS * 3.0 * PER_PART;
    printf("experts touched %d, bytes/expert %.1f MiB, threads %d\n", TRIALS, 3.0 * PER_PART / 1048576.0, THREADS);
    printf("split  3 regions: %.1f ms  %.1f GB/s\n", split_s * 1e3, bytes / split_s / 1e9);
    printf("packed 1 region : %.1f ms  %.1f GB/s\n", packed_s * 1e3, bytes / packed_s / 1e9);
    printf("packed is %.2fx the split rate\n", split_s / packed_s);
    printf("(sink %llu)\n", sink);
    return 0;
}
