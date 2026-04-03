/****************************************************************************
 *
 * omp-k-means.c -- Implementazione OpenMP dell'algoritmo K-Means clustering.
 *
 * Autore: Simone Brunelli
 * Matricola: 0001021672
 *
 * Basato sul codice seriale di Moreno Marzolla
 *
 * Compilare con:
 *
 *      gcc -std=c99 -Wall -Wpedantic -fopenmp omp-k-means.c -o omp-k-means
 *
 * Eseguire con:
 *
 *      OMP_NUM_THREADS=4 ./omp-k-means K input_file output_file
 *
 * Esempio:
 *
 *      OMP_NUM_THREADS=4 ./omp-k-means 5 demo.in demo.out
 *
 * --------------------------------------------------------------------------
 *
 * Strategia di parallelizzazione:
 *
 * Le due funzioni computazionalmente costose sono classify() e
 * update_centroids(). Entrambe operano su tutti gli N punti e vengono
 * parallelizzate con OpenMP.
 *
 * classify(): ogni iterazione del ciclo su i e' indipendente dalle altre,
 * per cui si parallelizza con #pragma omp parallel for. L'aggiornamento di
 * counts[] e' protetto con una reduction su array.
 *
 * update_centroids(): l'accumulazione in new_centroids[] presenta una
 * race condition perché piú thread possono scrivere sullo stesso
 * centroide contemporaneamente. Si risolve allocando un array privato per
 * ogni thread (local_nc e local_cnt), eseguendo le somme localmente, e
 * infine sommando i contributi dei vari thread in una sezione critica.
 *
 ****************************************************************************/

#if _XOPEN_SOURCE < 600
#define _XOPEN_SOURCE 600
#endif

#include <assert.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "hpc.h"

int n_dims;
int n_points;
int n_clusters;

float* data;
float* centroids;
float* new_centroids;
int* counts;
int* cluster_of;

void* safe_malloc(size_t size) {
  void* result = malloc(size);
  assert(result != NULL);
  return result;
}

void* safe_calloc(size_t nmemb, size_t size) {
  void* result = calloc(nmemb, size);
  assert(result != NULL);
  return result;
}

/******************************************************************************
 **
 ** Utility functions that operate on arrays of `n_dims` elements.
 **
 ******************************************************************************/

/* Set all components of vector `p` of size `n_dims` equal to zero. */
void vzero(float* p) {
  for (int d = 0; d < n_dims; d++) p[d] = 0.0f;
}

/* Add vector `p1` to vector `p2`; store result in `p1`. Both vectors
   have size `n_dims`. */
void vadd(float* p1, const float* p2) {
  for (int d = 0; d < n_dims; d++) p1[d] += p2[d];
}

/* Multiply each element of vector `p` of size `n_dims` by `v`. */
void vmul(float* p, float v) {
  for (int d = 0; d < n_dims; d++) p[d] *= v;
}

/* Copy `p2` into `p1`. */
void vcopy(float* p1, const float* p2) {
  for (int d = 0; d < n_dims; d++) p1[d] = p2[d];
}

/* Compute the Euclidean squared distance of `p1` and `p2`. */
float sqdist(float* p1, float* p2) {
  float result = 0.0;
  for (int d = 0; d < n_dims; d++) {
    result += (p1[d] - p2[d]) * (p1[d] - p2[d]);
  }
  return result;
}

/******************************************************************************
 **
 ** K-Means algorithm begins here.
 **
 ******************************************************************************/

/* This function can be used to access the arrays (actually, matrices)
   `data`, `centroids` and `new_centroids`. These are all matrices
   with `n_dims` columns. The function returns the linear index of row
   `i` and column `d`. Example: `data[IDX(i, d)]` is equivalent to
   `data[i*n_dims + d]`. */
int IDX(int i, int d) { return i * n_dims + d; }

/* Return a random integer in a..b. This function must not be
   parallelized, since `rand()` is not thread-safe. */
int randab(int a, int b) { return a + rand() % (b - a + 1); }

/* Centroids are initialized by randomly selecting `n_clusters` data
   points. To select `n_clusters` out of `n_data` elements, we use
   Knuths' algorithm as reported in J. Bentley, "Programming Pearls",
   2nd ed., Addison-Wesley, 2000, p. 126.

   DO NOT PARALLELIZE THIS FUNCTION: `rand()` is not thread-safe. */
void init_centroids(void) {
  int select = n_clusters;
  int remaining = n_points;
  for (int i = 0; (i < n_points) && (select > 0); i++) {
    if ((rand() % remaining) < select) {
      select--;
      /* Select point `i` as one of the centroids. */
      vcopy(&centroids[IDX(select, 0)], &data[IDX(i, 0)]);
    }
    remaining--;
  }
}

/*
 * classify() -- versione OpenMP.
 *
 * Il ciclo esterno su i e' parallelizzato: ogni thread gestisce un
 * sottoinsieme di punti. La variabile `nearest` e' privata (dichiarata
 * dentro il ciclo). L'aggiornamento di counts[] e cluster_of[] non
 * genera race condition perche' ogni iterazione scrive su indici
 * diversi di cluster_of[]; l'aggiornamento di counts[] e' protetto
 * con una reduction su array (OpenMP 4.5+).
 */
void classify(void) {
  /* Azzera i contatori. */
#pragma omp parallel for schedule(static)
  for (int j = 0; j < n_clusters; j++) counts[j] = 0;

#pragma omp parallel for schedule(static) reduction(+ : counts[ : n_clusters])
  for (int i = 0; i < n_points; i++) {
    int nearest = 0;
    float mindist = sqdist(&data[IDX(i, 0)], &centroids[IDX(0, 0)]);
    for (int j = 1; j < n_clusters; j++) {
      const float dist = sqdist(&data[IDX(i, 0)], &centroids[IDX(j, 0)]);
      if (dist < mindist) {
        mindist = dist;
        nearest = j;
      }
    }
    cluster_of[i] = nearest;
    counts[nearest]++;
  }
}

/*
 * update_centroids() -- versione OpenMP.
 *
 * L'accumulazione dei punti nei nuovi centroidi presenta una race
 * condition perche' piu' thread possono accedere allo stesso centroide.
 * La soluzione adottata e' quella degli array locali per thread:
 *
 *  1. Ogni thread alloca local_nc (n_clusters * n_dims float) e
 *     local_cnt (n_clusters int), inizializzati a zero.
 *  2. Ogni thread accumula i propri punti in locale.
 *  3. In una sezione critica, ogni thread somma il proprio contributo
 *     a new_centroids[] e counts[].
 *
 * Questo schema evita l'uso di atomic o lock per ogni singolo punto,
 * riducendo la contesa.
 */
float update_centroids(void) {
  /* Azzera new_centroids e counts. */
  memset(new_centroids, 0, n_clusters * n_dims * sizeof(*new_centroids));
  memset(counts, 0, n_clusters * sizeof(*counts));

#pragma omp parallel
  {
    /* Array locali per thread. */
    float* local_nc = (float*)safe_calloc(n_clusters * n_dims, sizeof(float));
    int* local_cnt = (int*)safe_calloc(n_clusters, sizeof(int));

    /* Accumulo locale: nessuna race condition. */
#pragma omp for schedule(static)
    for (int i = 0; i < n_points; i++) {
      const int c = cluster_of[i];
      local_cnt[c]++;
      for (int d = 0; d < n_dims; d++) local_nc[IDX(c, d)] += data[IDX(i, d)];
    }

    /* Merge nella struttura condivisa: sezione critica per evitare
       race condition tra thread diversi. */
#pragma omp critical
    {
      for (int j = 0; j < n_clusters; j++) {
        counts[j] += local_cnt[j];
        for (int d = 0; d < n_dims; d++)
          new_centroids[IDX(j, d)] += local_nc[IDX(j, d)];
      }
    }

    free(local_nc);
    free(local_cnt);
  } /* fine regione parallela */

  /* Calcola i nuovi centroidi e il massimo spostamento. */
  float maxsqshift = 0.0f;
  for (int j = 0; j < n_clusters; j++) {
    if (counts[j] == 0) {
      /* Cluster vuoto: mantieni il vecchio centroide. */
      vcopy(&new_centroids[IDX(j, 0)], &centroids[IDX(j, 0)]);
    } else {
      vmul(&new_centroids[IDX(j, 0)], 1.0f / counts[j]);
    }
    const float sqshift =
        sqdist(&centroids[IDX(j, 0)], &new_centroids[IDX(j, 0)]);
    if (sqshift > maxsqshift) maxsqshift = sqshift;
    vcopy(&centroids[IDX(j, 0)], &new_centroids[IDX(j, 0)]);
  }

  return maxsqshift;
}

/******************************************************************************
 ** I/O -- Non parallelizzare.
 ******************************************************************************/

void read_input(FILE* f) {
  const size_t BUFLEN = 1024;
  char buffer[BUFLEN];

  char* i_dont_care = fgets(buffer, BUFLEN, f);
  (void)i_dont_care;
  n_dims = -1;
  char *start, *end = buffer;
  do {
    start = end;
    strtof(start, &end);
    n_dims++;
  } while (end != start);

  assert(n_dims > 0);

  rewind(f);
  int n_items = 0;
  float dummy;
  while (1 == fscanf(f, "%f", &dummy)) n_items++;

  n_points = n_items / n_dims;
  assert(n_items % n_dims == 0);

  data = (float*)safe_malloc(n_points * n_dims * sizeof(*data));

  rewind(f);
  for (int i = 0; i < n_points; i++) {
    for (int d = 0; d < n_dims; d++) {
      const int nread = fscanf(f, "%f", &data[IDX(i, d)]);
      assert(nread == 1);
    }
  }
}

void save_results(FILE* f) {
  fprintf(f, "# Centroids:\n#\n");
  for (int j = 0; j < n_clusters; j++) {
    fprintf(f, "# %3d :", j);
    for (int d = 0; d < n_dims; d++) fprintf(f, " %f", centroids[IDX(j, d)]);
    fprintf(f, "\n");
  }
  fprintf(f, "#\n");
  for (int i = 0; i < n_points; i++) {
    for (int d = 0; d < n_dims; d++) fprintf(f, "%f ", data[IDX(i, d)]);
    fprintf(f, "%d\n", cluster_of[i]);
  }
}

/******************************************************************************
 ** Main
 ******************************************************************************/
int main(int argc, char* argv[]) {
  FILE *inputf, *outputf;
  const int MAXITER = 100;
  const float TOL = 1e-5;

  if (argc != 4) {
    fprintf(stderr, "Usage: %s K input_file output_file\n", argv[0]);
    return EXIT_FAILURE;
  }

  srand(123);

  n_clusters = atoi(argv[1]);

  if ((inputf = fopen(argv[2], "r")) == NULL) {
    fprintf(stderr, "FATAL: can not open input file \"%s\"\n", argv[2]);
    return EXIT_FAILURE;
  }

  read_input(inputf);
  fclose(inputf);

  assert(n_clusters < n_points);

  if ((outputf = fopen(argv[3], "w")) == NULL) {
    fprintf(stderr, "FATAL: can not create output file \"%s\"\n", argv[3]);
    return EXIT_FAILURE;
  }

  fprintf(outputf, "# Data points: %d\n", n_points);
  fprintf(outputf, "# Dimensions: %d\n", n_dims);
  fprintf(outputf, "# Clusters: %d\n", n_clusters);

  printf("\nInput file....... %s\n", argv[2]);
  printf("Output file...... %s\n", argv[3]);
  printf("Data points (N).. %d\n", n_points);
  printf("Dimensions (D)... %d\n", n_dims);
  printf("Clusters (K)..... %d\n", n_clusters);
  printf("OMP threads...... %d\n\n", omp_get_max_threads());

  centroids = (float*)safe_malloc(n_clusters * n_dims * sizeof(*centroids));
  new_centroids =
      (float*)safe_malloc(n_clusters * n_dims * sizeof(*new_centroids));
  cluster_of = (int*)safe_malloc(n_points * sizeof(*cluster_of));
  counts = (int*)safe_malloc(n_clusters * sizeof(*counts));

  init_centroids();

  printf("Main loop starts\n\n");

  float maxsqshift;
  int iter = 0;
  const double tstart = hpc_gettime();
  do {
    classify();
    maxsqshift = update_centroids();
    printf("Iteration %3d, maxsqshift = %f\n", iter, maxsqshift);
    iter++;
  } while ((maxsqshift > TOL * TOL) && (iter < MAXITER));
  const double elapsed = hpc_gettime() - tstart;

  printf("\nMain loop completed\n");
  printf("Elapsed time %.3f ms\n\n", elapsed * 1000.0);
  save_results(outputf);
  fclose(outputf);

  free(data);
  free(centroids);
  free(new_centroids);
  free(cluster_of);
  free(counts);

  return EXIT_SUCCESS;
}