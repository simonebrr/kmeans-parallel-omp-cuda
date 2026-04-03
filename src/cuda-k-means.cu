/****************************************************************************
 *
 * cuda-k-means.cu -- Implementazione CUDA dell'algoritmo K-Means clustering.
 *
 * Autore: Simone Brunelli
 * Matricola: 0001021672
 *
 * Basato sul codice seriale di Moreno Marzolla
 * <https://unibo.it/sitoweb/moreno.marzolla/>
 *
 * Compilare con:
 *
 *      nvcc -O2 -std=c++11 cuda-k-means.cu -o cuda-k-means
 *
 * Eseguire con:
 *
 *      ./cuda-k-means K input_file output_file
 *
 * Esempio:
 *
 *      ./cuda-k-means 5 demo.in demo.out
 *
 * --------------------------------------------------------------------------
 *
 * Strategia di parallelizzazione:
 *
 *  1. L'host legge il file di input e alloca i dati.
 *  2. I punti e i centroidi vengono copiati sul device con cudaMemcpy.
 *  3. Ogni iterazione:
 *       a. kernel_classify(): ogni thread CUDA classifica un punto,
 *          calcolando la distanza da tutti i centroidi e scegliendo
 *          il cluster piu' vicino. I centroidi vengono caricati
 *          cooperativamente in shared memory all'inizio di ogni blocco
 *          (se K*D*4 <= 48 KB); altrimenti vengono letti dalla global
 *          memory sfruttando la cache L2.
 *       b. Salvataggio dei centroidi correnti in d_old_centroids.
 *       c. kernel_reset(): azzera i buffer new_centroids e counts.
 *       d. kernel_accumulate(): riduzione per blocco in shared memory.
 *          Ogni blocco accumula i propri punti localmente (nessuna
 *          contesa intra-blocco), poi esegue un unico atomicAdd per
 *          blocco sulla global memory. Se K*D e' troppo grande per la
 *          shared memory, si usa atomicAdd diretto sulla global memory.
 *       e. kernel_divide(): calcola i nuovi centroidi dividendo le somme
 *          per i conteggi; i cluster vuoti restano invariati.
 *       f. kernel_convergence(): confronta d_old_centroids con i nuovi
 *          centroidi in d_centroids; il massimo spostamento viene
 *          ridotto sull'host.
 *  4. I cluster_of vengono copiati sull'host e scritti su file nel
 *     formato compatibile con il codice seriale di Marzolla.
 *
 * Nota sulla shared memory:
 *   La GTX 1070 dispone di 48 KB di shared memory per blocco.
 *   kernel_classify usa K*D*4 byte per i centroidi.
 *   kernel_accumulate usa (K*D + K)*4 byte per l'accumulo locale.
 *   Il kernel sceglie automaticamente la strategia in base a K e D.
 *   Non esistono vincoli fissi su K o D.
 *
 ****************************************************************************/

#include <cuda_runtime.h>
#include <float.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define BLOCK_SIZE 256
#define SHARED_MEM_MAX 49152 /* 48 KB — limite GTX 1070 */

#define CUDA_CHECK(call)                                                 \
  do {                                                                   \
    cudaError_t err = (call);                                            \
    if (err != cudaSuccess) {                                            \
      fprintf(stderr, "CUDA error at %s:%d -- %s\n", __FILE__, __LINE__, \
              cudaGetErrorString(err));                                  \
      exit(EXIT_FAILURE);                                                \
    }                                                                    \
  } while (0)

/**
 * kernel_classify_smem -- shared memory path
 *
 * Ogni blocco carica cooperativamente i centroidi in shared memory,
 * poi ogni thread calcola le distanze leggendo da li'.
 * Usato quando K*D*sizeof(float) <= SHARED_MEM_MAX.
 */
__global__ void kernel_classify_smem(const float* points,
                                     const float* centroids, int* cluster_of,
                                     int N, int K, int D) {
  extern __shared__ float s_centroids[];

  int total = K * D;
  for (int t = threadIdx.x; t < total; t += blockDim.x)
    s_centroids[t] = centroids[t];
  __syncthreads();

  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= N) return;

  float best_dist = FLT_MAX;
  int best_k = 0;

  for (int k = 0; k < K; k++) {
    float dist = 0.0f;
    for (int d = 0; d < D; d++) {
      float diff = points[i * D + d] - s_centroids[k * D + d];
      dist += diff * diff;
    }
    if (dist < best_dist) {
      best_dist = dist;
      best_k = k;
    }
  }
  cluster_of[i] = best_k;
}

/**
 * kernel_classify_gmem -- global memory fallback
 *
 * Usato quando K*D e' troppo grande per la shared memory.
 * I centroidi vengono letti dalla global memory; la cache L2
 * mitiga il costo grazie agli accessi broadcast.
 */
__global__ void kernel_classify_gmem(const float* points,
                                     const float* centroids, int* cluster_of,
                                     int N, int K, int D) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= N) return;

  float best_dist = FLT_MAX;
  int best_k = 0;

  for (int k = 0; k < K; k++) {
    float dist = 0.0f;
    for (int d = 0; d < D; d++) {
      float diff = points[i * D + d] - centroids[k * D + d];
      dist += diff * diff;
    }
    if (dist < best_dist) {
      best_dist = dist;
      best_k = k;
    }
  }
  cluster_of[i] = best_k;
}

/**
 * kernel_reset
 *
 * Azzera i buffer new_centroids e counts prima di ogni aggiornamento.
 */
__global__ void kernel_reset(float* new_centroids, int* counts, int K, int D) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < K * D) new_centroids[idx] = 0.0f;
  if (idx < K) counts[idx] = 0;
}

/**
 * kernel_accumulate_smem -- riduzione per blocco in shared memory
 *
 * Ogni blocco accumula i propri punti in shared memory locale
 * (nessuna contesa intra-blocco), poi esegue un unico atomicAdd
 * per blocco sulla global memory. Il numero di atomic globali scende
 * da N*D a (N/BLOCK_SIZE)*K*D, riducendo drasticamente la contesa.
 * Usato quando (K*D + K)*sizeof(float) <= SHARED_MEM_MAX.
 */
__global__ void kernel_accumulate_smem(const float* points,
                                       const int* cluster_of,
                                       float* new_centroids, int* counts, int N,
                                       int K, int D) {
  extern __shared__ char smem[];
  float* local_nc = (float*)smem;
  int* local_cnt = (int*)(smem + K * D * sizeof(float));

  int total_nc = K * D;
  int total_cnt = K;
  for (int t = threadIdx.x; t < total_nc; t += blockDim.x) local_nc[t] = 0.0f;
  for (int t = threadIdx.x; t < total_cnt; t += blockDim.x) local_cnt[t] = 0;
  __syncthreads();

  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < N) {
    int k = cluster_of[i];
    atomicAdd(&local_cnt[k], 1);
    for (int d = 0; d < D; d++)
      atomicAdd(&local_nc[k * D + d], points[i * D + d]);
  }
  __syncthreads();

  for (int t = threadIdx.x; t < total_nc; t += blockDim.x)
    atomicAdd(&new_centroids[t], local_nc[t]);
  for (int t = threadIdx.x; t < total_cnt; t += blockDim.x)
    atomicAdd(&counts[t], local_cnt[t]);
}

/**
 * kernel_accumulate_gmem -- global memory fallback
 *
 * Usato quando K*D e' troppo grande per la shared memory.
 * Ogni thread fa atomicAdd direttamente sulla global memory.
 */
__global__ void kernel_accumulate_gmem(const float* points,
                                       const int* cluster_of,
                                       float* new_centroids, int* counts, int N,
                                       int D) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= N) return;

  int k = cluster_of[i];
  atomicAdd(&counts[k], 1);
  for (int d = 0; d < D; d++)
    atomicAdd(&new_centroids[k * D + d], points[i * D + d]);
}

/**
 * kernel_divide
 *
 * Divide le somme accumulate per il numero di punti nel cluster,
 * ottenendo i nuovi centroidi. Se un cluster e' vuoto il centroide
 * rimane invariato.
 */
__global__ void kernel_divide(float* new_centroids, const int* counts,
                              float* centroids, int K, int D) {
  int k = blockIdx.x * blockDim.x + threadIdx.x;
  if (k >= K) return;

  if (counts[k] > 0) {
    for (int d = 0; d < D; d++)
      centroids[k * D + d] = new_centroids[k * D + d] / counts[k];
  }
  /* cluster vuoto non cambio nulla*/
}

/**
 * kernel_convergence
 *
 * Confronta i centroidi vecchi (old_centroids) con quelli nuovi
 * (centroids) e calcola lo spostamento quadratico per ciascun cluster.
 * Il massimo viene poi calcolato sull'host.
 */
__global__ void kernel_convergence(const float* old_centroids,
                                   const float* centroids, float* shifts, int K,
                                   int D) {
  int k = blockIdx.x * blockDim.x + threadIdx.x;
  if (k >= K) return;

  float s = 0.0f;
  for (int d = 0; d < D; d++) {
    float diff = old_centroids[k * D + d] - centroids[k * D + d];
    s += diff * diff;
  }
  shifts[k] = s;
}

/**
 * Legge il file di input: nessuna intestazione, ogni riga contiene
 * D valori float separati da spazi. Il numero di dimensioni viene
 * inferito dalla prima riga con strtof (stesso schema),
 * il numero di punti dal totale dei float letti.
 */
static float* read_input(const char* filename, int* n_points, int* n_dims) {
  FILE* f = fopen(filename, "r");
  if (!f) {
    perror(filename);
    exit(EXIT_FAILURE);
  }

  const size_t BUFLEN = 65536;
  char* buffer = (char*)malloc(BUFLEN);
  char* i_dont_care = fgets(buffer, BUFLEN, f);
  (void)i_dont_care;

  *n_dims = -1;
  char *start, *end = buffer;
  do {
    start = end;
    strtof(start, &end);
    (*n_dims)++;
  } while (end != start);

  rewind(f);
  int n_items = 0;
  float dummy;
  while (1 == fscanf(f, "%f", &dummy)) n_items++;

  *n_points = n_items / *n_dims;

  float* data = (float*)malloc((size_t)(*n_points) * (*n_dims) * sizeof(float));
  rewind(f);
  for (int i = 0; i < *n_points * *n_dims; i++) {
    if (fscanf(f, "%f", &data[i]) != 1) {
      fprintf(stderr, "Errore lettura punto %d\n", i);
      exit(EXIT_FAILURE);
    }
  }
  free(buffer);
  fclose(f);
  return data;
}

/**
 * Scrive il file di output nel formato identico a save_results()
 *   commenti con centroidi, poi una riga per punto con coordinate + cluster.
 */
static void write_output(const char* filename, const float* data,
                         const int* cluster_of, const float* centroids,
                         int n_points, int n_clusters, int n_dims) {
  FILE* f = fopen(filename, "w");
  if (!f) {
    perror(filename);
    exit(EXIT_FAILURE);
  }

  fprintf(f, "# Centroids:\n#\n");
  for (int j = 0; j < n_clusters; j++) {
    fprintf(f, "# %3d :", j);
    for (int d = 0; d < n_dims; d++)
      fprintf(f, " %f", centroids[j * n_dims + d]);
    fprintf(f, "\n");
  }
  fprintf(f, "#\n");

  for (int i = 0; i < n_points; i++) {
    for (int d = 0; d < n_dims; d++) fprintf(f, "%f ", data[i * n_dims + d]);
    fprintf(f, "%d\n", cluster_of[i]);
  }
  fclose(f);
}

/**
 * Inizializza i centroidi con lo stesso algoritmo reservoir
 * (srand(123), stesso seme usato in omp-k-means).
 */
static void init_centroids(const float* points, float* centroids, int n_points,
                           int n_clusters, int n_dims) {
  srand(123);
  int select = n_clusters;
  int remaining = n_points;
  for (int i = 0; (i < n_points) && (select > 0); i++) {
    if ((rand() % remaining) < select) {
      select--;
      memcpy(&centroids[select * n_dims], &points[i * n_dims],
             n_dims * sizeof(float));
    }
    remaining--;
  }
}

/**
 * Riduzione host: restituisce il massimo tra gli elementi di arr[].
 */
static float host_max(const float* arr, int n) {
  float m = 0.0f;
  for (int i = 0; i < n; i++)
    if (arr[i] > m) m = arr[i];
  return m;
}

int main(int argc, char** argv) {
  if (argc != 4) {
    return EXIT_FAILURE;
  }

  const int K = atoi(argv[1]);
  const char* infile = argv[2];
  const char* outfile = argv[3];

  if (K < 1) {
    return EXIT_FAILURE;
  }

  int N, D;
  float* h_points = read_input(infile, &N, &D);

  /* Calcola quanta shared memory serve per i due kernel. */
  size_t smem_classify = (size_t)K * D * sizeof(float);
  size_t smem_accumulate =
      (size_t)K * D * sizeof(float) + (size_t)K * sizeof(int);

  int use_smem_classify = (smem_classify <= SHARED_MEM_MAX);
  int use_smem_accumulate = (smem_accumulate <= SHARED_MEM_MAX);

  printf("\nInput file....... %s\n", infile);
  printf("Output file...... %s\n", outfile);
  printf("Data points (N).. %d\n", N);
  printf("Dimensions (D)... %d\n", D);
  printf("Clusters (K)..... %d\n", K);
  printf("classify path.... %s\n",
         use_smem_classify ? "shared memory" : "global memory");
  printf("accumulate path.. %s\n\n",
         use_smem_accumulate ? "shared memory" : "global memory");

  float* h_centroids = (float*)malloc((size_t)K * D * sizeof(float));
  int* h_cluster_of = (int*)malloc((size_t)N * sizeof(int));
  float* h_shifts = (float*)malloc((size_t)K * sizeof(float));

  init_centroids(h_points, h_centroids, N, K, D);

  float *d_points, *d_centroids, *d_old_centroids, *d_new_centroids, *d_shifts;
  int *d_cluster_of, *d_counts;

  CUDA_CHECK(cudaMalloc(&d_points, (size_t)N * D * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_centroids, (size_t)K * D * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_old_centroids, (size_t)K * D * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_new_centroids, (size_t)K * D * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_cluster_of, (size_t)N * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&d_counts, (size_t)K * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&d_shifts, (size_t)K * sizeof(float)));

  CUDA_CHECK(cudaMemcpy(d_points, h_points, (size_t)N * D * sizeof(float),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_centroids, h_centroids, (size_t)K * D * sizeof(float),
                        cudaMemcpyHostToDevice));

  int blocks_N = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
  int blocks_KD = (K * D + BLOCK_SIZE - 1) / BLOCK_SIZE;
  int blocks_K = (K + BLOCK_SIZE - 1) / BLOCK_SIZE;

  const float TOL = 1e-5f;
  const int MAXITER = 100;
  int iter = 0;

  printf("Main loop starts\n\n");

  cudaEvent_t t_start, t_stop;
  CUDA_CHECK(cudaEventCreate(&t_start));
  CUDA_CHECK(cudaEventCreate(&t_stop));
  CUDA_CHECK(cudaEventRecord(t_start));

  do {
    if (use_smem_classify) {
      kernel_classify_smem<<<blocks_N, BLOCK_SIZE, smem_classify>>>(
          d_points, d_centroids, d_cluster_of, N, K, D);
    } else {
      kernel_classify_gmem<<<blocks_N, BLOCK_SIZE>>>(d_points, d_centroids,
                                                     d_cluster_of, N, K, D);
    }
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaMemcpy(d_old_centroids, d_centroids,
                          (size_t)K * D * sizeof(float),
                          cudaMemcpyDeviceToDevice));

    kernel_reset<<<blocks_KD, BLOCK_SIZE>>>(d_new_centroids, d_counts, K, D);
    CUDA_CHECK(cudaGetLastError());

    if (use_smem_accumulate) {
      kernel_accumulate_smem<<<blocks_N, BLOCK_SIZE, smem_accumulate>>>(
          d_points, d_cluster_of, d_new_centroids, d_counts, N, K, D);
    } else {
      kernel_accumulate_gmem<<<blocks_N, BLOCK_SIZE>>>(
          d_points, d_cluster_of, d_new_centroids, d_counts, N, D);
    }
    CUDA_CHECK(cudaGetLastError());

    kernel_divide<<<blocks_K, BLOCK_SIZE>>>(d_new_centroids, d_counts,
                                            d_centroids, K, D);
    CUDA_CHECK(cudaGetLastError());

    kernel_convergence<<<blocks_K, BLOCK_SIZE>>>(d_old_centroids, d_centroids,
                                                 d_shifts, K, D);
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaMemcpy(h_shifts, d_shifts, (size_t)K * sizeof(float),
                          cudaMemcpyDeviceToHost));
    float maxshift = host_max(h_shifts, K);

    CUDA_CHECK(cudaMemcpy(h_centroids, d_centroids,
                          (size_t)K * D * sizeof(float),
                          cudaMemcpyDeviceToHost));

    printf("Iteration %3d, maxsqshift = %f\n", iter, maxshift);
    iter++;

    if (maxshift <= TOL * TOL) break;

  } while (iter <= MAXITER);

  CUDA_CHECK(cudaDeviceSynchronize());
  CUDA_CHECK(cudaEventRecord(t_stop));
  CUDA_CHECK(cudaEventSynchronize(t_stop));

  float elapsed_ms;
  CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, t_start, t_stop));

  printf("\nMain loop completed\n");
  printf("Elapsed time %.3f ms\n\n", elapsed_ms);

  CUDA_CHECK(cudaMemcpy(h_cluster_of, d_cluster_of, (size_t)N * sizeof(int),
                        cudaMemcpyDeviceToHost));

  write_output(outfile, h_points, h_cluster_of, h_centroids, N, K, D);

  cudaFree(d_points);
  cudaFree(d_centroids);
  cudaFree(d_old_centroids);
  cudaFree(d_new_centroids);
  cudaFree(d_cluster_of);
  cudaFree(d_counts);
  cudaFree(d_shifts);

  free(h_points);
  free(h_centroids);
  free(h_cluster_of);
  free(h_shifts);

  return EXIT_SUCCESS;
}
